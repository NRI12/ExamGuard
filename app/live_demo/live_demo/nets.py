"""Networks and preprocessing of the X3D cheating classifiers, copied from cheating_detection_v1 / v3 so the live
service reproduces their eval-videos logs exactly (same crops, resize, normalisation, RAFT flow convention).

  rgb    : X3D-L on 16 RGB frames 0.2 s apart (v1, v2, v1 variant A)
  fusion : RGB X3D-L + X3D-S on RAFT-small optical flow (v3 fine-tuned fusion, v3 late fusion)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {"x3d_xs": (4, 160), "x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}  # frames, crop px
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
Q_SCALE, Q_ZERO, INPUT_DIV = 4.0, 128, 8.0  # flow convention of cheating_v3.models.raft_flow


def build_x3d(name, num_classes):
    from pytorchvideo.models import hub
    m = getattr(hub, name)(pretrained=False)
    m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, num_classes)
    return m


def build_flow_x3d(num_classes, name="x3d_s"):
    m = build_x3d(name, num_classes)
    conv = m.blocks[0].conv.conv_t
    m.blocks[0].conv.conv_t = nn.Conv3d(2, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                                        bias=conv.bias is not None)
    m.blocks[-1].pool.pool = nn.AdaptiveAvgPool3d(1)
    return m


def x3d_features(m, x):
    for block in m.blocks[:-1]:
        x = block(x)
    return m.blocks[-1].pool(x).flatten(1)


class FusionNet(nn.Module):
    def __init__(self, rgb, flow, num_classes):
        super().__init__()
        self.rgb, self.flow = rgb, flow
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * 2048, num_classes)

    def forward(self, rgb, flow):
        return self.fc(self.dropout(torch.cat([x3d_features(self.rgb, rgb), x3d_features(self.flow, flow)], 1)))


def load_classifier(path, device):
    """-> (model in eval mode, classes, info). info = dict(arch 'rgb' | 'fusion', frames, sample_dt, rgb_size,
    flow_size). Reads the v1 legacy checkpoints (no 'info'), v1 registry checkpoints and v3 checkpoints."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes, nc, info = ck["classes"], len(ck["classes"]), ck.get("info")
    if info is None or info.get("arch") == "x3d":  # v1: legacy (args only) or registry format
        name = info["name"] if info else ck["args"]["model"]
        stride = info["stride"] if info else ck["args"]["stride"]
        info = dict(arch="rgb", frames=VARIANTS[name][0], sample_dt=stride / 25, rgb_name=name,
                    rgb_size=VARIANTS[name][1], flow_size=None)
    if info["arch"] == "rgb":
        m = build_x3d(info["rgb_name"], nc)
    elif info["arch"] == "fusion":
        m = FusionNet(build_x3d(info["rgb_name"], nc), build_flow_x3d(nc, info["flow_name"]), nc)
    else:
        raise ValueError(f"{path}: unsupported arch {info['arch']}")
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), classes, info


def gpu_eval_transform(frames, size):
    """RGB uint8 [B, T, H, W, 3] -> float [B, 3, T, size, size]: short-side resize (256/224 ratio) + centre crop."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255)
    resize = int(round(size * 256 / 224))
    if min(h, w) != resize:
        s = resize / min(h, w)
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    hh, ww = x.shape[-2:]
    y0, x0 = (hh - size) // 2, (ww - size) // 2
    x = x[:, :, y0:y0 + size, x0:x0 + size].reshape(b, t, 3, size, size).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(KINETICS_MEAN, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def square_crop_box(boxes, w, h, pad):
    """One fixed square covering the union of boxes over the clip (no jitter)."""
    b = np.stack(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x2 - x1, y2 - y1) * (1 + pad), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return x1, y1, x1 + int(side), y1 + int(side)


def map_box(b, w_img, h_img, bm):
    """YOLO full-body box -> CVAT-style 'upper body + desk' box the classifiers were trained on."""
    x1, y1, x2, _ = b
    w = x2 - x1
    ny1 = y1 + bm["dy1"] * w
    return np.clip([x1 + bm["dx1"] * w, ny1, x2 + bm["dx2"] * w, ny1 + bm["h"] * w], 0, [w_img, h_img, w_img, h_img])


class FlowEstimator:
    """RAFT-small (torchvision) -> flow in pixels, a -> b."""

    def __init__(self, device, iters=12):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.device, self.iters = device, iters

    @torch.no_grad()
    def __call__(self, a, b):
        a, b = a.to(self.device).float() / 127.5 - 1, b.to(self.device).float() / 127.5 - 1
        h, w = a.shape[-2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
        return self.model(a, b, num_flow_updates=self.iters)[-1][..., :h, :w]


def flow_to_input(flow_px):
    return flow_px.clamp(-Q_ZERO / Q_SCALE, (255 - Q_ZERO) / Q_SCALE) / INPUT_DIV


def resize_flow(flow, size):
    """[N, 2, H, W] pixels -> [N, 2, size, size], vectors rescaled to the new resolution."""
    h, w = flow.shape[-2:]
    out = F.interpolate(flow, size=(size, size), mode="bilinear", align_corners=False)
    out[:, 0] *= size / w
    out[:, 1] *= size / h
    return out
