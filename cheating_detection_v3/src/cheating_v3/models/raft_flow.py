"""Optical flow with RAFT-small (torchvision, pretrained on FlyingChairs / Things / Sintel / KITTI / HD1K).

Flow is stored and fed to the flow stream in one convention everywhere (clip pre-computation and real-time app):
  pixels at the flow stream resolution (flow_size, 160 px);  stored uint8 q = round(4 * px) + 128  (+-32 px)
  network input = px / 8 = (q - 128) / 32
"""
import torch
import torch.nn.functional as F

Q_SCALE, Q_ZERO, INPUT_DIV = 4.0, 128, 8.0


def quantize(flow_px):
    return (flow_px * Q_SCALE + Q_ZERO).round().clamp(0, 255).to(torch.uint8)


def flow_to_input(flow_px):
    return flow_px.clamp(-Q_ZERO / Q_SCALE, (255 - Q_ZERO) / Q_SCALE) / INPUT_DIV


class FlowEstimator:
    def __init__(self, device="cuda", iters=12):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.device, self.iters = device, iters

    @torch.no_grad()
    def __call__(self, a, b):
        """a, b: [N, 3, H, W] RGB in 0..255 (uint8 or float) -> flow [N, 2, H, W] in pixels (a -> b)."""
        a, b = a.to(self.device).float() / 127.5 - 1, b.to(self.device).float() / 127.5 - 1
        h, w = a.shape[-2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
        return self.model(a, b, num_flow_updates=self.iters)[-1][..., :h, :w]


def resize_flow(flow, size):
    """[N, 2, H, W] pixels -> [N, 2, size, size], vectors rescaled to the new resolution."""
    h, w = flow.shape[-2:]
    out = F.interpolate(flow, size=(size, size), mode="bilinear", align_corners=False)
    out[:, 0] *= size / w
    out[:, 1] *= size / h
    return out
