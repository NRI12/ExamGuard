"""Everything the pipeline needs to know about an architecture.

arch "rgb"   : X3D on RGB frames (the v1 model)
arch "flow"  : X3D-S on optical flow
arch "fusion": RGB X3D + flow X3D-S, feature fusion

info = dict(arch, frames, stride, sample_dt, rgb_name, rgb_size, flow_name, flow_size)
All streams read `frames` frames `stride` apart (16 x 0.2 s); flow frame f is the flow from frame f - stride to f.
"""
import torch

from .two_stream import FusionNet, build_flow_x3d
from .x3d import VARIANTS, build_x3d


def model_info(cfg):
    m, fps = cfg["model"], cfg["data"]["fps"]
    arch = m["arch"]
    frames = VARIANTS[m["rgb_name"]][0] if arch != "flow" else m["flow_frames"]
    return dict(arch=arch, frames=frames, stride=m["stride"], sample_dt=m["stride"] / fps,
                rgb_name=m.get("rgb_name") if arch != "flow" else None,
                rgb_size=VARIANTS[m["rgb_name"]][1] if arch != "flow" else None,
                flow_name=m.get("flow_name") if arch != "rgb" else None,
                flow_size=m.get("flow_size") if arch != "rgb" else None)


def uses_rgb(info):
    return info["arch"] in ("rgb", "fusion")


def uses_flow(info):
    return info["arch"] in ("flow", "fusion")


def window_frames(info, fps):
    return int(round((info["frames"] - 1) * info["sample_dt"] * fps))


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def build_model(cfg, num_classes, pretrained=True):
    info, m, p = model_info(cfg), cfg["model"], cfg["paths"]
    if info["arch"] == "rgb":
        return build_x3d(m["rgb_name"], num_classes, m["dropout"], pretrained), info
    if info["arch"] == "flow":
        return build_flow_x3d(num_classes, m["flow_name"], m["dropout"], pretrained), info
    rgb = build_x3d(m["rgb_name"], num_classes, m["dropout"], pretrained=False)
    flow = build_flow_x3d(num_classes, m["flow_name"], m["dropout"], pretrained=False)
    if pretrained:  # the two single-stream models already trained on this task
        rgb.load_state_dict(_state(p["rgb_init"]))
        flow.load_state_dict(_state(p["flow_init"]))
    net = FusionNet(rgb, flow, num_classes, m["dropout"])
    if pretrained:
        net.init_as_late_fusion()
    return net, info


def head_parameters(model, info):
    if info["arch"] == "fusion":
        return list(model.fc.parameters())
    return list(model.blocks[-1].proj.parameters())


def forward(model, info, rgb, flow):
    if info["arch"] == "rgb":
        return model(rgb)
    if info["arch"] == "flow":
        return model(flow)
    return model(rgb, flow)


def save_checkpoint(path, model, classes, cfg, info, epoch, metrics):
    torch.save(dict(model=model.state_dict(), classes=classes, info=info, epoch=epoch, metrics=metrics, config=cfg,
                    args=dict(model=info["rgb_name"] or info["flow_name"], stride=info["stride"])), path)


def load_model(path, data_cfg=None, device="cuda"):
    """-> (model in eval mode, checkpoint, info). Also reads v1-style RGB checkpoints (no 'info')."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nc = len(ck["classes"])
    info = ck.get("info")
    if info is None:
        name, stride = ck["args"]["model"], ck["args"]["stride"]
        info = dict(arch="rgb", frames=VARIANTS[name][0], stride=stride,
                    sample_dt=stride / (data_cfg or {}).get("fps", 25), rgb_name=name, rgb_size=VARIANTS[name][1],
                    flow_name=None, flow_size=None)
    if info["arch"] == "rgb":
        m = build_x3d(info["rgb_name"], nc, pretrained=False)
    elif info["arch"] == "flow":
        m = build_flow_x3d(nc, info["flow_name"], pretrained=False)
    else:
        m = FusionNet(build_x3d(info["rgb_name"], nc, pretrained=False),
                      build_flow_x3d(nc, info["flow_name"], pretrained=False), nc)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck, info
