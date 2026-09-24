"""One place that knows each architecture: how to build it, which frames it reads, which normalisation it uses.

info = dict(arch, name, frames, size, sampling, stride, sample_dt, norm, segments, shift_div)
  sampling "dense"  : `frames` frames `stride` apart (X3D)
  sampling "segment": `frames` segments over the clip, one frame each (TSM)
  sample_dt         : seconds between the frames the model sees (what the real-time app samples at)
"""
import torch

from .tsm import build_tsm
from .x3d import VARIANTS, build_x3d


def model_info(model_cfg, data_cfg):
    name, fps = model_cfg["name"], data_cfg["fps"]
    if name.startswith("tsm"):
        seg = model_cfg["segments"]
        return dict(arch="tsm", name=name, frames=seg, size=model_cfg.get("size", 224), sampling="segment",
                    stride=None, sample_dt=data_cfg["clip_len"] / seg / fps, norm="imagenet", segments=seg,
                    shift_div=model_cfg.get("shift_div", 8))
    frames, size = VARIANTS[name]
    return dict(arch="x3d", name=name, frames=frames, size=size, sampling="dense", stride=model_cfg["stride"],
                sample_dt=model_cfg["stride"] / fps, norm="kinetics", segments=None, shift_div=None)


def window_frames(info, fps):
    """Span of one model input, in video frames (first to last sampled frame)."""
    return int(round((info["frames"] - 1) * info["sample_dt"] * fps))


def build_model(cfg, num_classes, pretrained=True):
    info = model_info(cfg["model"], cfg["data"])
    if info["arch"] == "tsm":
        m = build_tsm(num_classes, info["segments"], info["shift_div"], cfg["model"]["dropout"],
                      cfg["paths"].get("pretrained") if pretrained else None)
        return m.to(memory_format=torch.channels_last), info
    return build_x3d(info["name"], num_classes, cfg["model"]["dropout"], pretrained), info


def head_parameters(model, info):
    return list((model.fc if info["arch"] == "tsm" else model.blocks[-1].proj).parameters())


def save_checkpoint(path, model, classes, cfg, info, epoch, metrics):
    torch.save(dict(model=model.state_dict(), classes=classes, info=info, epoch=epoch, metrics=metrics, config=cfg,
                    args=dict(model=info["name"], stride=info["stride"])), path)


def _legacy_info(ck, data_cfg):
    """Checkpoints written by the original experiment scripts (no 'info' key)."""
    a = ck["args"]
    if "segments" in a:  # train_tsm.py
        return model_info(dict(name="tsm_r50", segments=a["segments"], size=a["size"], shift_div=a["shift_div"]),
                          data_cfg)
    return model_info(dict(name=a["model"], stride=a["stride"]), data_cfg)


def load_model(path, data_cfg, device="cuda"):
    """-> (model in eval mode, checkpoint dict, info)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = ck.get("info") or _legacy_info(ck, data_cfg)
    nc = len(ck["classes"])
    if info["arch"] == "tsm":
        m = build_tsm(nc, info["segments"], info["shift_div"]).to(memory_format=torch.channels_last)
    else:
        m = build_x3d(info["name"], nc, pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck, info
