"""X3D (pytorchvideo) with a new classification head; checkpoint save / load."""
import torch
import torch.nn as nn

VARIANTS = {"x3d_xs": (4, 160), "x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}  # frames, crop px


def build_x3d(name, num_classes, dropout=0.5, pretrained=True):
    from pytorchvideo.models import hub
    m = getattr(hub, name)(pretrained=pretrained)  # Kinetics-400 weights
    head = m.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout = nn.Dropout(dropout)
    return m


def save_checkpoint(path, model, classes, cfg, epoch, metrics):
    torch.save(dict(model=model.state_dict(), classes=classes, epoch=epoch, metrics=metrics,
                    args=dict(model=cfg["model"]["name"], stride=cfg["model"]["stride"]), config=cfg), path)


def load_x3d(path, device="cuda"):
    """-> (model in eval mode, checkpoint dict with 'classes' and 'args': {model, stride})."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_x3d(ck["args"]["model"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
