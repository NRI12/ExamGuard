"""X3D (pytorchvideo, Kinetics-400) with a new classification head."""
import torch.nn as nn

VARIANTS = {"x3d_xs": (4, 160), "x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}  # frames, crop px


def build_x3d(name, num_classes, dropout=0.5, pretrained=True):
    from pytorchvideo.models import hub
    m = getattr(hub, name)(pretrained=pretrained)
    head = m.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout = nn.Dropout(dropout)
    return m
