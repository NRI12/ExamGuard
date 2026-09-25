"""Optical-flow stream and RGB + flow fusion.

Flow stream: X3D-S (pytorchvideo, Kinetics-400) with a 2-channel stem (RGB filters averaged, cross-modality
initialisation) and an adaptive head pool, so it takes 16 flow frames at 160 px.
Fusion: pooled 2048-d features of the RGB X3D-L and of the flow X3D-S, concatenated -> dropout -> linear.
The fusion layer starts as the average of the two single-stream heads, i.e. exactly late fusion of logits,
and is then fine-tuned together with both streams.
"""
import torch
import torch.nn as nn

from .x3d import build_x3d


def build_flow_x3d(num_classes, name="x3d_s", dropout=0.5, pretrained=True):
    m = build_x3d(name, num_classes, dropout, pretrained)
    conv = m.blocks[0].conv.conv_t  # stem spatial conv, in_channels = 3
    new = nn.Conv3d(2, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1, 1))
    m.blocks[0].conv.conv_t = new
    m.blocks[-1].pool.pool = nn.AdaptiveAvgPool3d(1)  # any T / H / W
    return m


def x3d_features(m, x):
    """Pooled 2048-d features of an X3D (everything before the head's dropout / projection)."""
    for block in m.blocks[:-1]:
        x = block(x)
    return m.blocks[-1].pool(x).flatten(1)


class FusionNet(nn.Module):
    """forward(rgb [B,3,T,H,W], flow [B,2,T,h,w]) -> logits."""

    def __init__(self, rgb, flow, num_classes, dropout=0.5):
        super().__init__()
        self.rgb, self.flow = rgb, flow
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * 2048, num_classes)

    def init_as_late_fusion(self):
        """fc = 0.5 * rgb head + 0.5 * flow head: identical to averaging the two streams' logits."""
        hr, hf = self.rgb.blocks[-1].proj, self.flow.blocks[-1].proj
        with torch.no_grad():
            self.fc.weight.copy_(torch.cat([hr.weight, hf.weight], 1) * 0.5)
            self.fc.bias.copy_((hr.bias + hf.bias) * 0.5)

    def forward(self, rgb, flow):
        f = torch.cat([x3d_features(self.rgb, rgb), x3d_features(self.flow, flow)], 1)
        return self.fc(self.dropout(f))
