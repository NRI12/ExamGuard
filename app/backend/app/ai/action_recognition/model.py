from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet50  # type: ignore[import-untyped]


class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, num_segments: int = 8, fold_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.fold_div = fold_div

    @staticmethod
    def shift(x: torch.Tensor, num_segments: int, fold_div: int = 8) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B*T,C,H,W], got {tuple(x.shape)}")
        nt, channels, height, width = x.shape
        if nt % num_segments:
            raise RuntimeError(f"B*T={nt} is not divisible by num_segments={num_segments}")
        batch, fold = nt // num_segments, channels // fold_div
        if fold <= 0:
            return x
        temporal = x.view(batch, num_segments, channels, height, width)
        shifted = torch.zeros_like(temporal)
        shifted[:, :-1, :fold] = temporal[:, 1:, :fold]
        shifted[:, 1:, fold : 2 * fold] = temporal[:, :-1, fold : 2 * fold]
        shifted[:, :, 2 * fold :] = temporal[:, :, 2 * fold :]
        return shifted.view(nt, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.shift(x, self.num_segments, self.fold_div))


def inject_blockres_tsm(backbone: nn.Module, num_segments: int, fold_div: int = 8) -> int:
    wrapped = 0
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(backbone, layer_name):
            block.conv1 = TemporalShift(block.conv1, num_segments, fold_div)
            wrapped += 1
    return wrapped


class KineticsTSMClassifier(nn.Module):
    """Exact R3 TSM-ResNet50 inference architecture."""

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 8,
        fold_div: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.fold_div = fold_div
        self.base_model = resnet50(weights=None)
        if inject_blockres_tsm(self.base_model, num_segments, fold_div) != 16:
            raise RuntimeError("R3 requires exactly 16 TSM-wrapped ResNet50 blocks")
        feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Dropout(p=dropout)
        self.new_fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        batch, segments, channels, height, width = x.shape
        if segments != self.num_segments:
            raise ValueError(f"Expected T={self.num_segments}, got T={segments}")
        features = self.base_model(x.reshape(batch * segments, channels, height, width))
        return self.new_fc(features).view(batch, segments, self.num_classes).mean(dim=1)
