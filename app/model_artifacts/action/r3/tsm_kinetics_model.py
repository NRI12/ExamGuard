from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class TemporalShift(nn.Module):
    """Official-style offline TSM wrapper for a residual-branch convolution."""

    def __init__(
        self,
        net: nn.Module,
        num_segments: int = 8,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.net = net
        self.num_segments = int(num_segments)
        self.fold_div = int(fold_div)

    @staticmethod
    def shift(
        x: torch.Tensor,
        num_segments: int,
        fold_div: int = 8,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B*T,C,H,W], got {tuple(x.shape)}"
            )

        nt, channels, height, width = x.shape

        if nt % num_segments != 0:
            raise RuntimeError(
                f"B*T={nt} is not divisible by num_segments={num_segments}"
            )

        batch = nt // num_segments
        fold = channels // fold_div

        if fold <= 0:
            return x

        x = x.view(
            batch,
            num_segments,
            channels,
            height,
            width,
        )

        out = torch.zeros_like(x)

        # Shift one channel group toward the previous time index.
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Shift the second channel group toward the next time index.
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        # Keep the remaining channels unchanged.
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(
            nt,
            channels,
            height,
            width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(
            self.shift(
                x,
                num_segments=self.num_segments,
                fold_div=self.fold_div,
            )
        )


def inject_blockres_tsm(
    backbone: nn.Module,
    num_segments: int,
    fold_div: int = 8,
) -> int:
    """
    Match the official TSM `shift_place=blockres` behavior for ResNet-50.

    The official implementation wraps block.conv1. For ResNet-50, layer3
    has fewer than 23 blocks, so n_round=1 and every residual block is wrapped.
    """
    wrapped = 0

    for layer_name in (
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ):
        layer = getattr(
            backbone,
            layer_name,
        )

        for block in layer:
            block.conv1 = TemporalShift(
                block.conv1,
                num_segments=num_segments,
                fold_div=fold_div,
            )
            wrapped += 1

    return wrapped


class KineticsTSMClassifier(nn.Module):
    """
    TSM-ResNet50 compatible with the official Kinetics checkpoint layout.

    Important naming:
      - `base_model` matches the official TSN implementation.
      - `new_fc` matches the official classifier name.

    The Kinetics classifier is intentionally NOT loaded when transferring to
    the 5-class exam dataset.
    """

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 8,
        fold_div: int = 8,
        dropout: float = 0.6,
        imagenet_init: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_segments = int(num_segments)
        self.fold_div = int(fold_div)

        weights = (
            ResNet50_Weights.DEFAULT
            if imagenet_init
            else None
        )

        self.base_model = resnet50(
            weights=weights
        )

        wrapped = inject_blockres_tsm(
            self.base_model,
            num_segments=self.num_segments,
            fold_div=self.fold_div,
        )

        if wrapped != 16:
            raise RuntimeError(
                f"Expected 16 TSM-wrapped ResNet50 blocks, found {wrapped}"
            )

        feature_dim = (
            self.base_model.fc.in_features
        )

        # Official TSN/TSM uses the ResNet `fc` slot as dropout and then a
        # separate `new_fc` classifier when dropout > 0.
        self.base_model.fc = nn.Dropout(
            p=float(dropout)
        )

        self.new_fc = nn.Linear(
            feature_dim,
            self.num_classes,
        )

        nn.init.normal_(
            self.new_fc.weight,
            mean=0.0,
            std=0.001,
        )
        nn.init.constant_(
            self.new_fc.bias,
            0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(x.shape)}"
            )

        (
            batch,
            segments,
            channels,
            height,
            width,
        ) = x.shape

        if segments != self.num_segments:
            raise ValueError(
                f"Expected T={self.num_segments}, got T={segments}"
            )

        x = x.reshape(
            batch * segments,
            channels,
            height,
            width,
        )

        features = self.base_model(x)
        segment_logits = self.new_fc(
            features
        )

        segment_logits = segment_logits.view(
            batch,
            segments,
            self.num_classes,
        )

        return segment_logits.mean(dim=1)


def _safe_torch_load(
    checkpoint_path: str | Path,
) -> Any:
    """
    Load tensor-only/primitive checkpoints safely on modern PyTorch.
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older torch versions that do not expose
        # `weights_only`.
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


def _normalize_official_key(
    key: str,
) -> str:
    # Official checkpoints are usually saved from DataParallel.
    if key.startswith("module."):
        key = key[len("module."):]
    return key


def load_official_kinetics_tsm(
    model: KineticsTSMClassifier,
    checkpoint_path: str | Path,
    minimum_backbone_coverage: float = 0.98,
) -> dict[str, Any]:
    """
    Transfer the official Kinetics-400 TSM-ResNet50 T=8 blockres checkpoint.

    Loads:
      - base_model convolution weights
      - BatchNorm parameters/buffers
      - TSM-wrapped conv1 weights (`.net.weight`)

    Does NOT load:
      - Kinetics `new_fc` (400 classes)

    A high backbone coverage threshold protects against silently loading an
    incompatible checkpoint.
    """
    checkpoint = _safe_torch_load(
        checkpoint_path
    )

    if isinstance(
        checkpoint,
        dict,
    ) and "state_dict" in checkpoint:
        source_state = checkpoint[
            "state_dict"
        ]
    elif isinstance(
        checkpoint,
        dict,
    ):
        source_state = checkpoint
    else:
        raise RuntimeError(
            "Unsupported checkpoint format"
        )

    source_state = {
        _normalize_official_key(
            str(key)
        ): value
        for key, value
        in source_state.items()
        if torch.is_tensor(value)
    }

    target_state = model.state_dict()

    transferable: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    skipped_head: list[str] = []

    for key, value in source_state.items():
        if key.startswith("new_fc."):
            skipped_head.append(key)
            continue

        # The transfer target is the pretrained video backbone.
        if not key.startswith("base_model."):
            continue

        target_value = target_state.get(
            key
        )

        if target_value is None:
            continue

        if tuple(
            target_value.shape
        ) != tuple(
            value.shape
        ):
            shape_mismatch.append(
                key
            )
            continue

        transferable[key] = value

    target_backbone_keys = [
        key
        for key
        in target_state
        if key.startswith("base_model.")
    ]

    # Coverage by tensor elements is more informative than raw key count.
    target_numel = sum(
        target_state[key].numel()
        for key in target_backbone_keys
    )

    loaded_numel = sum(
        target_state[key].numel()
        for key in transferable
    )

    coverage = (
        loaded_numel
        / max(target_numel, 1)
    )

    if coverage < float(
        minimum_backbone_coverage
    ):
        missing_examples = [
            key
            for key
            in target_backbone_keys
            if key not in transferable
        ][:20]

        raise RuntimeError(
            "Kinetics checkpoint compatibility check failed. "
            f"Backbone coverage={coverage:.4%}, "
            f"required>={minimum_backbone_coverage:.2%}. "
            f"Example missing keys={missing_examples}; "
            f"shape mismatch={shape_mismatch[:10]}"
        )

    incompatible = model.load_state_dict(
        transferable,
        strict=False,
    )

    report = {
        "checkpoint": str(
            checkpoint_path
        ),
        "source_tensor_keys": len(
            source_state
        ),
        "loaded_tensor_keys": len(
            transferable
        ),
        "backbone_coverage": coverage,
        "skipped_kinetics_head": sorted(
            skipped_head
        ),
        "shape_mismatch": sorted(
            shape_mismatch
        ),
        "missing_after_partial_load": list(
            incompatible.missing_keys
        ),
        "unexpected_after_partial_load": list(
            incompatible.unexpected_keys
        ),
    }

    return report


def count_tsm_modules(
    model: nn.Module,
) -> int:
    return sum(
        isinstance(
            module,
            TemporalShift,
        )
        for module
        in model.modules()
    )
