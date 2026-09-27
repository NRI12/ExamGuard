"""X3D-L cheating classifier of cheating_detection_v1 (4 classes, 16 RGB frames 0.2 s apart).

Preprocessing is identical to the training/eval code (square crop 356 px INTER_AREA, short-side
resize to 356, centre crop 312, Kinetics normalisation, bf16 autocast). X3D is mostly depthwise 3D
convolutions and elementwise ops (memory bound): torch.compile fuses them (RTX 3060, torch 2.7.1,
6 actors: 305 -> 245 ms per window). Batches are padded to a fixed size so the model compiles once,
when it is loaded; without Triton it falls back to eager mode.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.monitoring.config import CheatingClassifierConfig

logger = logging.getLogger(__name__)

X3D_VARIANTS = {"x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}  # frames, crop px
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


def square_crop_box(
    boxes: list[np.ndarray], width: int, height: int, pad: float
) -> tuple[int, int, int, int]:
    """One fixed square covering the union of an actor's boxes over the clip (no jitter)."""
    b = np.stack(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x2 - x1, y2 - y1) * (1 + pad), width, height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return left, top, left + int(side), top + int(side)


def map_box(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    config: CheatingClassifierConfig,
) -> np.ndarray:
    """YOLO full-body box -> 'upper body + desk' box the classifier was trained on."""
    x1, y1, x2, _ = box_xyxy
    w = x2 - x1
    bm = config.box_map
    top = y1 + bm.dy1 * w
    return np.clip(
        [x1 + bm.dx1 * w, top, x2 + bm.dx2 * w, top + bm.h * w], 0, [width, height, width, height]
    )


class CheatClassifierModel:
    """Loaded once per process and shared by every monitoring runtime (GPU calls serialised)."""

    def __init__(self, config: CheatingClassifierConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.path = Path(config.model)
        self.classes: tuple[str, ...] = ()
        self.num_frames = 16
        self.input_size = 312
        self.model_name = self.path.stem
        self._lock = threading.Lock()
        self._model: Any = None
        self._bf16 = False

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is None:
                self._load()

    def _load(self) -> None:
        import torch
        import torch.nn as nn
        from pytorchvideo.models import hub  # type: ignore[import-untyped]

        if not self.path.is_file():
            raise FileNotFoundError(f"X3D checkpoint not found: {self.path}")
        started = time.monotonic()
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        info = checkpoint.get("info") or {}
        name = info.get("name") or checkpoint["args"]["model"]
        self.num_frames, self.input_size = X3D_VARIANTS[name]
        self.classes = tuple(checkpoint["classes"])
        model = getattr(hub, name)(pretrained=False)
        model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, len(self.classes))
        model.load_state_dict(checkpoint["model"])
        model = model.to(self.device).eval()
        cuda = self.device.startswith("cuda")
        self._bf16 = cuda and torch.cuda.is_bf16_supported()
        if cuda and self.config.compile:
            # X3D graph-breaks once per ResStage: keep every stage compiled
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            model = torch.compile(model)
        self._model = model
        size = self.config.clip_size
        try:
            self._classify_locked(np.zeros((1, self.num_frames, size, size, 3), np.uint8))
        except Exception:  # e.g. Windows without Triton / a C++ compiler: same model, eager mode
            if not (cuda and self.config.compile):
                raise
            logger.warning("torch.compile unavailable, X3D runs in eager mode", exc_info=True)
            self._model = model._orig_mod
            self._classify_locked(np.zeros((1, self.num_frames, size, size, 3), np.uint8))
        logger.info(
            "X3D cheating classifier %s loaded on %s in %.1fs (classes=%s bf16=%s compiled=%s)",
            self.path.name,
            self.device,
            time.monotonic() - started,
            self.classes,
            self._bf16,
            cuda and self.config.compile,
        )

    def classify(self, clips: np.ndarray) -> np.ndarray:
        """uint8 RGB clips [B, T, S, S, 3] -> softmax probabilities [B, C]."""
        self.ensure_loaded()
        with self._lock:
            return self._classify_locked(clips)

    def _classify_locked(self, clips: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            x = torch.from_numpy(clips).to(self.device)
            b, t, h, w, _ = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255)
            resize = int(round(self.input_size * 256 / 224))
            if min(h, w) != resize:
                scale = resize / min(h, w)
                x = F.interpolate(
                    x,
                    size=(round(h * scale), round(w * scale)),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            hh, ww = x.shape[-2:]
            y0, x0 = (hh - self.input_size) // 2, (ww - self.input_size) // 2
            size = self.input_size
            x = x[:, :, y0 : y0 + size, x0 : x0 + size].reshape(b, t, 3, size, size)
            x = x.permute(0, 2, 1, 3, 4)
            mean = torch.tensor(KINETICS_MEAN, device=x.device).view(1, 3, 1, 1, 1)
            std = torch.tensor(KINETICS_STD, device=x.device).view(1, 3, 1, 1, 1)
            x = ((x - mean) / std).contiguous()
            batch = self.config.batch_size
            pad = (-b) % batch
            if pad:
                x = torch.cat([x, x[:1].expand(pad, *x.shape[1:])]).contiguous()
            outputs = []
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._bf16):
                for start in range(0, b + pad, batch):
                    outputs.append(self._model(x[start : start + batch].contiguous()))
            logits = torch.cat(outputs)[:b].float()
            return logits.softmax(-1).cpu().numpy()


class CheatModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str], CheatClassifierModel] = {}

    def get(self, config: CheatingClassifierConfig, device: str) -> CheatClassifierModel:
        key = (str(Path(config.model).resolve()), device)
        with self._lock:
            return self._models.setdefault(key, CheatClassifierModel(config, device))
