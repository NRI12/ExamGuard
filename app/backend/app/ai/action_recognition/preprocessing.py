from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as functional

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class PreprocessingProfile:
    tensor_assembly_ms: float
    resize_ms: float
    normalization_ms: float


def sample_segment_indices(frame_count: int, num_segments: int = 8) -> np.ndarray:
    """Exact deterministic evaluation sampling used by dataset_tsm_reference.py."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if frame_count < num_segments:
        return np.linspace(0, frame_count - 1, num_segments).round().astype(np.int64)
    boundaries = np.linspace(0, frame_count, num_segments + 1, dtype=np.int64)
    indices = []
    for index in range(num_segments):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        indices.append(min(frame_count - 1, (start + max(start + 1, end) - 1) // 2))
    return np.asarray(indices, dtype=np.int64)


def preprocess_rgb_clips(
    clips: list[tuple[np.ndarray, ...]],
    *,
    input_size: int = 224,
    num_segments: int = 8,
) -> torch.Tensor:
    tensor, _ = preprocess_rgb_clips_profiled(
        clips,
        input_size=input_size,
        num_segments=num_segments,
    )
    return tensor


def preprocess_rgb_clips_profiled(
    clips: list[tuple[np.ndarray, ...]],
    *,
    input_size: int = 224,
    num_segments: int = 8,
) -> tuple[torch.Tensor, PreprocessingProfile]:
    if not clips:
        raise ValueError("at least one clip is required")
    batches: list[torch.Tensor] = []
    tensor_assembly_ms = 0.0
    resize_ms = 0.0
    normalization_ms = 0.0
    for frames in clips:
        if len(frames) != num_segments:
            raise ValueError(f"expected {num_segments} sampled frames, got {len(frames)}")
        started = time.perf_counter()
        array = np.stack(frames)
        if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("RGB frames must be uint8 [T,H,W,3]")
        tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        tensor_assembly_ms += (time.perf_counter() - started) * 1000
        if tensor.shape[-2:] != (input_size, input_size):
            started = time.perf_counter()
            tensor = functional.interpolate(
                tensor,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            resize_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        batches.append((tensor - mean) / std)
        normalization_ms += (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    batch = torch.stack(batches, dim=0)
    tensor_assembly_ms += (time.perf_counter() - started) * 1000
    return batch, PreprocessingProfile(
        tensor_assembly_ms=tensor_assembly_ms,
        resize_ms=resize_ms,
        normalization_ms=normalization_ms,
    )
