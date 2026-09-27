from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def resolve_video_path(row: pd.Series, dataset_root: Path) -> Path:
    if "clip_path" in row and pd.notna(row["clip_path"]) and str(row["clip_path"]).strip():
        path = Path(str(row["clip_path"]))
        if path.exists():
            return path

    if "output_relpath" in row and pd.notna(row["output_relpath"]) and str(row["output_relpath"]).strip():
        path = dataset_root / str(row["output_relpath"])
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot resolve video path for logical_sample_id={row.get('logical_sample_id', '')}"
    )


def decode_video_cv2(path: Path) -> np.ndarray:
    """Decode the complete short clip as RGB uint8 [N,H,W,3]."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")

    return np.stack(frames, axis=0)


def sample_segment_indices(num_frames: int, num_segments: int, training: bool) -> np.ndarray:
    """
    TSM-style segment sampling:
      - training: one random frame from each temporal segment
      - eval: center frame from each segment
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")

    if num_frames < num_segments:
        return np.linspace(0, num_frames - 1, num_segments).round().astype(np.int64)

    boundaries = np.linspace(0, num_frames, num_segments + 1)
    indices: list[int] = []

    for segment in range(num_segments):
        start = int(np.floor(boundaries[segment]))
        end = int(np.floor(boundaries[segment + 1])) - 1
        start = min(max(start, 0), num_frames - 1)
        end = min(max(end, start), num_frames - 1)

        if training:
            index = random.randint(start, end)
        else:
            index = (start + end) // 2
        indices.append(index)

    return np.asarray(indices, dtype=np.int64)


def apply_clip_consistent_augmentation(
    frames: torch.Tensor,
    *,
    horizontal_flip_p: float,
    brightness_jitter: float,
    contrast_jitter: float,
) -> torch.Tensor:
    """Apply the same random transform to every sampled frame."""
    if random.random() < horizontal_flip_p:
        frames = torch.flip(frames, dims=[3])

    if brightness_jitter > 0:
        factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        frames = frames * factor

    if contrast_jitter > 0:
        factor = 1.0 + random.uniform(-contrast_jitter, contrast_jitter)
        mean = frames.mean(dim=(1, 2, 3), keepdim=True)
        frames = (frames - mean) * factor + mean

    return frames.clamp_(0.0, 1.0)


class ExamVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        label_to_index: dict[str, int],
        num_segments: int = 8,
        input_size: int = 224,
        training: bool = False,
        horizontal_flip_p: float = 0.5,
        brightness_jitter: float = 0.10,
        contrast_jitter: float = 0.10,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.label_to_index = dict(label_to_index)
        self.num_segments = int(num_segments)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.horizontal_flip_p = float(horizontal_flip_p)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)

        self.df = pd.read_csv(self.manifest_path)
        required = {"logical_sample_id", "training_label"}
        missing = required - set(self.df.columns)
        if missing:
            raise RuntimeError(f"{self.manifest_path} missing columns: {sorted(missing)}")

        unknown_labels = sorted(set(self.df["training_label"].astype(str)) - set(self.label_to_index))
        if unknown_labels:
            raise RuntimeError(f"Unknown labels: {unknown_labels}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        video_path = resolve_video_path(row, self.dataset_root)

        all_frames = decode_video_cv2(video_path)
        sample_indices = sample_segment_indices(
            len(all_frames), self.num_segments, self.training
        )
        frames = all_frames[sample_indices]

        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        if frames_t.shape[-2:] != (self.input_size, self.input_size):
            frames_t = torch.nn.functional.interpolate(
                frames_t,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.training:
            frames_t = apply_clip_consistent_augmentation(
                frames_t,
                horizontal_flip_p=self.horizontal_flip_p,
                brightness_jitter=self.brightness_jitter,
                contrast_jitter=self.contrast_jitter,
            )

        frames_t = (frames_t - IMAGENET_MEAN) / IMAGENET_STD

        label_name = str(row["training_label"])
        return {
            "video": frames_t,  # [T,C,H,W]
            "label": torch.tensor(self.label_to_index[label_name], dtype=torch.long),
            "label_name": label_name,
            "logical_sample_id": str(row["logical_sample_id"]),
            "session_id": str(row.get("session_id", "")),
            "video_path": str(video_path),
        }
