from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from app.ai.logical_tracking.types import NormalizedRect


class AppearanceEncoder(Protocol):
    """Batch appearance encoder used only by sparse recovery/initial seeding."""

    def encode(
        self,
        frame_bgr: np.ndarray,
        bboxes: Sequence[NormalizedRect],
    ) -> tuple[np.ndarray | None, ...]: ...


def l2_normalize(value: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


class HsvHistogramAppearanceEncoder:
    """Small CPU appearance baseline; one normalized embedding per person crop.

    This is deliberately not presented as a learned person-ReID model. It gives
    EXP-TRACK-B an inexpensive, locally reproducible appearance signal without
    adding a checkpoint or consuming detector/TSM GPU capacity.
    """

    def __init__(self, *, hue_bins: int = 16, saturation_bins: int = 8) -> None:
        self.hue_bins = hue_bins
        self.saturation_bins = saturation_bins

    def encode(
        self,
        frame_bgr: np.ndarray,
        bboxes: Sequence[NormalizedRect],
    ) -> tuple[np.ndarray | None, ...]:
        height, width = frame_bgr.shape[:2]
        embeddings: list[np.ndarray | None] = []
        for x1, y1, x2, y2 in bboxes:
            left = max(0, min(width, int(x1 * width)))
            top = max(0, min(height, int(y1 * height)))
            right = max(0, min(width, int(np.ceil(x2 * width))))
            bottom = max(0, min(height, int(np.ceil(y2 * height))))
            crop = frame_bgr[top:bottom, left:right]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                embeddings.append(None)
                continue
            crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            cells: list[np.ndarray] = []
            for row in range(2):
                for column in range(2):
                    cell = hsv[row * 64 : (row + 1) * 64, column * 32 : (column + 1) * 32]
                    histogram = cv2.calcHist(
                        [cell],
                        [0, 1],
                        None,
                        [self.hue_bins, self.saturation_bins],
                        [0, 180, 0, 256],
                    )
                    cells.append(histogram.reshape(-1))
            embeddings.append(l2_normalize(np.concatenate(cells)))
        return tuple(embeddings)
