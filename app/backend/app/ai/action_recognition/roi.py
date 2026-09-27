from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RoiPreparationProfile:
    crop_ms: float
    resize_ms: float
    color_conversion_ms: float

    @property
    def total_ms(self) -> float:
        return self.crop_ms + self.resize_ms + self.color_conversion_ms


def union_bbox(
    boxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[float, float, float, float]:
    if not boxes:
        raise ValueError("at least one actor bbox is required")
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if len(valid) != len(boxes):
        raise ValueError("actor bbox must have positive area")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def training_crop_box(
    bbox_norm: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
    min_width_px: int,
    min_height_px: int,
) -> tuple[int, int, int, int]:
    """Reproduce 04d_generate_r3_r4_rois.py crop geometry exactly."""
    if width < 2 or height < 2:
        raise ValueError("source dimensions must be at least 2x2")
    x0, y0, x1, y1 = (
        bbox_norm[0] * width,
        bbox_norm[1] * height,
        bbox_norm[2] * width,
        bbox_norm[3] * height,
    )
    x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
    y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("actor bbox has no visible area")
    box_width, box_height = x1 - x0, y1 - y0
    ax0, ax1 = x0 - box_width * expand_x, x1 + box_width * expand_x
    ay0, ay1 = y0 - box_height * expand_top, y1 + box_height * expand_bottom
    crop_width = min(max(ax1 - ax0, float(min_width_px)), float(width))
    crop_height = min(max(ay1 - ay0, float(min_height_px)), float(height))
    center_x, center_y = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    left = max(0.0, min(center_x - crop_width / 2, width - crop_width))
    top = max(0.0, min(center_y - crop_height / 2, height - crop_height))
    even_width = min(max(2, int(round(crop_width)) // 2 * 2), width - width % 2)
    even_height = min(max(2, int(round(crop_height)) // 2 * 2), height - height % 2)
    x = max(0, min(int(round(left)) // 2 * 2, width - even_width))
    y = max(0, min(int(round(top)) // 2 * 2, height - even_height))
    return x, y, even_width, even_height


def extract_training_roi(
    frame_bgr: np.ndarray,
    bbox_norm: tuple[float, float, float, float],
    *,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
    min_width_px: int,
    min_height_px: int,
    output_size: int = 224,
) -> np.ndarray:
    roi, _ = extract_training_roi_profiled(
        frame_bgr,
        bbox_norm,
        expand_x=expand_x,
        expand_top=expand_top,
        expand_bottom=expand_bottom,
        min_width_px=min_width_px,
        min_height_px=min_height_px,
        output_size=output_size,
    )
    return roi


def extract_training_roi_profiled(
    frame_bgr: np.ndarray,
    bbox_norm: tuple[float, float, float, float],
    *,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
    min_width_px: int,
    min_height_px: int,
    output_size: int = 224,
) -> tuple[np.ndarray, RoiPreparationProfile]:
    height, width = frame_bgr.shape[:2]
    started = time.perf_counter()
    x, y, crop_width, crop_height = training_crop_box(
        bbox_norm,
        width,
        height,
        expand_x=expand_x,
        expand_top=expand_top,
        expand_bottom=expand_bottom,
        min_width_px=min_width_px,
        min_height_px=min_height_px,
    )
    crop = frame_bgr[y : y + crop_height, x : x + crop_width]
    crop_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    resize_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    color_conversion_ms = (time.perf_counter() - started) * 1000
    return rgb, RoiPreparationProfile(crop_ms, resize_ms, color_conversion_ms)
