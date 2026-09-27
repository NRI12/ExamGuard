from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_list(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    def clip(self, width: int, height: int) -> "Box":
        x1 = min(max(self.x1, 0.0), float(width - 1))
        y1 = min(max(self.y1, 0.0), float(height - 1))
        x2 = min(max(self.x2, x1 + 1.0), float(width))
        y2 = min(max(self.y2, y1 + 1.0), float(height))
        return Box(x1, y1, x2, y2)

    def to_int(self, width: int, height: int) -> tuple[int, int, int, int]:
        b = self.clip(width, height)
        x1 = int(round(b.x1))
        y1 = int(round(b.y1))
        x2 = int(round(b.x2))
        y2 = int(round(b.y2))
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        return x1, y1, x2, y2


def normalized_to_pixels(box: Box, width: int, height: int) -> Box:
    return Box(
        box.x1 * width,
        box.y1 * height,
        box.x2 * width,
        box.y2 * height,
    ).clip(width, height)


def pixels_to_normalized(box: Box, width: int, height: int) -> Box:
    return Box(
        box.x1 / width,
        box.y1 / height,
        box.x2 / width,
        box.y2 / height,
    )


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return 0.0 if union <= 0 else inter / union


def center_inside(inner: Box, outer: Box) -> bool:
    return outer.x1 <= inner.cx <= outer.x2 and outer.y1 <= inner.cy <= outer.y2


def association_score(
    detection: Box,
    anchor: Box,
    iou_weight: float,
    center_weight: float,
    inside_bonus_weight: float,
) -> float:
    anchor_diag = max(hypot(anchor.width, anchor.height), 1.0)
    center_dist = hypot(detection.cx - anchor.cx, detection.cy - anchor.cy)
    center_score = exp(-center_dist / anchor_diag)
    inside = 1.0 if center_inside(detection, anchor) else 0.0
    return (
        iou_weight * iou(detection, anchor)
        + center_weight * center_score
        + inside_bonus_weight * inside
    )


def robust_stable_box(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ValueError("robust_stable_box requires at least one box")
    arr = np.array([b.as_list() for b in boxes], dtype=np.float32)
    # 10/90% keeps moderate body movement while reducing one-frame detector outliers.
    x1 = float(np.quantile(arr[:, 0], 0.10))
    y1 = float(np.quantile(arr[:, 1], 0.10))
    x2 = float(np.quantile(arr[:, 2], 0.90))
    y2 = float(np.quantile(arr[:, 3], 0.90))
    return Box(x1, y1, x2, y2)


def expand_box(
    box: Box,
    width: int,
    height: int,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> Box:
    w, h = box.width, box.height
    return Box(
        box.x1 - left * w,
        box.y1 - top * h,
        box.x2 + right * w,
        box.y2 + bottom * h,
    ).clip(width, height)


def union_boxes(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("union_boxes requires at least one box")
    return Box(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )


def expand_uniform(box: Box, width: int, height: int, margin_ratio: float) -> Box:
    m_x = box.width * margin_ratio
    m_y = box.height * margin_ratio
    return Box(
        box.x1 - m_x,
        box.y1 - m_y,
        box.x2 + m_x,
        box.y2 + m_y,
    ).clip(width, height)
