from __future__ import annotations

import math

from app.ai.seat_identity.types import NormalizedRect


def clip_rect(rect: NormalizedRect) -> NormalizedRect:
    values = tuple(float(value) if math.isfinite(value) else 0.0 for value in rect)
    x1, y1, x2, y2 = values
    return (
        min(1.0, max(0.0, x1)),
        min(1.0, max(0.0, y1)),
        min(1.0, max(0.0, x2)),
        min(1.0, max(0.0, y2)),
    )


def rect_area(rect: NormalizedRect) -> float:
    x1, y1, x2, y2 = clip_rect(rect)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(left: NormalizedRect, right: NormalizedRect) -> float:
    lx1, ly1, lx2, ly2 = clip_rect(left)
    rx1, ry1, rx2, ry2 = clip_rect(right)
    return max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))


def rect_center(rect: NormalizedRect) -> tuple[float, float]:
    x1, y1, x2, y2 = clip_rect(rect)
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def expand_seat(rect: NormalizedRect, ratio: float) -> NormalizedRect:
    if ratio < 0 or not math.isfinite(ratio):
        raise ValueError("seat expansion ratio must be finite and non-negative")
    x1, y1, x2, y2 = clip_rect(rect)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x1, y1)
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(1.0, x2 + dx),
        min(1.0, y2 + dy),
    )


def overlap_score(track: NormalizedRect, expanded_seat: NormalizedRect) -> float:
    track_area = rect_area(track)
    if track_area <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection_area(track, expanded_seat) / track_area))


def distance_score(track: NormalizedRect, seat: NormalizedRect) -> float:
    sx1, sy1, sx2, sy2 = clip_rect(seat)
    diagonal = math.hypot(sx2 - sx1, sy2 - sy1)
    if diagonal <= 0 or rect_area(track) <= 0:
        return 0.0
    track_center = rect_center(track)
    seat_center = rect_center(seat)
    normalized_distance = (
        math.hypot(
            track_center[0] - seat_center[0],
            track_center[1] - seat_center[1],
        )
        / diagonal
    )
    return min(1.0, max(0.0, 1.0 - normalized_distance))


def combined_score(
    overlap: float,
    distance: float,
    overlap_weight: float,
    distance_weight: float,
) -> float:
    return overlap_weight * overlap + distance_weight * distance
