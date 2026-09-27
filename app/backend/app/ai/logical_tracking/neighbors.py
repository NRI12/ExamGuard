from __future__ import annotations

from dataclasses import dataclass

from app.ai.domain import Track
from app.monitoring.config import DynamicNeighborsConfig


@dataclass(frozen=True, slots=True)
class DynamicNeighbor:
    left_actor_id: str
    right_actor_id: str
    distance: float

    @property
    def pair_id(self) -> str:
        return f"pair:{self.left_actor_id}:{self.right_actor_id}"


def canonical_pair_id(left_actor_id: str, right_actor_id: str) -> str:
    left, right = sorted((left_actor_id, right_actor_id))
    return f"pair:{left}:{right}"


def dynamic_neighbor_pairs(
    tracks: tuple[Track, ...],
    config: DynamicNeighborsConfig,
) -> tuple[tuple[str, str], ...]:
    if not config.enabled or config.max_neighbors_per_actor == 0:
        return ()
    candidates: list[DynamicNeighbor] = []
    ordered = sorted((track for track in tracks if track.actor_id), key=lambda item: item.actor_id)
    for index, left in enumerate(ordered):
        lx1, ly1, lx2, ly2 = left.bbox_norm
        lw, lh = lx2 - lx1, ly2 - ly1
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        for right in ordered[index + 1 :]:
            rx1, ry1, rx2, ry2 = right.bbox_norm
            rw, rh = rx2 - rx1, ry2 - ry1
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            if min(lw, lh, rw, rh) <= 0:
                continue
            scale_similarity = min(lw * lh, rw * rh) / max(lw * lh, rw * rh)
            if scale_similarity < config.min_scale_similarity:
                continue
            vertical_ratio = abs(lcy - rcy) / max(lh, rh)
            horizontal_gap = max(0.0, max(lx1, rx1) - min(lx2, rx2))
            horizontal_ratio = horizontal_gap / max(lw, rw)
            if (
                vertical_ratio > config.max_vertical_gap_ratio
                or horizontal_ratio > config.max_horizontal_gap_ratio
            ):
                continue
            actor_left, actor_right = sorted((left.actor_id, right.actor_id))
            distance = ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5
            candidates.append(DynamicNeighbor(actor_left, actor_right, distance))

    degrees: dict[str, int] = {}
    selected: list[tuple[str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.distance, item.left_actor_id, item.right_actor_id),
    ):
        if (
            degrees.get(candidate.left_actor_id, 0) >= config.max_neighbors_per_actor
            or degrees.get(candidate.right_actor_id, 0) >= config.max_neighbors_per_actor
        ):
            continue
        selected.append((candidate.left_actor_id, candidate.right_actor_id))
        degrees[candidate.left_actor_id] = degrees.get(candidate.left_actor_id, 0) + 1
        degrees[candidate.right_actor_id] = degrees.get(candidate.right_actor_id, 0) + 1
    return tuple(sorted(selected))
