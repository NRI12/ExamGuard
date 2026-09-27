from __future__ import annotations

import uuid
from collections.abc import Mapping

from app.ai.seat_identity.geometry import (
    combined_score,
    distance_score,
    expand_seat,
    overlap_score,
)
from app.ai.seat_identity.types import (
    NormalizedRect,
    SeatDefinition,
    SeatMatchCandidate,
)
from app.monitoring.config import SeatAssignmentConfig


def score_track_seat(
    track_id: int,
    track_bbox: NormalizedRect,
    seat: SeatDefinition,
    config: SeatAssignmentConfig,
) -> SeatMatchCandidate:
    expanded = expand_seat(seat.bbox_norm, config.seat_expand_ratio)
    overlap = overlap_score(track_bbox, expanded)
    distance = distance_score(track_bbox, expanded)
    score = combined_score(
        overlap,
        distance,
        config.overlap_weight,
        config.distance_weight,
    )
    return SeatMatchCandidate(track_id, seat.id, overlap, distance, score)


def build_candidates(
    track_boxes: Mapping[int, NormalizedRect],
    seats: tuple[SeatDefinition, ...],
    config: SeatAssignmentConfig,
) -> list[SeatMatchCandidate]:
    candidates = [
        score_track_seat(track_id, bbox, seat, config)
        for track_id, bbox in track_boxes.items()
        for seat in seats
    ]
    return [candidate for candidate in candidates if candidate.score >= config.min_score]


def greedy_one_to_one(
    candidates: list[SeatMatchCandidate],
    *,
    used_tracks: set[int] | None = None,
    used_seats: set[uuid.UUID] | None = None,
) -> dict[int, SeatMatchCandidate]:
    claimed_tracks = set(used_tracks or ())
    claimed_seats = set(used_seats or ())
    selected: dict[int, SeatMatchCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.score, -item.overlap, item.track_id, str(item.seat_id)),
    ):
        if candidate.track_id in claimed_tracks or candidate.seat_id in claimed_seats:
            continue
        selected[candidate.track_id] = candidate
        claimed_tracks.add(candidate.track_id)
        claimed_seats.add(candidate.seat_id)
    return selected
