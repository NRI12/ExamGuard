from __future__ import annotations

import uuid

from app.ai.action_recognition.roi import union_bbox
from app.ai.action_recognition.types import ActionProposal, ProposalType
from app.ai.domain import Track
from app.ai.seat_identity.types import AssignmentState, SeatDefinition


def _center(seat: SeatDefinition) -> tuple[float, float]:
    x1, y1, x2, y2 = seat.bbox_norm
    return (x1 + x2) / 2, (y1 + y2) / 2


def adjacent_seat_pairs(
    seats: tuple[SeatDefinition, ...],
    *,
    row_tolerance_ratio: float,
    max_gap_ratio: float,
) -> tuple[tuple[uuid.UUID, uuid.UUID], ...]:
    """Connect only consecutive left/right seats in deterministic geometry-derived rows."""
    remaining = sorted(seats, key=lambda seat: (_center(seat)[1], _center(seat)[0], str(seat.id)))
    rows: list[list[SeatDefinition]] = []
    for seat in remaining:
        center_y = _center(seat)[1]
        height = seat.bbox_norm[3] - seat.bbox_norm[1]
        placed = False
        for row in rows:
            anchor_y = sum(_center(item)[1] for item in row) / len(row)
            anchor_height = max(item.bbox_norm[3] - item.bbox_norm[1] for item in row)
            if abs(center_y - anchor_y) <= row_tolerance_ratio * max(height, anchor_height):
                row.append(seat)
                placed = True
                break
        if not placed:
            rows.append([seat])

    pairs: list[tuple[uuid.UUID, uuid.UUID]] = []
    for row in rows:
        ordered = sorted(row, key=lambda seat: (_center(seat)[0], str(seat.id)))
        for left, right in zip(ordered, ordered[1:], strict=False):
            horizontal_gap = right.bbox_norm[0] - left.bbox_norm[2]
            reference_width = max(
                left.bbox_norm[2] - left.bbox_norm[0],
                right.bbox_norm[2] - right.bbox_norm[0],
            )
            if horizontal_gap <= max_gap_ratio * reference_width:
                pairs.append((left.id, right.id))
    return tuple(pairs)


def build_action_proposals(
    tracks: tuple[Track, ...],
    adjacent_pairs: tuple[tuple[uuid.UUID, uuid.UUID], ...],
    timestamp_ms: int,
) -> tuple[ActionProposal, ...]:
    assigned = {
        track.identity.seat_id: track
        for track in tracks
        if track.identity.state is AssignmentState.ASSIGNED
        and track.identity.seat_id is not None
        and track.identity.session_candidate_id is not None
    }
    proposals: list[ActionProposal] = []
    ordered_tracks = sorted(
        assigned.values(), key=lambda item: str(item.identity.session_candidate_id)
    )
    for track in ordered_tracks:
        candidate_id = track.identity.session_candidate_id
        seat_id = track.identity.seat_id
        assert candidate_id is not None and seat_id is not None
        proposals.append(
            ActionProposal(
                proposal_id=f"single:{candidate_id}",
                proposal_type=ProposalType.SINGLE,
                session_candidate_ids=(candidate_id,),
                seat_ids=(seat_id,),
                seat_codes=(track.identity.seat_code or "",),
                current_track_ids=(track.track_id,),
                bbox_norm=track.bbox_norm,
                timestamp_ms=timestamp_ms,
            )
        )
    for left_seat, right_seat in adjacent_pairs:
        left, right = assigned.get(left_seat), assigned.get(right_seat)
        if left is None or right is None:
            continue
        actors = sorted(
            (
                (left.identity.session_candidate_id, left_seat, left),
                (right.identity.session_candidate_id, right_seat, right),
            ),
            key=lambda item: str(item[0]),
        )
        assert actors[0][0] is not None and actors[1][0] is not None
        candidate_ids = (actors[0][0], actors[1][0])
        proposals.append(
            ActionProposal(
                proposal_id=f"pair:{candidate_ids[0]}:{candidate_ids[1]}",
                proposal_type=ProposalType.PAIR,
                session_candidate_ids=(candidate_ids[0], candidate_ids[1]),
                seat_ids=(actors[0][1], actors[1][1]),
                seat_codes=(
                    actors[0][2].identity.seat_code or "",
                    actors[1][2].identity.seat_code or "",
                ),
                current_track_ids=(actors[0][2].track_id, actors[1][2].track_id),
                bbox_norm=union_bbox((left.bbox_norm, right.bbox_norm)),
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(proposals)


def build_logical_action_proposals(
    tracks: tuple[Track, ...],
    neighbor_pairs: tuple[tuple[str, str], ...],
    timestamp_ms: int,
) -> tuple[ActionProposal, ...]:
    """Build stable Actor-ID proposals without requiring Seat geometry."""
    active = {track.actor_id: track for track in tracks if track.actor_id}
    proposals: list[ActionProposal] = []
    for actor_id, track in sorted(active.items()):
        candidate_ids = (
            (track.identity.session_candidate_id,)
            if track.identity.session_candidate_id is not None
            else ()
        )
        seat_ids = (track.identity.seat_id,) if track.identity.seat_id is not None else ()
        seat_codes = (track.identity.seat_code,) if track.identity.seat_code else ()
        proposals.append(
            ActionProposal(
                proposal_id=f"single:{actor_id}",
                proposal_type=ProposalType.SINGLE,
                actor_ids=(actor_id,),
                session_candidate_ids=candidate_ids,
                seat_ids=seat_ids,
                seat_codes=seat_codes,
                current_track_ids=(track.track_id,),
                bbox_norm=track.bbox_norm,
                timestamp_ms=timestamp_ms,
            )
        )
    for left_actor, right_actor in sorted(neighbor_pairs):
        left, right = active.get(left_actor), active.get(right_actor)
        if left is None or right is None:
            continue
        actor_ids = tuple(sorted((left_actor, right_actor)))
        candidate_values = (left.identity.session_candidate_id, right.identity.session_candidate_id)
        pair_candidate_ids: tuple[uuid.UUID, ...] = ()
        if all(value is not None for value in candidate_values):
            pair_candidate_ids = tuple(
                sorted((value for value in candidate_values if value is not None), key=str)
            )
        seat_values = (left.identity.seat_id, right.identity.seat_id)
        pair_seat_ids: tuple[uuid.UUID, ...] = ()
        if all(value is not None for value in seat_values):
            pair_seat_ids = tuple(value for value in seat_values if value is not None)
        seat_code_values = (left.identity.seat_code, right.identity.seat_code)
        pair_seat_codes: tuple[str, ...] = ()
        if all(value is not None for value in seat_code_values):
            pair_seat_codes = tuple(value for value in seat_code_values if value is not None)
        proposals.append(
            ActionProposal(
                proposal_id=f"pair:{actor_ids[0]}:{actor_ids[1]}",
                proposal_type=ProposalType.PAIR,
                actor_ids=actor_ids,
                session_candidate_ids=pair_candidate_ids,
                seat_ids=pair_seat_ids,
                seat_codes=pair_seat_codes,
                current_track_ids=(left.track_id, right.track_id),
                bbox_norm=union_bbox((left.bbox_norm, right.bbox_norm)),
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(proposals)
