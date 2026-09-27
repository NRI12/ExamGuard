from __future__ import annotations

import time
import uuid

import pytest

from app.ai.domain import Track
from app.ai.seat_identity.assignment import SeatAssignmentEngine
from app.ai.seat_identity.geometry import (
    clip_rect,
    combined_score,
    distance_score,
    expand_seat,
    intersection_area,
    overlap_score,
    rect_center,
)
from app.ai.seat_identity.matcher import build_candidates, greedy_one_to_one
from app.ai.seat_identity.types import (
    AssignmentState,
    SeatCandidateBinding,
    SeatDefinition,
    SeatIdentityContext,
    SeatOccupancyState,
    TrackIdentity,
)
from app.monitoring.config import SeatAssignmentConfig


def seat(code: str, bbox: tuple[float, float, float, float]) -> SeatDefinition:
    return SeatDefinition(id=uuid.uuid5(uuid.NAMESPACE_DNS, code), code=code, bbox_norm=bbox)


SEAT_3 = seat("B03", (0.10, 0.10, 0.30, 0.70))
SEAT_4 = seat("B04", (0.40, 0.10, 0.60, 0.70))


def context(*seats: SeatDefinition) -> SeatIdentityContext:
    bindings = tuple(
        SeatCandidateBinding(
            seat_id=item.id,
            session_candidate_id=uuid.uuid5(uuid.NAMESPACE_URL, f"assignment-{item.code}"),
            candidate_id=uuid.uuid5(uuid.NAMESPACE_URL, f"candidate-{item.code}"),
            candidate_code=f"SV{item.code[-2:]}",
            candidate_name=f"Candidate {item.code}",
        )
        for item in seats
    )
    return SeatIdentityContext(session_id=uuid.uuid4(), seats=tuple(seats), bindings=bindings)


def config(**overrides: object) -> SeatAssignmentConfig:
    return SeatAssignmentConfig.model_validate(
        {
            "confirm_ms": 600,
            "release_ms": 1500,
            "switch_confirm_ms": 800,
            **overrides,
        }
    )


def track(track_id: int, bbox: tuple[float, float, float, float]) -> Track:
    return Track(
        track_id=track_id,
        bbox_norm=bbox,
        confidence=0.9,
        identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
    )


def test_geometry_clips_expands_and_scores_safely() -> None:
    assert intersection_area((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)) == pytest.approx(0.0625)
    assert clip_rect((-0.2, 0.1, 1.2, 0.9)) == (0.0, 0.1, 1.0, 0.9)
    assert expand_seat((0.1, 0.2, 0.3, 0.6), 0.1) == pytest.approx((0.08, 0.16, 0.32, 0.64))
    assert expand_seat((0.0, 0.0, 0.2, 0.2), 0.5) == pytest.approx(
        (0.0, 0.0, 0.3, 0.3)
    )
    assert rect_center((0.1, 0.2, 0.3, 0.6)) == pytest.approx((0.2, 0.4))
    assert overlap_score((0.1, 0.1, 0.3, 0.3), (0.2, 0.1, 0.4, 0.3)) == pytest.approx(0.5)
    assert overlap_score((0.2, 0.2, 0.2, 0.5), SEAT_3.bbox_norm) == 0
    assert distance_score(SEAT_3.bbox_norm, SEAT_3.bbox_norm) == 1
    assert distance_score((2, 2, 3, 3), SEAT_3.bbox_norm) == 0
    assert combined_score(0.5, 0.25, 0.7, 0.3) == pytest.approx(0.425)


def test_matcher_clear_far_adjacent_and_conflict() -> None:
    cfg = config(confirm_ms=0)
    clear = build_candidates({1: SEAT_3.bbox_norm}, (SEAT_3, SEAT_4), cfg)
    assert greedy_one_to_one(clear)[1].seat_id == SEAT_3.id

    far = build_candidates({1: (0.75, 0.10, 0.90, 0.70)}, (SEAT_3, SEAT_4), cfg)
    assert greedy_one_to_one(far) == {}

    adjacent = build_candidates(
        {1: SEAT_3.bbox_norm, 2: SEAT_4.bbox_norm},
        (SEAT_3, SEAT_4),
        cfg,
    )
    selected = greedy_one_to_one(adjacent)
    assert {track_id: item.seat_id for track_id, item in selected.items()} == {
        1: SEAT_3.id,
        2: SEAT_4.id,
    }

    conflict = build_candidates(
        {1: SEAT_3.bbox_norm, 2: SEAT_3.bbox_norm},
        (SEAT_3,),
        cfg,
    )
    selected = greedy_one_to_one(conflict)
    assert len(selected) == 1
    assert next(iter(selected.values())).seat_id == SEAT_3.id


def test_temporal_confirmation_pause_and_supervisor_unassigned() -> None:
    engine = SeatAssignmentEngine(context(SEAT_3), config())
    initial = engine.update((track(1, SEAT_3.bbox_norm),), 100)
    assert initial.identities[1].state == AssignmentState.TENTATIVE
    time.sleep(0.01)  # wall-clock passage must not advance video-time confirmation
    still_tentative = engine.update((track(1, SEAT_3.bbox_norm),), 200)
    assert still_tentative.identities[1].state == AssignmentState.TENTATIVE
    assigned = engine.update((track(1, SEAT_3.bbox_norm),), 700)
    assert assigned.identities[1].state == AssignmentState.ASSIGNED

    supervisor = SeatAssignmentEngine(context(SEAT_3, SEAT_4), config())
    for timestamp in (100, 800, 1600):
        snapshot = supervisor.update((track(9, (0.72, 0.10, 0.82, 0.60)),), timestamp)
        assert snapshot.identities[9].state == AssignmentState.UNASSIGNED
        assert snapshot.identities[9].session_candidate_id is None


def test_adjacent_jitter_does_not_switch_but_genuine_move_does() -> None:
    engine = SeatAssignmentEngine(context(SEAT_3, SEAT_4), config())
    engine.update((track(1, SEAT_3.bbox_norm),), 100)
    assigned = engine.update((track(1, SEAT_3.bbox_norm),), 700)
    assert assigned.identities[1].seat_id == SEAT_3.id

    spike = engine.update((track(1, SEAT_4.bbox_norm),), 800)
    assert spike.identities[1].seat_id == SEAT_3.id
    recovered = engine.update((track(1, SEAT_3.bbox_norm),), 900)
    assert recovered.identities[1].seat_id == SEAT_3.id
    assert recovered.seat_switches == 0

    engine.update((track(1, SEAT_4.bbox_norm),), 1000)
    switched = engine.update((track(1, SEAT_4.bbox_norm),), 1800)
    assert switched.identities[1].state == AssignmentState.ASSIGNED
    assert switched.identities[1].seat_id == SEAT_4.id
    assert switched.seat_switches == 1


def test_fragmentation_recovers_same_session_candidate_via_grace_seat() -> None:
    identity_context = context(SEAT_3)
    engine = SeatAssignmentEngine(identity_context, config())
    engine.update((track(17, SEAT_3.bbox_norm),), 100)
    before = engine.update((track(17, SEAT_3.bbox_norm),), 700)
    expected_candidate = before.identities[17].session_candidate_id
    assert expected_candidate is not None

    missing = engine.update((), 800)
    assert missing.seats[0].state == SeatOccupancyState.GRACE
    tentative = engine.update((track(24, SEAT_3.bbox_norm),), 900)
    assert tentative.identities[24].state == AssignmentState.TENTATIVE
    recovered = engine.update((track(24, SEAT_3.bbox_norm),), 1500)
    assert recovered.identities[24].state == AssignmentState.ASSIGNED
    assert recovered.identities[24].session_candidate_id == expected_candidate
    assert recovered.identity_recoveries == 1
    assert recovered.seats[0].state == SeatOccupancyState.OCCUPIED


def test_candidate_leaving_is_released_and_reset_removes_all_runtime_state() -> None:
    engine = SeatAssignmentEngine(context(SEAT_3), config())
    engine.update((track(1, SEAT_3.bbox_norm),), 100)
    engine.update((track(1, SEAT_3.bbox_norm),), 700)
    grace = engine.update((track(1, (0.75, 0.1, 0.9, 0.7)),), 800)
    assert grace.identities[1].state == AssignmentState.ASSIGNED
    assert grace.seats[0].state == SeatOccupancyState.GRACE
    released = engine.update((track(1, (0.75, 0.1, 0.9, 0.7)),), 2300)
    assert released.identities[1].state == AssignmentState.UNASSIGNED
    assert released.seats[0].state == SeatOccupancyState.EMPTY

    engine.reset()
    restarted = engine.update((track(1, SEAT_3.bbox_norm),), 50)
    assert restarted.identities[1].state == AssignmentState.TENTATIVE
    assert restarted.identity_recoveries == 0
    assert restarted.seat_switches == 0


def test_config_rejects_invalid_seat_assignment_values() -> None:
    with pytest.raises(ValueError):
        SeatAssignmentConfig(overlap_weight=0, distance_weight=0)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(min_score=1.1)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(seat_expand_ratio=-0.01)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(confirm_ms=-1)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(release_ms=-1)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(switch_margin=-0.01)
    with pytest.raises(ValueError):
        SeatAssignmentConfig(switch_confirm_ms=-1)
