from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.ai.action_recognition.proposals import build_logical_action_proposals
from app.ai.domain import Track
from app.ai.logical_tracking.manager import LogicalTrackManager
from app.ai.logical_tracking.neighbors import canonical_pair_id, dynamic_neighbor_pairs
from app.ai.logical_tracking.types import LogicalTrackObservation
from app.ai.seat_identity.types import AssignmentState, TrackIdentity
from app.monitoring.config import (
    DynamicNeighborsConfig,
    LogicalRecoveryConfig,
    LogicalTrackingConfig,
    ReIdConfig,
)


def observation(
    track_id: int,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> LogicalTrackObservation:
    return LogicalTrackObservation(track_id, bbox, confidence)


def manager(*, reid: bool = False, encoder: object | None = None) -> LogicalTrackManager:
    return LogicalTrackManager(
        LogicalTrackingConfig(
            recovery=LogicalRecoveryConfig(
                max_lost_ms=1000,
                max_position_distance=0.25,
                min_scale_similarity=0.60,
                min_score=0.60,
                ambiguity_margin=0.08,
                position_weight=0.55,
                motion_weight=0.25,
                scale_weight=0.20,
            )
        ),
        ReIdConfig(
            enabled=reid,
            min_similarity=0.70,
            min_combined_score=0.68,
            ambiguity_margin=0.05,
            prototype_seed_delay_ms=0,
            min_prototype_confidence=0.50,
        ),
        encoder,  # type: ignore[arg-type]
    )


def test_same_raw_track_keeps_actor_and_near_fragment_recovers() -> None:
    logical = manager()
    first = logical.update((observation(17, (0.10, 0.10, 0.30, 0.60)),), 0)
    actor_id = first.actors_by_track[17].actor_id
    same = logical.update((observation(17, (0.11, 0.10, 0.31, 0.60)),), 100)
    assert same.actors_by_track[17].actor_id == actor_id
    logical.update((), 200)
    recovered = logical.update((observation(42, (0.12, 0.10, 0.32, 0.60)),), 300)
    assert recovered.actors_by_track[42].actor_id == actor_id
    assert recovered.actors_by_track[42].recovered is True
    assert recovered.diagnostics.motion_recoveries == 1


def test_same_raw_id_reappearing_is_not_counted_as_fragmentation_recovery() -> None:
    logical = manager()
    first = logical.update((observation(17, (0.10, 0.10, 0.30, 0.60)),), 0)
    actor_id = first.actors_by_track[17].actor_id
    logical.update((), 100)

    reappeared = logical.update((observation(17, (0.11, 0.10, 0.31, 0.60)),), 200)

    assert reappeared.actors_by_track[17].actor_id == actor_id
    assert reappeared.actors_by_track[17].recovered is False
    assert reappeared.diagnostics.recoveries_total == 0


def test_far_gap_and_scale_mismatch_create_new_actors() -> None:
    for replacement, replacement_ms in (
        ((0.70, 0.10, 0.90, 0.60), 300),
        ((0.10, 0.10, 0.15, 0.22), 300),
        ((0.10, 0.10, 0.30, 0.60), 1500),
    ):
        logical = manager()
        original = logical.update((observation(1, (0.10, 0.10, 0.30, 0.60)),), 0)
        original_id = original.actors_by_track[1].actor_id
        logical.update((), 100)
        result = logical.update((observation(2, replacement),), replacement_ms)
        assert result.actors_by_track[2].actor_id != original_id


def test_ambiguity_is_deterministic_and_does_not_force_a_guess() -> None:
    logical = manager()
    first = logical.update(
        (
            observation(1, (0.10, 0.10, 0.30, 0.60)),
            observation(2, (0.20, 0.10, 0.40, 0.60)),
        ),
        0,
    )
    old_ids = {actor.actor_id for actor in first.active_actors}
    logical.update((), 100)
    result = logical.update((observation(3, (0.15, 0.10, 0.35, 0.60)),), 200)
    assert result.actors_by_track[3].actor_id not in old_ids
    assert result.diagnostics.ambiguous_recoveries == 1


def test_recovery_is_one_to_one() -> None:
    logical = manager()
    first = logical.update((observation(1, (0.10, 0.10, 0.30, 0.60)),), 0)
    original_id = first.actors_by_track[1].actor_id
    logical.update((), 100)
    result = logical.update(
        (
            observation(2, (0.11, 0.10, 0.31, 0.60)),
            observation(3, (0.12, 0.10, 0.32, 0.60)),
        ),
        200,
    )
    assert sum(actor.actor_id == original_id for actor in result.active_actors) == 1
    assert len({actor.actor_id for actor in result.active_actors}) == 2


class FakeAppearanceEncoder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode(
        self,
        frame_bgr: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, ...]:
        del frame_bgr
        self.calls.append(len(bboxes))
        return tuple(
            np.asarray([1.0, 0.0], dtype=np.float32)
            if bbox[0] < 0.18
            else np.asarray([0.0, 1.0], dtype=np.float32)
            for bbox in bboxes
        )


def test_sparse_reid_only_runs_for_seed_and_ambiguous_recovery() -> None:
    encoder = FakeAppearanceEncoder()
    logical = manager(reid=True, encoder=encoder)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = logical.update(
        (
            observation(1, (0.10, 0.10, 0.30, 0.60)),
            observation(2, (0.20, 0.10, 0.40, 0.60)),
        ),
        0,
        frame,
    )
    left_actor = first.actors_by_track[1].actor_id
    prototype = logical._actors[left_actor].appearance_prototype
    assert prototype is not None
    prototype_before = prototype.copy()
    assert encoder.calls == [2]
    logical.update(
        (
            observation(1, (0.10, 0.10, 0.30, 0.60)),
            observation(2, (0.20, 0.10, 0.40, 0.60)),
        ),
        100,
        frame,
    )
    assert encoder.calls == [2]
    logical.update((), 200, frame)
    result = logical.update((observation(3, (0.15, 0.10, 0.35, 0.60)),), 300, frame)
    assert result.actors_by_track[3].actor_id == left_actor
    assert result.diagnostics.reid_recoveries == 1
    assert encoder.calls == [2, 1]
    np.testing.assert_array_equal(
        logical._actors[left_actor].appearance_prototype, prototype_before
    )


def test_unambiguous_motion_and_impossible_geometry_skip_reid() -> None:
    encoder = FakeAppearanceEncoder()
    logical = manager(reid=True, encoder=encoder)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logical.update((observation(1, (0.10, 0.10, 0.30, 0.60)),), 0, frame)
    encoder.calls.clear()
    logical.update((), 100, frame)
    logical.update((observation(2, (0.11, 0.10, 0.31, 0.60)),), 200, frame)
    assert encoder.calls == []
    logical.update((), 300, frame)
    logical.update((observation(3, (0.75, 0.10, 0.95, 0.60)),), 400, frame)
    assert encoder.calls == [1]  # one-time prototype seed for the genuinely new actor


def make_track(actor_id: str, track_id: int, bbox: tuple[float, float, float, float]) -> Track:
    return Track(
        track_id=track_id,
        bbox_norm=bbox,
        confidence=0.9,
        identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
        actor_id=actor_id,
    )


def test_dynamic_neighbors_limits_and_stable_pair_identity() -> None:
    config = DynamicNeighborsConfig(
        max_neighbors_per_actor=1,
        max_horizontal_gap_ratio=1.5,
        max_vertical_gap_ratio=0.5,
        min_scale_similarity=0.5,
    )
    tracks = (
        make_track("A0001", 1, (0.10, 0.10, 0.20, 0.50)),
        make_track("A0002", 2, (0.22, 0.11, 0.32, 0.51)),
        make_track("A0003", 3, (0.70, 0.10, 0.80, 0.50)),
        make_track("A0004", 4, (0.10, 0.70, 0.20, 0.95)),
    )
    pairs = dynamic_neighbor_pairs(tracks, config)
    assert pairs == (("A0001", "A0002"),)
    assert canonical_pair_id("A0002", "A0001") == "pair:A0001:A0002"
    before = build_logical_action_proposals(tracks, pairs, 100)
    pair_before = next(item for item in before if item.proposal_type.value == "PAIR")
    fragmented = tuple(
        replace(track, track_id=42) if track.actor_id == "A0001" else track for track in tracks
    )
    after = build_logical_action_proposals(fragmented, pairs, 200)
    pair_after = next(item for item in after if item.proposal_type.value == "PAIR")
    assert pair_before.proposal_id == pair_after.proposal_id == "pair:A0001:A0002"
    assert (
        next(item for item in after if item.actor_ids == ("A0001",)).proposal_id == "single:A0001"
    )
