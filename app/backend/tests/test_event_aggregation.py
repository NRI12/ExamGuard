from __future__ import annotations

import uuid
from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.action_recognition.types import ActionPrediction, ProposalType
from app.ai.event_aggregation.calibration import (
    CalibrationPrediction,
    GroundTruthEvent,
    align_predictions,
    assert_calibration_only,
    bounded_parameter_search,
    event_metrics,
    probability_distributions,
)
from app.ai.event_aggregation.runtime import EventAggregationRuntime
from app.ai.event_aggregation.types import AggregatedEvent, EventBehavior
from app.db.models.audit import AuditLog
from app.db.models.candidate import Candidate
from app.db.models.event import Event, EventActor
from app.db.models.room import Room, Seat
from app.db.models.session import ExamSession, SessionCandidate
from app.db.models.user import User
from app.features.events.service import persist_ai_event
from app.monitoring.config import (
    EventBehaviorConfig,
    EventDetectionConfig,
    EventSmoothingConfig,
)


def behavior_config(proposal_type: str) -> EventBehaviorConfig:
    return EventBehaviorConfig(
        proposal_types=[proposal_type],  # type: ignore[list-item]
        smoothing=EventSmoothingConfig(alpha=1.0),
        start_threshold=0.7,
        keep_threshold=0.5,
        min_active_ms=1000,
        end_grace_ms=1000,
        merge_gap_ms=500,
    )


def event_config() -> EventDetectionConfig:
    return EventDetectionConfig(
        enabled=True,
        max_discontinuity_ms=5000,
        dedup_temporal_iou=0.3,
        dedup_max_gap_ms=500,
        suspicious_looking=behavior_config("SINGLE"),
        communicating=behavior_config("PAIR"),
        exchange_object=behavior_config("PAIR"),
        using_phone_cheat_sheet=behavior_config("SINGLE"),
    )


def prediction(
    behavior: str,
    timestamp_ms: int,
    actors: tuple[uuid.UUID, ...],
    *,
    proposal_id: str | None = None,
    score: float = 0.9,
) -> ActionPrediction:
    proposal_type = ProposalType.SINGLE if len(actors) == 1 else ProposalType.PAIR
    names = (
        "normal",
        "suspicious_looking",
        "communicating",
        "exchange_object",
        "using_phone/cheat_sheet",
    )
    probabilities = tuple(score if name == behavior else 0.025 for name in names)
    canonical = tuple(sorted(actors, key=str))
    return ActionPrediction(
        proposal_id=proposal_id
        or (
            f"single:{canonical[0]}"
            if len(canonical) == 1
            else f"pair:{canonical[0]}:{canonical[1]}"
        ),
        proposal_type=proposal_type,
        session_candidate_ids=canonical,
        seat_codes=tuple(f"A{index + 1:02d}" for index in range(len(canonical))),
        timestamp_ms=timestamp_ms,
        probabilities=probabilities,
        predicted_class=behavior,
        confidence=score,
    )


def run_sequence(
    rows: list[ActionPrediction],
    *,
    stop: bool = True,
) -> tuple[EventAggregationRuntime, list[AggregatedEvent]]:
    events: list[AggregatedEvent] = []
    runtime = EventAggregationRuntime(
        session_id=uuid.uuid4(), config=event_config(), emit=events.append
    )
    for row in rows:
        runtime.consume((row,), 0)
    if stop:
        runtime.stop()
    return runtime, events


def closing_sequence(
    behavior: str,
    actors: tuple[uuid.UUID, ...],
    *,
    proposal_id: str | None = None,
) -> list[ActionPrediction]:
    return [
        prediction(behavior, 0, actors, proposal_id=proposal_id),
        prediction(behavior, 1000, actors, proposal_id=proposal_id),
        prediction(behavior, 2000, actors, proposal_id=proposal_id, score=0.1),
        prediction(behavior, 3000, actors, proposal_id=proposal_id, score=0.1),
    ]


def test_normal_never_creates_event_and_single_spike_is_suppressed() -> None:
    actor = (uuid.uuid4(),)
    normal_rows = [
        prediction("normal", timestamp, actor, score=0.9) for timestamp in (0, 1000, 2000)
    ]
    _, normal_events = run_sequence(normal_rows)
    assert normal_events == []

    _, spike_events = run_sequence(
        [
            prediction("suspicious_looking", 0, actor),
            prediction("suspicious_looking", 1000, actor, score=0.1),
        ]
    )
    assert spike_events == []


@pytest.mark.parametrize(
    "behavior",
    ["suspicious_looking", "using_phone/cheat_sheet"],
)
def test_sustained_single_behavior_creates_one_event(behavior: str) -> None:
    actor = (uuid.uuid4(),)
    _, events = run_sequence(closing_sequence(behavior, actor))
    assert len(events) == 1
    assert events[0].behavior.value == behavior.replace("/", "_")
    assert events[0].session_candidate_ids == actor
    assert events[0].start_ms == 0
    assert events[0].end_ms == 2000


@pytest.mark.parametrize("behavior", ["communicating", "exchange_object"])
def test_sustained_pair_behavior_creates_one_event_with_two_actors(behavior: str) -> None:
    actors = tuple(sorted((uuid.uuid4(), uuid.uuid4()), key=str))
    _, events = run_sequence(closing_sequence(behavior, actors))
    assert len(events) == 1
    assert events[0].session_candidate_ids == actors


def test_brief_drop_and_cooldown_recovery_keep_same_event() -> None:
    actor = (uuid.uuid4(),)
    rows = [
        prediction("suspicious_looking", 0, actor),
        prediction("suspicious_looking", 1000, actor),
        prediction("suspicious_looking", 1500, actor, score=0.1),
        prediction("suspicious_looking", 2000, actor),
        prediction("suspicious_looking", 3000, actor, score=0.1),
        prediction("suspicious_looking", 4000, actor, score=0.1),
    ]
    _, events = run_sequence(rows)
    assert len(events) == 1
    assert events[0].start_ms == 0
    assert events[0].end_ms == 3000


def test_long_drop_closes_and_later_evidence_starts_second_event() -> None:
    actor = (uuid.uuid4(),)
    first = closing_sequence("suspicious_looking", actor)
    second = [
        prediction("suspicious_looking", 4000, actor),
        prediction("suspicious_looking", 5000, actor),
        prediction("suspicious_looking", 6000, actor, score=0.1),
        prediction("suspicious_looking", 7000, actor, score=0.1),
    ]
    _, events = run_sequence(first + second)
    assert len(events) == 2


def test_invalid_proposal_behavior_routing_creates_no_event() -> None:
    actor = (uuid.uuid4(),)
    _, events = run_sequence(closing_sequence("communicating", actor))
    assert events == []


def test_seek_reset_discards_partial_event_and_timestamp_pause_does_not_advance() -> None:
    actor = (uuid.uuid4(),)
    events: list[AggregatedEvent] = []
    runtime = EventAggregationRuntime(
        session_id=uuid.uuid4(), config=event_config(), emit=events.append
    )
    runtime.consume((prediction("suspicious_looking", 0, actor),), 0)
    runtime.consume((prediction("suspicious_looking", 1000, actor),), 0)
    runtime.reset(1)
    runtime.stop()
    assert events == []

    runtime = EventAggregationRuntime(
        session_id=uuid.uuid4(), config=event_config(), emit=events.append
    )
    runtime.consume((prediction("suspicious_looking", 100, actor),), 0)
    assert runtime.diagnostics().candidate_fsms == 1
    runtime.consume((prediction("suspicious_looking", 1100, actor),), 0)
    assert runtime.diagnostics().active_fsms == 1
    runtime.stop()
    assert events[-1].end_ms == 1100


def test_stop_flushes_active_and_discards_insufficient_candidate() -> None:
    active_actor = (uuid.uuid4(),)
    runtime, events = run_sequence(
        [
            prediction("suspicious_looking", 0, active_actor),
            prediction("suspicious_looking", 1000, active_actor),
        ]
    )
    assert len(events) == 1 and events[0].end_ms == 1000
    assert runtime.diagnostics().events_created_total == 1

    candidate_actor = (uuid.uuid4(),)
    runtime, events = run_sequence([prediction("suspicious_looking", 0, candidate_actor)])
    assert events == []
    assert runtime.diagnostics().events_suppressed_total == 1


def test_proposal_expiry_uses_source_timestamp_and_resets_smoother() -> None:
    actor = (uuid.uuid4(),)
    events: list[AggregatedEvent] = []
    runtime = EventAggregationRuntime(
        session_id=uuid.uuid4(), config=event_config(), emit=events.append
    )
    runtime.consume((prediction("suspicious_looking", 0, actor),), 0)
    runtime.consume((prediction("suspicious_looking", 1000, actor),), 0)
    runtime.advance(7000, 0)
    assert len(events) == 1
    assert events[0].end_ms == 1000
    assert runtime.diagnostics().active_fsms == 0


def test_true_duplicate_is_merged_but_different_actors_are_not() -> None:
    actor = (uuid.uuid4(),)
    rows: list[ActionPrediction] = []
    for timestamp, score in ((0, 0.9), (1000, 0.9), (2000, 0.1), (3000, 0.1)):
        rows.extend(
            [
                prediction(
                    "suspicious_looking",
                    timestamp,
                    actor,
                    proposal_id="single:primary",
                    score=score,
                ),
                prediction(
                    "suspicious_looking", timestamp, actor, proposal_id="single:retry", score=score
                ),
            ]
        )
    runtime, events = run_sequence(rows)
    assert len(events) == 1
    assert runtime.diagnostics().events_deduplicated_total == 1

    other = (uuid.uuid4(),)
    _, events = run_sequence(
        closing_sequence("suspicious_looking", actor)
        + closing_sequence("suspicious_looking", other)
    )
    assert len(events) == 2


def _database_event_context(db: Session) -> tuple[ExamSession, tuple[SessionCandidate, ...]]:
    user = User(
        username="phase7-admin",
        password_hash="unused",
        full_name="Phase 7",
        role="ADMIN",
        is_active=True,
    )
    room = Room(code="P7", name="Phase 7", description=None, is_active=True)
    db.add_all([user, room])
    db.flush()
    seats = [
        Seat(
            room_id=room.id,
            code=f"A0{index}",
            x=0.1 + index * 0.3,
            y=0.1,
            width=0.2,
            height=0.2,
            sort_order=index,
            is_active=True,
        )
        for index in (1, 2)
    ]
    candidates = [
        Candidate(candidate_code=f"P7-{index}", full_name=f"Candidate {index}") for index in (1, 2)
    ]
    db.add_all([*seats, *candidates])
    db.flush()
    session = ExamSession(
        session_code="PHASE7-SESSION",
        exam_name="Phase 7",
        room_id=room.id,
        status="RUNNING",
        runtime_profile="gtx1650",
        created_by=user.id,
    )
    db.add(session)
    db.flush()
    assignments = tuple(
        SessionCandidate(session_id=session.id, candidate_id=candidate.id, seat_id=seat.id)
        for candidate, seat in zip(candidates, seats, strict=True)
    )
    db.add_all(assignments)
    db.commit()
    return session, assignments


def test_pair_event_persistence_is_pending_review_idempotent_and_audited(db: Session) -> None:
    session, assignments = _database_event_context(db)
    actors = tuple(sorted((item.id for item in assignments), key=str))
    candidate = AggregatedEvent(
        session_id=session.id,
        behavior=EventBehavior.COMMUNICATING,
        session_candidate_ids=actors,
        proposal_ids=("pair:test",),
        start_ms=1000,
        end_ms=3000,
        peak_ms=2000,
        peak_probability=0.92,
        active_probability_sum=1.72,
        active_prediction_count=2,
    )
    model, created = persist_ai_event(db, candidate)
    repeated, repeated_created = persist_ai_event(db, candidate)
    assert created is True and repeated_created is False and repeated.id == model.id
    assert model.source == "AI"
    assert model.status == "PENDING_REVIEW"
    assert model.created_by is None
    assert db.scalar(select(func.count(Event.id))) == 1
    assert db.scalar(select(func.count(EventActor.id))) == 2
    audit_count = db.scalar(
        select(func.count(AuditLog.id)).where(AuditLog.action == "AI_EVENT_CREATED")
    )
    assert audit_count == 1


def test_calibration_excludes_final_test_and_aligns_time_and_actors() -> None:
    actor = uuid.uuid4()
    session_id = uuid.uuid4()
    with pytest.raises(ValueError, match="final-test"):
        assert_calibration_only(["development", "final_test"])
    item = CalibrationPrediction(
        session_id=session_id,
        video_id="video-a",
        split="development",
        prediction=prediction("suspicious_looking", 1500, (actor,)),
    )
    truth = GroundTruthEvent(
        event_id="GT-1",
        session_id=session_id,
        video_id="video-a",
        split="development",
        behavior="suspicious_looking",
        start_ms=1000,
        end_ms=2000,
        session_candidate_ids=(actor,),
    )
    aligned, invalid = align_predictions([item], [truth])
    assert invalid == []
    assert aligned[0]["matched_gt_event_id"] == "GT-1"
    wrong_actor = replace(truth, event_id="GT-2", session_candidate_ids=(uuid.uuid4(),))
    aligned, _ = align_predictions([item], [wrong_actor])
    assert aligned[0]["matched_gt_behavior"] == "normal"


def test_calibration_search_is_reproducible_and_reports_invalid_annotations() -> None:
    actor = uuid.uuid4()
    session_id = uuid.uuid4()
    predictions = [
        CalibrationPrediction(
            session_id=session_id,
            video_id="video-a",
            split="validation",
            prediction=prediction("suspicious_looking", timestamp, (actor,), score=score),
        )
        for timestamp, score in ((0, 0.9), (1000, 0.9), (2000, 0.1), (3000, 0.1))
    ]
    truth = [
        GroundTruthEvent(
            event_id="GT-1",
            session_id=session_id,
            video_id="video-a",
            split="validation",
            behavior="suspicious_looking",
            start_ms=0,
            end_ms=2000,
            session_candidate_ids=(actor,),
        )
    ]
    space: dict[str, list[float | int]] = {
        "alpha": [1.0],
        "start_threshold": [0.7, 0.8],
        "keep_threshold": [0.5],
        "min_active_ms": [1000],
        "end_grace_ms": [1000],
        "merge_gap_ms": [500],
    }
    first, first_metrics = bounded_parameter_search(
        predictions, truth, duration_minutes=1.0, search_space=space
    )
    second, second_metrics = bounded_parameter_search(
        predictions, truth, duration_minutes=1.0, search_space=space
    )
    assert first.model_dump() == second.model_dump()
    assert first_metrics == second_metrics

    invalid_truth = [replace(truth[0], event_id="BAD", end_ms=-1)]
    _, invalid = align_predictions(predictions, invalid_truth)
    assert invalid == ["BAD: invalid time range"]


def test_event_metrics_exclude_normal_hard_negative_intervals_from_false_negatives() -> None:
    session_id = uuid.uuid4()
    actor = uuid.uuid4()
    normal = GroundTruthEvent(
        event_id="HARD-NEGATIVE",
        session_id=session_id,
        video_id="video-a",
        split="development",
        behavior="normal",
        start_ms=1000,
        end_ms=3000,
        session_candidate_ids=(actor,),
    )

    metrics = event_metrics([], [normal], duration_minutes=1.0)

    assert metrics["fn"] == 0
    assert metrics["f1"] == 0.0


def test_probability_distribution_maps_phone_runtime_name_to_raw_class_name() -> None:
    row = {
        "matched_gt_behavior": "using_phone/cheat_sheet",
        "prob_suspicious_looking": 0.01,
        "prob_communicating": 0.01,
        "prob_exchange_object": 0.01,
        "prob_using_phone_cheat_sheet": 0.96,
    }

    distributions = probability_distributions([row])

    phone = distributions["using_phone_cheat_sheet"]
    assert phone["positive"]["count"] == 1
    assert phone["positive"]["max"] == 0.96
    assert phone["negative"]["count"] == 0
