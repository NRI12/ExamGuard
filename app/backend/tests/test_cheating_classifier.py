from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.cheating_classifier.model import map_box, square_crop_box
from app.ai.cheating_classifier.runtime import CheatingClassifierRuntime
from app.ai.domain import Track
from app.ai.event_aggregation.types import AggregatedEvent, EventBehavior
from app.ai.seat_identity.types import AssignmentState, TrackIdentity
from app.core.config import Settings
from app.core.security import create_access_token
from app.db.models.audit import AuditLog
from app.db.models.event import Event, EventReview
from app.db.models.user import User
from app.features.events.service import persist_ai_event
from app.monitoring.config import CheatingClassifierConfig, load_runtime_profile_from_paths
from tests.test_event_aggregation import _database_event_context

CLASSES = ("normal", "looking", "interaction", "phone_cheatsheet")
NORMAL = np.array([0.94, 0.02, 0.02, 0.02], np.float32)
PHONE = np.array([0.04, 0.03, 0.03, 0.90], np.float32)


class FakeModel:
    """Stands in for the X3D-L checkpoint: returns scripted probabilities, no GPU."""

    device = "cpu"
    classes = CLASSES
    num_frames = 16
    model_name = "x3d_l_v1_best"

    def __init__(self) -> None:
        self.probs = NORMAL
        self.batches: list[int] = []

    def ensure_loaded(self) -> None:
        pass

    def classify(self, clips: np.ndarray) -> np.ndarray:
        assert clips.shape[1:] == (16, 356, 356, 3) and clips.dtype == np.uint8
        self.batches.append(len(clips))
        return np.repeat(self.probs[None], len(clips), axis=0)


def track(candidate: uuid.UUID | None, actor: str = "A1") -> Track:
    identity = (
        TrackIdentity(AssignmentState.ASSIGNED, seat_code="A01", session_candidate_id=candidate)
        if candidate
        else TrackIdentity(AssignmentState.UNASSIGNED)
    )
    return Track(
        track_id=7,
        bbox_norm=(0.30, 0.30, 0.45, 0.80),
        confidence=0.9,
        identity=identity,
        actor_id=actor,
    )


def run(
    runtime: CheatingClassifierRuntime,
    tracks: tuple[Track, ...],
    start_ms: int,
    end_ms: int,
    generation: int = 0,
) -> None:
    frame = np.zeros((360, 640, 3), np.uint8)
    for timestamp_ms in range(start_ms, end_ms, 100):  # 10 FPS analysis, like a slow GPU profile
        runtime.update(frame, tracks, timestamp_ms, generation)
        deadline = time.monotonic() + 5  # classifier thread: finish each window first
        while not runtime.idle() and time.monotonic() < deadline:
            time.sleep(0.002)


def make_runtime(
    model: FakeModel, events: list[AggregatedEvent], messages: list[dict]
) -> CheatingClassifierRuntime:
    return CheatingClassifierRuntime(
        session_id=uuid.uuid4(),
        runtime_instance_id=uuid.uuid4(),
        config=CheatingClassifierConfig(enabled=True),
        model=model,  # type: ignore[arg-type]
        publish=messages.append,
        emit_event=events.append,
    )


def test_box_mapping_and_square_crop_match_the_training_geometry() -> None:
    config = CheatingClassifierConfig()
    box = map_box((100.0, 200.0, 300.0, 700.0), 1920, 1080, config)
    np.testing.assert_allclose(box, [79.8, 173.8, 318.8, 378.0], atol=0.1)
    assert square_crop_box([box, box], 1920, 1080, 0.15) == (61, 138, 335, 412)


def test_sustained_cheating_of_a_seated_candidate_becomes_one_event() -> None:
    model, events, messages = FakeModel(), [], []
    runtime = make_runtime(model, events, messages)
    candidate = uuid.uuid4()
    run(runtime, (track(candidate),), 0, 4000)  # first window at 3.0 s (16 samples of 0.2 s)
    model.probs = PHONE
    run(runtime, (track(candidate),), 4000, 9000)
    model.probs = NORMAL
    run(runtime, (track(candidate),), 9000, 14000)
    assert runtime.close()
    assert len(events) == 1
    event = events[0]
    assert event.behavior is EventBehavior.USING_PHONE_CHEAT_SHEET
    assert event.session_candidate_ids == (candidate,)
    assert 5000 <= event.start_ms <= event.peak_ms <= event.end_ms <= 11000
    assert event.active_prediction_count >= 2
    predictions = [m for m in messages if m["type"] == "cheat_prediction"]
    assert predictions and predictions[-1]["classes"] == list(CLASSES)
    assert any(actor["alert"] for m in predictions for actor in m["actors"])
    assert all(size >= 1 for size in model.batches)


def test_alert_of_an_unseated_actor_is_shown_but_not_persisted() -> None:
    model, events, messages = FakeModel(), [], []
    runtime = make_runtime(model, events, messages)
    model.probs = PHONE
    run(runtime, (track(None, "B1"),), 0, 8000)
    assert runtime.close()
    assert events == []
    diagnostics = runtime.diagnostics()
    assert diagnostics["alerts_total"] == 1 and diagnostics["alerts_without_candidate"] == 1


def test_short_spike_below_min_windows_and_seek_reset() -> None:
    model, events, messages = FakeModel(), [], []
    runtime = make_runtime(model, events, messages)
    candidate = uuid.uuid4()
    run(runtime, (track(candidate),), 0, 3100)
    model.probs = PHONE
    run(runtime, (track(candidate),), 3100, 4100)  # one alert window only
    model.probs = NORMAL
    run(runtime, (track(candidate),), 4100, 9000)
    assert events == []
    model.probs = PHONE
    run(runtime, (track(candidate),), 9000, 13100)
    run(runtime, (track(candidate),), 20000, 20500, generation=1)  # seek: open alert is finalised
    assert len(events) == 1
    assert runtime.close()


def test_single_actor_interaction_event_is_persisted(db: Session) -> None:
    session, assignments = _database_event_context(db)
    event = AggregatedEvent(
        session_id=session.id,
        behavior=EventBehavior.COMMUNICATING,
        session_candidate_ids=(assignments[0].id,),
        proposal_ids=("actor:A1",),
        start_ms=1000,
        end_ms=4000,
        peak_ms=3000,
        peak_probability=0.93,
        active_probability_sum=2.7,
        active_prediction_count=3,
    )
    model, created = persist_ai_event(db, event)
    assert created and model.behavior_type == "COMMUNICATING" and model.status == "PENDING_REVIEW"


def test_events_api_lists_and_reviews_ai_events(
    client: TestClient, db: Session, settings: Settings
) -> None:
    session, assignments = _database_event_context(db)
    persist_ai_event(
        db,
        AggregatedEvent(
            session_id=session.id,
            behavior=EventBehavior.USING_PHONE_CHEAT_SHEET,
            session_candidate_ids=(assignments[1].id,),
            proposal_ids=("actor:A2",),
            start_ms=5000,
            end_ms=9000,
            peak_ms=7000,
            peak_probability=0.97,
            active_probability_sum=3.6,
            active_prediction_count=4,
        ),
    )
    admin = db.scalar(select(User).where(User.username == "phase7-admin"))
    assert admin is not None
    headers = {"Authorization": f"Bearer {create_access_token(admin.id, settings)}"}

    listed = client.get("/api/v1/events", params={"session_id": str(session.id)}, headers=headers)
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["behavior_type"] == "USING_PHONE_CHEAT_SHEET"
    assert item["actors"][0]["candidate_code"] == "P7-2" and item["actors"][0]["seat_code"] == "A02"
    assert item["session_code"] == "PHASE7-SESSION"

    counts = client.get("/api/v1/events/counts", headers=headers).json()
    assert counts == {
        "total": 1,
        "pending_review": 1,
        "confirmed": 0,
        "dismissed": 0,
        "needs_review": 0,
    }

    reviewed = client.post(
        f"/api/v1/events/{item['id']}/reviews",
        json={"decision": "CONFIRM", "note": "  Seen on video  "},
        headers=headers,
    )
    assert reviewed.status_code == 201, reviewed.text
    detail = reviewed.json()
    assert detail["status"] == "CONFIRMED"
    assert detail["reviews"][0]["note"] == "Seen on video"
    assert db.scalar(select(func.count(EventReview.id))) == 1
    assert (
        db.scalar(select(func.count(AuditLog.id)).where(AuditLog.action == "EVENT_REVIEWED")) == 1
    )
    assert db.scalar(select(Event.status)) == "CONFIRMED"

    filtered = client.get("/api/v1/events", params={"status": "PENDING_REVIEW"}, headers=headers)
    assert filtered.json()["total"] == 0
    assert client.get("/api/v1/events").status_code == 401


def test_rtx3060_profile_enables_the_x3d_classifier() -> None:
    backend = Path(__file__).resolve().parents[1]
    profile = load_runtime_profile_from_paths(
        config_root=backend / "configs",
        model_root=Path("/srv/models"),
        profile_name="rtx3060",
    )
    config = profile.cheating_classifier
    assert config.enabled and not profile.event_detection.enabled
    assert config.model == Path("/srv/models/cheating/x3d_l_v1_best.pt")
    assert (config.rule.start_threshold, config.rule.keep_threshold) == (0.85, 0.70)


def test_sessions_left_running_by_a_restart_go_back_to_ready(db: Session) -> None:
    from app.features.monitoring.service import reset_interrupted_sessions

    session, _ = _database_event_context(db)  # status RUNNING, no runtime after a restart
    assert reset_interrupted_sessions(db) == 1
    db.refresh(session)
    assert session.status == "READY"
    audit = db.scalar(select(AuditLog).where(AuditLog.action == "SESSION_MONITORING_INTERRUPTED"))
    assert audit is not None and audit.audit_metadata["previous_status"] == "RUNNING"
    assert reset_interrupted_sessions(db) == 0
