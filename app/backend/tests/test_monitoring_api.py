from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from app.ai.detector.yolo import PersonDetector
from app.core.config import Settings
from app.core.security import hash_password
from app.db.models.audit import AuditLog
from app.db.models.camera import Camera
from app.db.models.candidate import Candidate
from app.db.models.media import MediaAsset
from app.db.models.room import Room, Seat
from app.db.models.session import ExamSession, SessionCandidate, SessionSourceType
from app.db.models.user import User, UserRole
from app.features.monitoring.schemas import DiagnosticsMessage, TrackingMessage
from app.monitoring.publisher import LatestWebSocketPublisher, Subscriber


def status_payload(session_id: uuid.UUID, state: str) -> dict[str, object]:
    return {
        "session_id": str(session_id),
        "state": state,
        "profile": "gtx1650",
        "error": None,
        "subscriber_count": 0,
        "queue_size": 0,
        "dropped_analysis_frames": 0,
        "diagnostics": None,
    }


class FakeRuntimeManager:
    def __init__(self) -> None:
        self.state = "INACTIVE"
        self.publisher = LatestWebSocketPublisher()

    def start(
        self,
        session_id: uuid.UUID,
        _: Path,
        __: object,
        ___: int,
        **kwargs: object,
    ) -> dict[str, object]:
        self.identity_context = kwargs.get("seat_identity_context")
        self.loop_source = kwargs.get("loop_source")
        self.state = "INITIALIZING"
        return status_payload(session_id, self.state)

    def pause(self, session_id: uuid.UUID, _: int | None = None) -> dict[str, object]:
        self.state = "PAUSED"
        return status_payload(session_id, self.state)

    def resume(self, session_id: uuid.UUID, _: int | None = None) -> dict[str, object]:
        self.state = "RUNNING"
        return status_payload(session_id, self.state)

    def seek(self, session_id: uuid.UUID, _: int) -> dict[str, object]:
        return status_payload(session_id, self.state)

    def stop(self, session_id: uuid.UUID) -> dict[str, object]:
        self.state = "COMPLETED"
        return status_payload(session_id, self.state)

    def status(self, session_id: uuid.UUID) -> dict[str, object]:
        return status_payload(session_id, self.state)

    def subscribe(self, session_id: uuid.UUID) -> Subscriber:
        subscriber = self.publisher.subscribe()
        runtime_instance_id = uuid.uuid4()
        self.publisher.publish(
            {
                "type": "tracking",
                "session_id": str(session_id),
                "runtime_instance_id": str(runtime_instance_id),
                "runtime_generation": 0,
                "tracker_instance_id": str(uuid.uuid4()),
                "tracking_seq": 1,
                "timestamp_ms": 1200,
                "frame_id": 30,
                "source_width": 1920,
                "source_height": 1080,
                "tracks": [
                    {
                        "track_id": 7,
                        "actor_id": "A0001",
                        "actor_state": "ACTIVE",
                        "recovered": False,
                        "bbox_norm": [0.1, 0.2, 0.3, 0.8],
                        "confidence": 0.9,
                        "identity": {
                            "state": "ASSIGNED",
                            "seat_id": str(uuid.uuid4()),
                            "seat_code": "A01",
                            "session_candidate_id": str(uuid.uuid4()),
                            "score": 0.9,
                        },
                    }
                ],
                "seats": [],
            }
        )
        return subscriber

    def unsubscribe(self, _: uuid.UUID, subscriber_id: uuid.UUID) -> None:
        self.publisher.unsubscribe(subscriber_id)


def prepare_ready_session(db: Session, settings: Settings) -> tuple[User, ExamSession]:
    user = User(
        username="monitor-supervisor",
        password_hash=hash_password("monitor-password"),
        full_name="Monitor Supervisor",
        role=UserRole.SUPERVISOR.value,
        is_active=True,
    )
    room = Room(code="MON-ROOM", name="Monitoring Room", is_active=True)
    candidate = Candidate(candidate_code="MON-001", full_name="Candidate One")
    db.add_all([user, room, candidate])
    db.flush()
    seat = Seat(
        room_id=room.id,
        code="A01",
        x=0.1,
        y=0.1,
        width=0.2,
        height=0.3,
        is_active=True,
    )
    media_id = uuid.uuid4()
    directory = settings.upload_root / str(media_id)
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "source.mp4"
    video_path.write_bytes(b"video")
    media = MediaAsset(
        id=media_id,
        original_filename="exam.mp4",
        stored_filename="source.mp4",
        storage_path=f"{media_id}/source.mp4",
        mime_type="video/mp4",
        codec="h264",
        width=1920,
        height=1080,
        fps=25,
        duration_ms=90_000,
        size_bytes=5,
        sha256="a" * 64,
        created_by=user.id,
    )
    db.add_all([seat, media])
    db.flush()
    exam_session = ExamSession(
        session_code="MON-SESSION",
        exam_name="Realtime Monitoring",
        room_id=room.id,
        video_asset_id=media.id,
        status="READY",
        runtime_profile="gtx1650",
        created_by=user.id,
    )
    db.add(exam_session)
    db.flush()
    db.add(SessionCandidate(session_id=exam_session.id, candidate_id=candidate.id, seat_id=seat.id))
    db.commit()
    return user, exam_session


def login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "monitor-supervisor", "password": "monitor-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_monitoring_lifecycle_transitions_and_audit(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PersonDetector, "validate_environment", lambda _: None)
    _, exam_session = prepare_ready_session(db, settings)
    manager = FakeRuntimeManager()
    client.app.state.monitoring_runtime = manager
    headers = login(client)
    base = f"/api/v1/sessions/{exam_session.id}"
    started = client.post(f"{base}/start", json={"timestamp_ms": 400}, headers=headers)
    assert started.status_code == 200
    assert started.json()["state"] == "INITIALIZING"
    assert manager.identity_context.session_id == exam_session.id  # type: ignore[union-attr]
    assert manager.identity_context.seats[0].code == "A01"  # type: ignore[union-attr]
    assert manager.loop_source is False
    assert db.get(ExamSession, exam_session.id).status == "RUNNING"  # type: ignore[union-attr]
    paused = client.post(f"{base}/pause", json={"timestamp_ms": 700}, headers=headers)
    assert paused.json()["state"] == "PAUSED"
    seeked = client.post(f"{base}/seek", json={"timestamp_ms": 5000}, headers=headers)
    assert seeked.status_code == 200
    resumed = client.post(f"{base}/resume", json={"timestamp_ms": 5000}, headers=headers)
    assert resumed.json()["state"] == "RUNNING"
    assert client.post(f"{base}/stop", headers=headers).json()["state"] == "COMPLETED"
    db.expire_all()
    stored = db.get(ExamSession, exam_session.id)
    assert stored is not None and stored.status == "COMPLETED" and stored.actual_end is not None
    actions = set(db.scalars(select(AuditLog.action)))
    assert {"SESSION_STARTED", "SESSION_PAUSED", "SESSION_RESUMED", "SESSION_STOPPED"} <= actions


def test_camera_session_starts_runtime_with_source_looping(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PersonDetector, "validate_environment", lambda _: None)
    _, exam_session = prepare_ready_session(db, settings)
    camera = Camera(
        name="Camera MON",
        room_id=exam_session.room_id,
        source_media_asset_id=exam_session.video_asset_id,
        is_active=True,
    )
    db.add(camera)
    db.flush()
    exam_session.source_type = SessionSourceType.CAMERA.value
    exam_session.camera_id = camera.id
    db.commit()

    manager = FakeRuntimeManager()
    client.app.state.monitoring_runtime = manager
    started = client.post(
        f"/api/v1/sessions/{exam_session.id}/start",
        json={"timestamp_ms": 0},
        headers=login(client),
    )

    assert started.status_code == 200
    assert manager.loop_source is True


def test_zero_seat_session_starts_monitoring(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PersonDetector, "validate_environment", lambda _: None)
    _, exam_session = prepare_ready_session(db, settings)
    db.execute(delete(SessionCandidate).where(SessionCandidate.session_id == exam_session.id))
    db.execute(delete(Seat).where(Seat.room_id == exam_session.room_id))
    db.commit()
    manager = FakeRuntimeManager()
    client.app.state.monitoring_runtime = manager
    headers = login(client)

    detail = client.get(f"/api/v1/sessions/{exam_session.id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["readiness"]["seat_layout_available"] is False
    assert detail.json()["readiness"]["candidates_assigned"] == 0
    assert detail.json()["readiness"]["can_mark_ready"] is True

    started = client.post(
        f"/api/v1/sessions/{exam_session.id}/start",
        json={"timestamp_ms": 0},
        headers=headers,
    )
    assert started.status_code == 200
    assert manager.identity_context.seats == ()  # type: ignore[union-attr]


def test_websocket_requires_auth_and_sends_typed_tracking(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    _, exam_session = prepare_ready_session(db, settings)
    manager = FakeRuntimeManager()
    manager.state = "RUNNING"
    client.app.state.monitoring_runtime = manager
    try:
        with client.websocket_connect(f"/ws/monitoring/{exam_session.id}"):
            raise AssertionError("anonymous websocket unexpectedly connected")
    except WebSocketDisconnect as error:
        assert error.code == 4401
    login(client)
    with client.websocket_connect(f"/ws/monitoring/{exam_session.id}") as websocket:
        payload = websocket.receive_json()
        parsed = TrackingMessage.model_validate(payload)
        assert parsed.tracks[0].track_id == 7
        assert parsed.timestamp_ms == 1200


def test_diagnostics_schema_rejects_fabricated_invalid_ranges() -> None:
    payload: dict[str, Any] = {
        "type": "diagnostics",
        "session_id": str(uuid.uuid4()),
        "runtime_instance_id": str(uuid.uuid4()),
        "runtime_generation": 0,
        "worker_instance_id": str(uuid.uuid4()),
        "tracker_instance_id": str(uuid.uuid4()),
        "tracking_seq": 4,
        "latest_frame_id": 30,
        "latest_timestamp_ms": 1200,
        "raw_detection_count": 6,
        "active_track_count": 6,
        "source_fps": 25,
        "target_analysis_fps": 12.5,
        "analysis_fps": 11.8,
        "detector_ms": 58.4,
        "tracker_ms": 2.1,
        "pipeline_ms": 67.2,
        "seat_assignment_ms": 0.3,
        "analysis_lag_ms": 81,
        "gpu_util_pct": None,
        "vram_used_mb": None,
        "cpu_util_pct": 30,
        "ram_used_mb": 6700,
        "dropped_analysis_frames": 18,
        "queue_size": 1,
        "assigned_tracks": 5,
        "tentative_tracks": 0,
        "unassigned_tracks": 1,
        "occupied_seats": 5,
        "grace_seats": 0,
        "empty_seats": 1,
        "seat_switches": 0,
        "identity_recoveries": 1,
        "profile": "gtx1650",
    }
    assert DiagnosticsMessage.model_validate(payload).gpu_util_pct is None
