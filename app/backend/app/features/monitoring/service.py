from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.detector.yolo import PersonDetector
from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models.audit import AuditLog
from app.db.models.media import MediaAsset
from app.db.models.session import ExamSession, ExamSessionStatus, SessionSourceType
from app.db.models.user import User
from app.features.media.service import media_file_path
from app.features.monitoring.identity_context import load_seat_identity_context
from app.features.sessions.service import session_or_error, session_response
from app.monitoring.config import load_runtime_profile
from app.monitoring.manager import MonitoringRuntimeManager, RuntimeState
from app.shared.audit import AuditAction, AuditService


def _runtime_error(error: Exception) -> ApiError:
    message = str(error)
    if "model YOLO11n" in message or "CUDA" in message or "cấu hình" in message:
        return ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "AI_INITIALIZATION_FAILED", message)
    return ApiError(status.HTTP_409_CONFLICT, "INVALID_MONITORING_STATE", message)


def start_monitoring(
    db: Session,
    session_id: uuid.UUID,
    timestamp_ms: int,
    actor: User,
    settings: Settings,
    manager: MonitoringRuntimeManager,
) -> dict[str, object]:
    exam_session = session_or_error(db, session_id)
    if exam_session.status != ExamSessionStatus.READY.value:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_NOT_READY",
            "Only a READY session can start monitoring",
        )
    response = session_response(db, exam_session)
    if not response.readiness.room_active or exam_session.video_asset_id is None:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_NOT_READY",
            "An active room and source video are required",
        )
    asset = db.get(MediaAsset, exam_session.video_asset_id)
    if asset is None:
        raise ApiError(status.HTTP_409_CONFLICT, "MEDIA_NOT_FOUND", "Session video was not found")
    path = media_file_path(asset, settings)
    identity_context = load_seat_identity_context(db, exam_session)
    try:
        profile = load_runtime_profile(settings, exam_session.runtime_profile)
        PersonDetector.validate_environment(profile.detector)
        if manager.status(exam_session.id)["state"] in {
            RuntimeState.INITIALIZING.value,
            RuntimeState.RUNNING.value,
            RuntimeState.PAUSED.value,
        }:
            raise RuntimeError("Monitoring runtime đã hoạt động")
    except (RuntimeError, ValueError) as error:
        raise _runtime_error(error) from error

    exam_session.status = ExamSessionStatus.RUNNING.value
    exam_session.actual_start = exam_session.actual_start or datetime.now(UTC)
    AuditService.record(
        db,
        actor=actor,
        action=AuditAction.SESSION_STARTED,
        entity_type="EXAM_SESSION",
        entity_id=exam_session.id,
        metadata={"timestamp_ms": timestamp_ms, "runtime_profile": profile.profile},
    )
    db.commit()
    try:
        return manager.start(
            exam_session.id,
            path,
            profile,
            timestamp_ms,
            loop_source=exam_session.source_type == SessionSourceType.CAMERA.value,
            seat_identity_context=identity_context,
        )
    except (RuntimeError, ValueError) as error:
        exam_session.status = ExamSessionStatus.ERROR.value
        db.commit()
        raise _runtime_error(error) from error


def pause_monitoring(
    db: Session,
    session_id: uuid.UUID,
    timestamp_ms: int | None,
    actor: User,
    manager: MonitoringRuntimeManager,
) -> dict[str, object]:
    exam_session = session_or_error(db, session_id)
    if exam_session.status != ExamSessionStatus.RUNNING.value:
        raise ApiError(status.HTTP_409_CONFLICT, "INVALID_SESSION_STATE", "Session is not running")
    try:
        runtime_status = manager.pause(session_id, timestamp_ms)
    except RuntimeError as error:
        raise _runtime_error(error) from error
    exam_session.status = ExamSessionStatus.PAUSED.value
    AuditService.record(
        db,
        actor=actor,
        action=AuditAction.SESSION_PAUSED,
        entity_type="EXAM_SESSION",
        entity_id=session_id,
        metadata={"timestamp_ms": timestamp_ms},
    )
    db.commit()
    return runtime_status


def resume_monitoring(
    db: Session,
    session_id: uuid.UUID,
    timestamp_ms: int | None,
    actor: User,
    manager: MonitoringRuntimeManager,
) -> dict[str, object]:
    exam_session = session_or_error(db, session_id)
    if exam_session.status != ExamSessionStatus.PAUSED.value:
        raise ApiError(status.HTTP_409_CONFLICT, "INVALID_SESSION_STATE", "Session is not paused")
    try:
        runtime_status = manager.resume(session_id, timestamp_ms)
    except RuntimeError as error:
        raise _runtime_error(error) from error
    exam_session.status = ExamSessionStatus.RUNNING.value
    AuditService.record(
        db,
        actor=actor,
        action=AuditAction.SESSION_RESUMED,
        entity_type="EXAM_SESSION",
        entity_id=session_id,
        metadata={"timestamp_ms": timestamp_ms},
    )
    db.commit()
    return runtime_status


def stop_monitoring(
    db: Session,
    session_id: uuid.UUID,
    actor: User,
    manager: MonitoringRuntimeManager,
) -> dict[str, object]:
    exam_session = session_or_error(db, session_id)
    if exam_session.status not in {
        ExamSessionStatus.RUNNING.value,
        ExamSessionStatus.PAUSED.value,
    }:
        raise ApiError(status.HTTP_409_CONFLICT, "INVALID_SESSION_STATE", "Session is not active")
    try:
        runtime_status = manager.stop(session_id)
    except RuntimeError as error:
        raise _runtime_error(error) from error
    exam_session.status = ExamSessionStatus.COMPLETED.value
    exam_session.actual_end = datetime.now(UTC)
    AuditService.record(
        db,
        actor=actor,
        action=AuditAction.SESSION_STOPPED,
        entity_type="EXAM_SESSION",
        entity_id=session_id,
    )
    db.commit()
    return runtime_status


def seek_monitoring(
    session_id: uuid.UUID,
    timestamp_ms: int,
    manager: MonitoringRuntimeManager,
) -> dict[str, object]:
    try:
        return manager.seek(session_id, timestamp_ms)
    except RuntimeError as error:
        raise _runtime_error(error) from error


def terminal_database_update(
    db: Session,
    session_id: uuid.UUID,
    state: RuntimeState,
) -> None:
    exam_session = db.get(ExamSession, session_id)
    if exam_session is None:
        return
    if exam_session.status not in {
        ExamSessionStatus.RUNNING.value,
        ExamSessionStatus.PAUSED.value,
    }:
        return
    if state == RuntimeState.ERROR:
        exam_session.status = ExamSessionStatus.ERROR.value
    elif state == RuntimeState.COMPLETED:
        exam_session.status = ExamSessionStatus.COMPLETED.value
        exam_session.actual_end = datetime.now(UTC)
    db.commit()


def reset_interrupted_sessions(db: Session) -> int:
    """No monitoring runtime exists at start-up: sessions left RUNNING / PAUSED by a restart, a
    crash or a database restored elsewhere go back to READY (audited) to be monitored again."""
    stuck = db.scalars(
        select(ExamSession).where(
            ExamSession.status.in_(
                (ExamSessionStatus.RUNNING.value, ExamSessionStatus.PAUSED.value)
            )
        )
    ).all()
    for exam_session in stuck:
        previous = exam_session.status
        exam_session.status = ExamSessionStatus.READY.value
        db.add(
            AuditLog(
                actor_user_id=None,
                action="SESSION_MONITORING_INTERRUPTED",
                entity_type="EXAM_SESSION",
                entity_id=exam_session.id,
                audit_metadata={"previous_status": previous, "status": exam_session.status},
            )
        )
    db.commit()
    return len(stuck)
