from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from app.core.errors import ApiError
from app.core.permissions import Permission, has_permission
from app.db.models.session import ExamSession
from app.features.auth.dependencies import (
    WEBSOCKET_COOKIE_NAME,
    ApplicationSettings,
    DatabaseSession,
    SessionMonitor,
    TrackingReader,
    resolve_user_from_token,
)
from app.features.monitoring.schemas import MonitoringPosition, MonitoringStatusResponse
from app.features.monitoring.service import (
    pause_monitoring,
    resume_monitoring,
    seek_monitoring,
    start_monitoring,
    stop_monitoring,
)
from app.features.sessions.service import session_or_error
from app.monitoring.manager import MonitoringRuntimeManager

router = APIRouter(prefix="/sessions", tags=["monitoring"])
websocket_router = APIRouter()


def get_runtime_manager(request: Request) -> MonitoringRuntimeManager:
    return request.app.state.monitoring_runtime  # type: ignore[no-any-return]


RuntimeManager = Annotated[MonitoringRuntimeManager, Depends(get_runtime_manager)]


@router.post("/{session_id}/start", response_model=MonitoringStatusResponse)
def start_session_monitoring(
    session_id: uuid.UUID,
    actor: SessionMonitor,
    db: DatabaseSession,
    settings: ApplicationSettings,
    manager: RuntimeManager,
    payload: MonitoringPosition | None = None,
) -> dict[str, object]:
    return start_monitoring(
        db,
        session_id,
        payload.timestamp_ms if payload else 0,
        actor,
        settings,
        manager,
    )


@router.post("/{session_id}/pause", response_model=MonitoringStatusResponse)
def pause_session_monitoring(
    session_id: uuid.UUID,
    actor: SessionMonitor,
    db: DatabaseSession,
    manager: RuntimeManager,
    payload: MonitoringPosition | None = None,
) -> dict[str, object]:
    return pause_monitoring(
        db,
        session_id,
        payload.timestamp_ms if payload else None,
        actor,
        manager,
    )


@router.post("/{session_id}/resume", response_model=MonitoringStatusResponse)
def resume_session_monitoring(
    session_id: uuid.UUID,
    actor: SessionMonitor,
    db: DatabaseSession,
    manager: RuntimeManager,
    payload: MonitoringPosition | None = None,
) -> dict[str, object]:
    return resume_monitoring(
        db,
        session_id,
        payload.timestamp_ms if payload else None,
        actor,
        manager,
    )


@router.post("/{session_id}/stop", response_model=MonitoringStatusResponse)
def stop_session_monitoring(
    session_id: uuid.UUID,
    actor: SessionMonitor,
    db: DatabaseSession,
    manager: RuntimeManager,
) -> dict[str, object]:
    return stop_monitoring(db, session_id, actor, manager)


@router.post("/{session_id}/seek", response_model=MonitoringStatusResponse)
def seek_session_monitoring(
    session_id: uuid.UUID,
    payload: MonitoringPosition,
    _: SessionMonitor,
    db: DatabaseSession,
    manager: RuntimeManager,
) -> dict[str, object]:
    session_or_error(db, session_id)
    return seek_monitoring(session_id, payload.timestamp_ms, manager)


@router.get("/{session_id}/monitoring-status", response_model=MonitoringStatusResponse)
def get_session_monitoring_status(
    session_id: uuid.UUID,
    _: TrackingReader,
    db: DatabaseSession,
    manager: RuntimeManager,
) -> dict[str, object]:
    session_or_error(db, session_id)
    return manager.status(session_id)


@websocket_router.websocket("/ws/monitoring/{session_id}")
async def monitoring_socket(
    websocket: WebSocket,
    session_id: uuid.UUID,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> None:
    token = websocket.cookies.get(WEBSOCKET_COOKIE_NAME)
    if token is None:
        await websocket.close(code=4401, reason="Authentication required")
        return
    try:
        user = resolve_user_from_token(db, settings, token)
    except ApiError as error:
        await websocket.close(code=4403 if error.status_code == 403 else 4401, reason=error.message)
        return
    if not has_permission(user.role, Permission.TRACKING_READ):
        await websocket.close(code=4403, reason="You do not have permission to view tracking.")
        return
    if db.get(ExamSession, session_id) is None:
        await websocket.close(code=4404, reason="Session not found")
        return
    manager: MonitoringRuntimeManager = websocket.app.state.monitoring_runtime
    try:
        subscriber = manager.subscribe(session_id)
    except RuntimeError:
        await websocket.close(code=4409, reason="Monitoring runtime is not active")
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(await subscriber.queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(session_id, subscriber.id)
