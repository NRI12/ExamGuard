import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.db.models.session import ExamSessionStatus
from app.features.auth.dependencies import (
    ApplicationSettings,
    DatabaseSession,
    SessionManager,
    SessionReader,
)
from app.features.sessions.schemas import (
    SessionCandidatesUpdate,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
)
from app.features.sessions.service import (
    create_session,
    list_sessions,
    replace_assignments,
    session_or_error,
    session_response,
    update_session,
)
from app.shared.pagination import Page

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=Page[SessionResponse])
def get_sessions(
    _: SessionReader,
    db: DatabaseSession,
    q: str | None = Query(default=None, max_length=255),
    session_status: Annotated[ExamSessionStatus | None, Query(alias="status")] = None,
    room_id: Annotated[uuid.UUID | None, Query()] = None,
    scheduled_date: Annotated[date | None, Query(alias="date")] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[SessionResponse]:
    sessions, total = list_sessions(
        db,
        query=q,
        session_status=session_status,
        page=page,
        page_size=page_size,
        room_id=room_id,
        scheduled_date=scheduled_date,
    )
    return Page(items=sessions, page=page, page_size=page_size, total=total)


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def post_session(
    payload: SessionCreate,
    actor: SessionManager,
    db: DatabaseSession,
) -> SessionResponse:
    return create_session(db, payload, actor)


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: uuid.UUID,
    _: SessionReader,
    db: DatabaseSession,
) -> SessionResponse:
    return session_response(db, session_or_error(db, session_id))


@router.patch("/{session_id}", response_model=SessionResponse)
def patch_session(
    session_id: uuid.UUID,
    payload: SessionUpdate,
    actor: SessionManager,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> SessionResponse:
    return update_session(db, session_id, payload, actor, settings)


@router.put("/{session_id}/candidates", response_model=SessionResponse)
def put_session_candidates(
    session_id: uuid.UUID,
    payload: SessionCandidatesUpdate,
    actor: SessionManager,
    db: DatabaseSession,
) -> SessionResponse:
    return replace_assignments(db, session_id, payload, actor)
