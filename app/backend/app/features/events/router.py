import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.db.models.event import BehaviorType, EventStatus
from app.features.auth.dependencies import DatabaseSession, EventReader, EventReviewer
from app.features.events.schemas import (
    EventCounts,
    EventDetailResponse,
    EventResponse,
    EventReviewCreate,
)
from app.features.events.service import (
    event_counts,
    event_or_error,
    event_payloads,
    list_events,
    review_event,
)
from app.shared.pagination import Page

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=Page[EventResponse])
def get_events(
    _: EventReader,
    db: DatabaseSession,
    session_id: Annotated[uuid.UUID | None, Query()] = None,
    status_filter: Annotated[EventStatus | None, Query(alias="status")] = None,
    behavior: Annotated[BehaviorType | None, Query()] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[EventResponse]:
    events, total = list_events(
        db,
        session_id=session_id,
        status_filter=status_filter.value if status_filter else None,
        behavior=behavior.value if behavior else None,
        page=page,
        page_size=page_size,
    )
    return Page(
        items=[EventResponse.model_validate(item) for item in event_payloads(db, events)],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/counts", response_model=EventCounts)
def get_event_counts(
    _: EventReader,
    db: DatabaseSession,
    session_id: Annotated[uuid.UUID | None, Query()] = None,
) -> EventCounts:
    return EventCounts.model_validate(event_counts(db, session_id))


@router.get("/{event_id}", response_model=EventDetailResponse)
def get_event(event_id: uuid.UUID, _: EventReader, db: DatabaseSession) -> EventDetailResponse:
    event = event_or_error(db, event_id)
    return EventDetailResponse.model_validate(event_payloads(db, [event], with_reviews=True)[0])


@router.post(
    "/{event_id}/reviews",
    response_model=EventDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_event_review(
    event_id: uuid.UUID,
    payload: EventReviewCreate,
    reviewer: EventReviewer,
    db: DatabaseSession,
) -> EventDetailResponse:
    event = review_event(db, event_id, payload.decision, payload.note, reviewer)
    return EventDetailResponse.model_validate(event_payloads(db, [event], with_reviews=True)[0])
