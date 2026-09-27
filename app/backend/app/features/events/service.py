from __future__ import annotations

import uuid
from typing import Any

from fastapi import status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.event_aggregation.types import AggregatedEvent, EventBehavior
from app.core.errors import ApiError
from app.db.models.audit import AuditLog
from app.db.models.candidate import Candidate
from app.db.models.event import (
    BehaviorType,
    Event,
    EventActor,
    EventReview,
    EventSource,
    EventStatus,
    ReviewDecision,
)
from app.db.models.room import Seat
from app.db.models.session import ExamSession, SessionCandidate
from app.db.models.user import User
from app.shared.audit import AuditAction, AuditService


def ai_event_code(event: AggregatedEvent) -> str:
    """Stable retry key derived only from the finalized semantic event."""
    return f"AI-{event.fingerprint[:40].upper()}"


def persist_ai_event(db: Session, event: AggregatedEvent) -> tuple[Event, bool]:
    """Persist one AI event and actors atomically; return ``created=False`` on retry."""
    event_code = ai_event_code(event)
    existing = db.scalar(select(Event).where(Event.event_code == event_code))
    if existing is not None:
        return existing, False

    actor_ids = tuple(sorted(set(event.session_candidate_ids), key=str))
    # Pair behaviours have one or two actors: the R3 pair proposals name both people, the per-person
    # X3D classifier flags each participant of an interaction on its own.
    allowed_actor_counts = (
        {1, 2}
        if event.behavior in {EventBehavior.COMMUNICATING, EventBehavior.EXCHANGE_OBJECT}
        else {1}
    )
    if len(actor_ids) not in allowed_actor_counts:
        raise ValueError(
            f"{event.behavior.value} requires {sorted(allowed_actor_counts)} EventActor(s)"
        )
    rows = tuple(
        db.scalars(select(SessionCandidate).where(SessionCandidate.id.in_(actor_ids))).all()
    )
    if len(rows) != len(actor_ids) or any(row.session_id != event.session_id for row in rows):
        raise ValueError("AI event actors must be SessionCandidates from the event session")

    behavior = BehaviorType(event.behavior.value.upper())
    model = Event(
        event_code=event_code,
        session_id=event.session_id,
        source=EventSource.AI.value,
        behavior_type=behavior.value,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        peak_ms=event.peak_ms,
        ai_confidence=event.ai_confidence,
        status=EventStatus.PENDING_REVIEW.value,
        created_by=None,
    )
    db.add(model)
    db.flush()
    db.add_all(
        EventActor(event_id=model.id, session_candidate_id=actor_id, role=None)
        for actor_id in actor_ids
    )
    db.add(
        AuditLog(
            actor_user_id=None,
            action="AI_EVENT_CREATED",
            entity_type="EVENT",
            entity_id=model.id,
            audit_metadata={
                "event_code": event_code,
                "behavior": behavior.value,
                "session_id": str(event.session_id),
                "session_candidate_ids": [str(value) for value in actor_ids],
                "runtime_fingerprint": event.fingerprint,
            },
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(select(Event).where(Event.event_code == event_code))
        if concurrent is None:
            raise
        return concurrent, False
    return model, True


# ---------- read / review ----------

_STATUS_BY_DECISION = {
    ReviewDecision.CONFIRM.value: EventStatus.CONFIRMED.value,
    ReviewDecision.DISMISS.value: EventStatus.DISMISSED.value,
    ReviewDecision.NEEDS_REVIEW.value: EventStatus.NEEDS_REVIEW.value,
}
_REVIEWABLE = {
    EventStatus.PENDING_REVIEW.value,
    EventStatus.NEEDS_REVIEW.value,
    EventStatus.CONFIRMED.value,
    EventStatus.DISMISSED.value,
}


def event_or_error(db: Session, event_id: uuid.UUID) -> Event:
    event = db.get(Event, event_id)
    if event is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "EVENT_NOT_FOUND", "Event not found")
    return event


def list_events(
    db: Session,
    *,
    session_id: uuid.UUID | None,
    status_filter: str | None,
    behavior: str | None,
    page: int,
    page_size: int,
) -> tuple[list[Event], int]:
    query = select(Event)
    if session_id is not None:
        query = query.where(Event.session_id == session_id)
    if status_filter:
        query = query.where(Event.status == status_filter)
    if behavior:
        query = query.where(Event.behavior_type == behavior)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(
        query.order_by(Event.created_at.desc(), Event.start_ms.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return list(rows), total


def event_counts(db: Session, session_id: uuid.UUID | None) -> dict[str, int]:
    query = select(Event.status, func.count()).group_by(Event.status)
    if session_id is not None:
        query = query.where(Event.session_id == session_id)
    counts = {status_value: count for status_value, count in db.execute(query).tuples().all()}
    return {
        "total": sum(counts.values()),
        "pending_review": counts.get(EventStatus.PENDING_REVIEW.value, 0),
        "confirmed": counts.get(EventStatus.CONFIRMED.value, 0),
        "dismissed": counts.get(EventStatus.DISMISSED.value, 0),
        "needs_review": counts.get(EventStatus.NEEDS_REVIEW.value, 0),
    }


def event_payloads(
    db: Session, events: list[Event], *, with_reviews: bool = False
) -> list[dict[str, Any]]:
    """Events joined with their session and actors (candidate + seat) in a few queries."""
    if not events:
        return []
    ids = [event.id for event in events]
    sessions = {
        row.id: row
        for row in db.scalars(
            select(ExamSession).where(ExamSession.id.in_({e.session_id for e in events}))
        )
    }
    actors: dict[uuid.UUID, list[dict[str, Any]]] = {event_id: [] for event_id in ids}
    rows = db.execute(
        select(EventActor.event_id, SessionCandidate, Candidate, Seat.code)
        .join(SessionCandidate, SessionCandidate.id == EventActor.session_candidate_id)
        .join(Candidate, Candidate.id == SessionCandidate.candidate_id)
        .outerjoin(Seat, Seat.id == SessionCandidate.seat_id)
        .where(EventActor.event_id.in_(ids))
    ).all()
    for event_id, session_candidate, candidate, seat_code in rows:
        actors[event_id].append(
            {
                "session_candidate_id": session_candidate.id,
                "candidate_id": candidate.id,
                "candidate_code": candidate.candidate_code,
                "full_name": candidate.full_name,
                "seat_code": seat_code,
            }
        )
    reviews: dict[uuid.UUID, list[dict[str, Any]]] = {event_id: [] for event_id in ids}
    if with_reviews:
        for review, user in db.execute(
            select(EventReview, User)
            .join(User, User.id == EventReview.reviewer_id)
            .where(EventReview.event_id.in_(ids))
            .order_by(EventReview.created_at)
        ).all():
            reviews[review.event_id].append(
                {
                    "id": review.id,
                    "decision": review.decision,
                    "note": review.note,
                    "reviewer_id": user.id,
                    "reviewer_name": user.full_name or user.username,
                    "created_at": review.created_at,
                }
            )
    payloads = []
    for event in events:
        exam_session = sessions[event.session_id]
        payload: dict[str, Any] = {
            "id": event.id,
            "event_code": event.event_code,
            "session_id": event.session_id,
            "session_code": exam_session.session_code,
            "exam_name": exam_session.exam_name,
            "video_asset_id": exam_session.video_asset_id,
            "source": event.source,
            "behavior_type": event.behavior_type,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "peak_ms": event.peak_ms,
            "ai_confidence": event.ai_confidence,
            "status": event.status,
            "created_at": event.created_at,
            "updated_at": event.updated_at,
            "actors": sorted(actors[event.id], key=lambda a: a["candidate_code"]),
        }
        if with_reviews:
            payload["reviews"] = reviews[event.id]
        payloads.append(payload)
    return payloads


def review_event(
    db: Session,
    event_id: uuid.UUID,
    decision: str,
    note: str | None,
    reviewer: User,
) -> Event:
    """Append a review (earlier reviews are kept) and move the event to the decided status."""
    event = event_or_error(db, event_id)
    if event.status not in _REVIEWABLE:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "EVENT_NOT_REVIEWABLE",
            "Event is under dispute and cannot be reviewed here",
        )
    previous = event.status
    event.status = _STATUS_BY_DECISION[decision]
    review = EventReview(event_id=event.id, reviewer_id=reviewer.id, decision=decision, note=note)
    db.add(review)
    db.flush()
    AuditService.record(
        db,
        actor=reviewer,
        action=AuditAction.EVENT_REVIEWED,
        entity_type="EVENT",
        entity_id=event.id,
        metadata={
            "decision": decision,
            "previous_status": previous,
            "status": event.status,
            "review_id": str(review.id),
        },
    )
    db.commit()
    db.refresh(event)
    return event
