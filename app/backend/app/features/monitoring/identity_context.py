from __future__ import annotations

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.ai.seat_identity.types import (
    SeatCandidateBinding,
    SeatDefinition,
    SeatIdentityContext,
)
from app.db.models.candidate import Candidate
from app.db.models.room import Seat
from app.db.models.session import ExamSession, SessionCandidate


def load_seat_identity_context(
    db: Session,
    exam_session: ExamSession,
) -> SeatIdentityContext:
    """Load immutable runtime identity data once before the AI worker starts."""
    rows = db.execute(
        select(Seat, SessionCandidate, Candidate)
        .outerjoin(
            SessionCandidate,
            and_(
                SessionCandidate.seat_id == Seat.id,
                SessionCandidate.session_id == exam_session.id,
            ),
        )
        .outerjoin(Candidate, Candidate.id == SessionCandidate.candidate_id)
        .where(Seat.room_id == exam_session.room_id, Seat.is_active.is_(True))
        .order_by(Seat.sort_order.nulls_last(), Seat.code)
    ).all()
    seats = tuple(
        SeatDefinition(
            id=seat.id,
            code=seat.code,
            bbox_norm=(seat.x, seat.y, seat.x + seat.width, seat.y + seat.height),
        )
        for seat, _, _ in rows
    )
    bindings = tuple(
        SeatCandidateBinding(
            seat_id=seat.id,
            session_candidate_id=assignment.id,
            candidate_id=candidate.id,
            candidate_code=candidate.candidate_code,
            candidate_name=candidate.full_name,
        )
        for seat, assignment, candidate in rows
        if assignment is not None and candidate is not None
    )
    return SeatIdentityContext(
        session_id=exam_session.id,
        seats=seats,
        bindings=bindings,
    )
