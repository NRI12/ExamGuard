import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class AppealStatus(enum.StrEnum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    UPHELD = "UPHELD"
    OVERTURNED = "OVERTURNED"
    CLOSED = "CLOSED"


class AppealCase(TimestampMixin, Base):
    __tablename__ = "appeal_cases"
    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN', 'UNDER_REVIEW', 'UPHELD', 'OVERTURNED', 'CLOSED')",
            name="appeal_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    session_candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("session_candidates.id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AppealEvent(Base):
    __tablename__ = "appeal_events"

    appeal_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("appeal_cases.id"),
        primary_key=True,
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("events.id"),
        primary_key=True,
    )
