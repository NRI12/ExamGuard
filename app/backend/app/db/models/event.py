import enum
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, CreatedAtMixin, TimestampMixin


class EventSource(enum.StrEnum):
    AI = "AI"
    MANUAL = "MANUAL"


class BehaviorType(enum.StrEnum):
    SUSPICIOUS_LOOKING = "SUSPICIOUS_LOOKING"
    COMMUNICATING = "COMMUNICATING"
    EXCHANGE_OBJECT = "EXCHANGE_OBJECT"
    USING_PHONE_CHEAT_SHEET = "USING_PHONE_CHEAT_SHEET"
    OTHER = "OTHER"


class EventStatus(enum.StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DISPUTED = "DISPUTED"
    RESOLVED = "RESOLVED"


class ReviewDecision(enum.StrEnum):
    CONFIRM = "CONFIRM"
    DISMISS = "DISMISS"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class EvidenceKind(enum.StrEnum):
    CONTEXT_VIDEO = "CONTEXT_VIDEO"
    FOCUSED_VIDEO = "FOCUSED_VIDEO"
    SNAPSHOT = "SNAPSHOT"
    REPORT = "REPORT"


class Event(TimestampMixin, Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint("source IN ('AI', 'MANUAL')", name="event_source"),
        CheckConstraint(
            "behavior_type IS NULL OR behavior_type IN "
            "('SUSPICIOUS_LOOKING', 'COMMUNICATING', 'EXCHANGE_OBJECT', "
            "'USING_PHONE_CHEAT_SHEET', 'OTHER')",
            name="behavior_type",
        ),
        CheckConstraint(
            "status IN ('PENDING_REVIEW', 'CONFIRMED', 'DISMISSED', "
            "'NEEDS_REVIEW', 'DISPUTED', 'RESOLVED')",
            name="event_status",
        ),
        CheckConstraint("start_ms >= 0", name="start_nonnegative"),
        CheckConstraint("end_ms >= start_ms", name="time_range"),
        CheckConstraint(
            "peak_ms IS NULL OR (peak_ms >= start_ms AND peak_ms <= end_ms)",
            name="peak_range",
        ),
        CheckConstraint(
            "ai_confidence IS NULL OR (ai_confidence >= 0 AND ai_confidence <= 1)",
            name="ai_confidence_range",
        ),
        Index("ix_events_session_id_start_ms", "session_id", "start_ms"),
        Index("ix_events_session_id_status", "session_id", "status"),
        Index("ix_events_session_id_behavior_type", "session_id", "behavior_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    event_code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("exam_sessions.id"),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String(10), nullable=False)
    behavior_type: Mapped[str | None] = mapped_column(String(40))
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    peak_ms: Mapped[int | None] = mapped_column(BigInteger)
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id"))


class EventActor(CreatedAtMixin, Base):
    __tablename__ = "event_actors"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "session_candidate_id",
            name="uq_event_actors_event_candidate",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("events.id"), nullable=False)
    session_candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("session_candidates.id"),
        nullable=False,
    )
    role: Mapped[str | None] = mapped_column(String(100))


class EventReview(CreatedAtMixin, Base):
    __tablename__ = "event_reviews"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('CONFIRM', 'DISMISS', 'NEEDS_REVIEW')",
            name="review_decision",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("events.id"), nullable=False)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)


class EvidenceAsset(CreatedAtMixin, Base):
    __tablename__ = "evidence_assets"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('CONTEXT_VIDEO', 'FOCUSED_VIDEO', 'SNAPSHOT', 'REPORT')",
            name="evidence_kind",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("events.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    start_ms: Mapped[int | None] = mapped_column(BigInteger)
    end_ms: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    locked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
