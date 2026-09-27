import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, CreatedAtMixin, TimestampMixin


class ExamSessionStatus(enum.StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class SessionSourceType(enum.StrEnum):
    CAMERA = "CAMERA"
    VIDEO_UPLOAD = "VIDEO_UPLOAD"


class ExamSession(TimestampMixin, Base):
    __tablename__ = "exam_sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT', 'READY', 'RUNNING', 'PAUSED', 'COMPLETED', 'CANCELLED', 'ERROR')",
            name="exam_session_status",
        ),
        CheckConstraint(
            "source_type IN ('CAMERA', 'VIDEO_UPLOAD')",
            name="exam_session_source_type",
        ),
        CheckConstraint(
            "(source_type = 'VIDEO_UPLOAD' AND camera_id IS NULL) OR "
            "(source_type = 'CAMERA' AND camera_id IS NOT NULL AND video_asset_id IS NOT NULL)",
            name="exam_session_source_consistency",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    exam_name: Mapped[str] = mapped_column(String(255), nullable=False)
    room_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("rooms.id"), nullable=False)
    video_asset_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("media_assets.id"))
    source_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=SessionSourceType.VIDEO_UPLOAD.value,
    )
    camera_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("cameras.id"))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    scheduled_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scheduled_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runtime_profile: Mapped[str | None] = mapped_column(String(100))
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)


class SessionCandidate(CreatedAtMixin, Base):
    __tablename__ = "session_candidates"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "candidate_id",
            name="uq_session_candidates_session_candidate",
        ),
        UniqueConstraint("session_id", "seat_id", name="uq_session_candidates_session_seat"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("exam_sessions.id"),
        nullable=False,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("candidates.id"),
        nullable=False,
    )
    seat_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("seats.id"), nullable=False)
