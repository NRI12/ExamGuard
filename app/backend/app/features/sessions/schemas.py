import uuid
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models.session import ExamSessionStatus, SessionSourceType


def normalize_code(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Session code cannot be blank")
    return normalized


def normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Exam name cannot be blank")
    return normalized


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class SessionCreate(BaseModel):
    session_code: str = Field(min_length=1, max_length=100)
    exam_name: str = Field(min_length=3, max_length=100)
    room_id: uuid.UUID
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    duration_minutes: int | None = Field(default=None, ge=15, le=360)
    status: ExamSessionStatus = ExamSessionStatus.DRAFT
    runtime_profile: str | None = Field(default=None, max_length=100)

    _normalize_code = field_validator("session_code")(normalize_code)
    _normalize_name = field_validator("exam_name")(normalize_name)
    _normalize_runtime_profile = field_validator("runtime_profile")(normalize_optional)

    @model_validator(mode="after")
    def validate_schedule(self) -> "SessionCreate":
        if self.duration_minutes is not None:
            if self.scheduled_start is None:
                raise ValueError("scheduled_start is required with duration_minutes")
            if self.scheduled_end is None:
                self.scheduled_end = self.scheduled_start + timedelta(minutes=self.duration_minutes)
        if (
            self.scheduled_start is not None
            and self.scheduled_end is not None
            and self.scheduled_end <= self.scheduled_start
        ):
            raise ValueError("scheduled_end must be after scheduled_start")
        return self


class SessionUpdate(BaseModel):
    session_code: str | None = Field(default=None, min_length=1, max_length=100)
    exam_name: str | None = Field(default=None, min_length=3, max_length=100)
    room_id: uuid.UUID | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    status: ExamSessionStatus | None = None
    runtime_profile: str | None = Field(default=None, max_length=100)
    source_type: SessionSourceType | None = None
    camera_id: uuid.UUID | None = None
    video_asset_id: uuid.UUID | None = None

    _normalize_code = field_validator("session_code")(normalize_code)
    _normalize_name = field_validator("exam_name")(normalize_name)
    _normalize_runtime_profile = field_validator("runtime_profile")(normalize_optional)

    @model_validator(mode="after")
    def require_change(self) -> "SessionUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        if "room_id" in self.model_fields_set and self.room_id is None:
            raise ValueError("room_id cannot be null")
        if "status" in self.model_fields_set and self.status is None:
            raise ValueError("status cannot be null")
        if "source_type" in self.model_fields_set and self.source_type is None:
            raise ValueError("source_type cannot be null")
        return self


class RoomSummary(BaseModel):
    id: uuid.UUID
    code: str
    name: str
    is_active: bool


class CandidateSummary(BaseModel):
    id: uuid.UUID
    candidate_code: str
    full_name: str
    class_name: str | None


class SeatSummary(BaseModel):
    id: uuid.UUID
    code: str
    is_active: bool


class SessionAssignmentResponse(BaseModel):
    id: uuid.UUID
    candidate: CandidateSummary
    seat: SeatSummary


class SessionReadiness(BaseModel):
    room_selected: bool
    room_active: bool
    seat_layout_available: bool
    active_seats: int
    candidates_assigned: int
    video_configured: bool
    monitoring_status: str
    can_mark_ready: bool


class MediaSummary(BaseModel):
    id: uuid.UUID
    original_filename: str
    media_url: str
    codec: str
    width: int
    height: int
    fps: float
    duration_ms: int
    size_bytes: int


class CameraSummary(BaseModel):
    id: uuid.UUID
    name: str
    is_active: bool


class UserSummary(BaseModel):
    id: uuid.UUID
    username: str
    full_name: str | None


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_code: str
    exam_name: str
    room_id: uuid.UUID
    room: RoomSummary
    source_type: SessionSourceType
    camera_id: uuid.UUID | None
    camera: CameraSummary | None
    video_asset_id: uuid.UUID | None
    video: MediaSummary | None
    status: ExamSessionStatus
    scheduled_start: datetime | None
    scheduled_end: datetime | None
    actual_start: datetime | None
    actual_end: datetime | None
    runtime_profile: str | None
    created_by: uuid.UUID
    created_by_user: UserSummary
    candidate_count: int
    assignments: list[SessionAssignmentResponse]
    readiness: SessionReadiness
    created_at: datetime
    updated_at: datetime


class SessionCandidateWrite(BaseModel):
    candidate_id: uuid.UUID
    seat_id: uuid.UUID


class SessionCandidatesUpdate(BaseModel):
    assignments: list[SessionCandidateWrite] = Field(max_length=500)
