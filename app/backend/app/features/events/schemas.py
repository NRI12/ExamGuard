import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EventActorResponse(BaseModel):
    session_candidate_id: uuid.UUID
    candidate_id: uuid.UUID
    candidate_code: str
    full_name: str
    seat_code: str | None


class EventReviewResponse(BaseModel):
    id: uuid.UUID
    decision: str
    note: str | None
    reviewer_id: uuid.UUID
    reviewer_name: str
    created_at: datetime


class EventResponse(BaseModel):
    id: uuid.UUID
    event_code: str
    session_id: uuid.UUID
    session_code: str
    exam_name: str
    video_asset_id: uuid.UUID | None
    source: str
    behavior_type: str | None
    start_ms: int
    end_ms: int
    peak_ms: int | None
    ai_confidence: float | None
    status: str
    created_at: datetime
    updated_at: datetime
    actors: list[EventActorResponse]


class EventDetailResponse(EventResponse):
    reviews: list[EventReviewResponse]


class EventReviewCreate(BaseModel):
    decision: Literal["CONFIRM", "DISMISS", "NEEDS_REVIEW"]
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class EventCounts(BaseModel):
    total: int
    pending_review: int
    confirmed: int
    dismissed: int
    needs_review: int
