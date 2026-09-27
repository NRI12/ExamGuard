import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Camera name cannot be blank")
    return normalized


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class CameraCreate(BaseModel):
    name: str = Field(min_length=3, max_length=255)
    room_id: uuid.UUID
    source_media_asset_id: uuid.UUID
    description: str | None = Field(default=None, max_length=5000)
    is_active: bool = True

    _normalize_name = field_validator("name")(normalize_name)
    _normalize_description = field_validator("description")(normalize_optional)


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=3, max_length=255)
    room_id: uuid.UUID | None = None
    source_media_asset_id: uuid.UUID | None = None
    description: str | None = Field(default=None, max_length=5000)
    is_active: bool | None = None

    _normalize_name = field_validator("name")(normalize_name)
    _normalize_description = field_validator("description")(normalize_optional)

    @model_validator(mode="after")
    def require_change(self) -> "CameraUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        if "room_id" in self.model_fields_set and self.room_id is None:
            raise ValueError("room_id cannot be null")
        if "source_media_asset_id" in self.model_fields_set and self.source_media_asset_id is None:
            raise ValueError("source_media_asset_id cannot be null")
        if "is_active" in self.model_fields_set and self.is_active is None:
            raise ValueError("is_active cannot be null")
        return self


class CameraRoomSummary(BaseModel):
    id: uuid.UUID
    code: str
    name: str


class CameraMediaSummary(BaseModel):
    id: uuid.UUID
    original_filename: str
    media_url: str
    mime_type: str | None
    codec: str
    width: int
    height: int
    fps: float
    duration_ms: int
    size_bytes: int
    sha256: str
    created_at: datetime


class CameraResponse(BaseModel):
    id: uuid.UUID
    name: str
    room_id: uuid.UUID
    room: CameraRoomSummary
    source_media_asset_id: uuid.UUID
    source_media: CameraMediaSummary | None
    description: str | None
    is_active: bool
    status: Literal["READY", "IN_USE", "ERROR", "DISABLED"]
    created_at: datetime
    updated_at: datetime
