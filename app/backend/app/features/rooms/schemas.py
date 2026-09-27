import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_code(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Code cannot be blank")
    return normalized


def normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Name cannot be blank")
    return normalized


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class RoomCreate(BaseModel):
    code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    is_active: bool = True

    _normalize_code = field_validator("code")(normalize_code)
    _normalize_name = field_validator("name")(normalize_name)
    _normalize_description = field_validator("description")(normalize_optional)


class RoomUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=1, max_length=100)
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    is_active: bool | None = None

    _normalize_code = field_validator("code")(normalize_code)
    _normalize_name = field_validator("name")(normalize_name)
    _normalize_description = field_validator("description")(normalize_optional)

    @model_validator(mode="after")
    def require_change(self) -> "RoomUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        return self


class RoomResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    code: str
    name: str
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class SeatWrite(BaseModel):
    id: uuid.UUID | None = None
    code: str = Field(min_length=1, max_length=100)
    x: float
    y: float
    width: float
    height: float
    sort_order: int | None = None
    is_active: bool = True

    _normalize_code = field_validator("code")(normalize_code)


class SeatLayoutUpdate(BaseModel):
    seats: list[SeatWrite] = Field(max_length=500)


class SeatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    room_id: uuid.UUID
    code: str
    x: float
    y: float
    width: float
    height: float
    sort_order: int | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
