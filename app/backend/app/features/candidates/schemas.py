import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_code(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("Candidate code cannot be blank")
    return normalized


def normalize_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Full name cannot be blank")
    return normalized


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class CandidateCreate(BaseModel):
    candidate_code: str = Field(min_length=1, max_length=100)
    full_name: str = Field(min_length=1, max_length=255)
    class_name: str | None = Field(default=None, max_length=255)
    note: str | None = Field(default=None, max_length=5000)

    _normalize_code = field_validator("candidate_code")(normalize_code)
    _normalize_name = field_validator("full_name")(normalize_name)
    _normalize_optional = field_validator("class_name", "note")(normalize_optional)


class CandidateUpdate(BaseModel):
    candidate_code: str | None = Field(default=None, min_length=1, max_length=100)
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    class_name: str | None = Field(default=None, max_length=255)
    note: str | None = Field(default=None, max_length=5000)

    _normalize_code = field_validator("candidate_code")(normalize_code)
    _normalize_name = field_validator("full_name")(normalize_name)
    _normalize_optional = field_validator("class_name", "note")(normalize_optional)

    @model_validator(mode="after")
    def require_change(self) -> "CandidateUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        return self


class CandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    candidate_code: str
    full_name: str
    class_name: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime
