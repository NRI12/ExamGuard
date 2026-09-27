import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models.user import UserRole


def normalize_username(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("Username cannot be blank")
    if re.fullmatch(r"[a-z0-9._-]+", normalized) is None:
        raise ValueError("Username contains unsupported characters")
    return normalized


def normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def validate_password(value: str) -> str:
    if not any(character.isalpha() for character in value) or not any(
        character.isdigit() for character in value
    ):
        raise ValueError("Password must contain at least one letter and one number")
    return value


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    full_name: str | None = Field(default=None, max_length=255)
    role: UserRole
    password: str = Field(min_length=12, max_length=128)
    is_active: bool = True

    _normalize_username = field_validator("username")(normalize_username)
    _normalize_name = field_validator("full_name")(normalize_optional_name)
    _validate_password = field_validator("password")(validate_password)


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=255)
    role: UserRole | None = None
    is_active: bool | None = None

    _normalize_name = field_validator("full_name")(normalize_optional_name)

    @model_validator(mode="after")
    def require_change(self) -> "UserUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one field is required")
        if "role" in self.model_fields_set and self.role is None:
            raise ValueError("Role cannot be null")
        if "is_active" in self.model_fields_set and self.is_active is None:
            raise ValueError("Active status cannot be null")
        return self


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
