import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.audit import AuditLog
from app.db.models.user import User


class AuditAction(StrEnum):
    ROOM_CREATED = "ROOM_CREATED"
    ROOM_UPDATED = "ROOM_UPDATED"
    SEAT_LAYOUT_UPDATED = "SEAT_LAYOUT_UPDATED"
    CAMERA_CREATED = "CAMERA_CREATED"
    CAMERA_UPDATED = "CAMERA_UPDATED"
    CANDIDATE_CREATED = "CANDIDATE_CREATED"
    CANDIDATE_UPDATED = "CANDIDATE_UPDATED"
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_UPDATED = "SESSION_UPDATED"
    SESSION_CANCELLED = "SESSION_CANCELLED"
    SESSION_CANDIDATES_UPDATED = "SESSION_CANDIDATES_UPDATED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_PAUSED = "SESSION_PAUSED"
    SESSION_RESUMED = "SESSION_RESUMED"
    SESSION_STOPPED = "SESSION_STOPPED"
    MEDIA_VIDEO_UPLOADED = "MEDIA_VIDEO_UPLOADED"
    EVENT_REVIEWED = "EVENT_REVIEWED"
    USER_CREATED = "USER_CREATED"
    USER_UPDATED = "USER_UPDATED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    USER_DEACTIVATED = "USER_DEACTIVATED"
    USER_REACTIVATED = "USER_REACTIVATED"


class AuditService:
    @staticmethod
    def record(
        db: Session,
        *,
        actor: User,
        action: AuditAction,
        entity_type: str,
        entity_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            actor_user_id=actor.id,
            action=action.value,
            entity_type=entity_type,
            entity_id=entity_id,
            audit_metadata=metadata,
        )
        db.add(entry)
        return entry
