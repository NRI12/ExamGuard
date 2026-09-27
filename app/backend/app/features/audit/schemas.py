import uuid
from datetime import datetime

from pydantic import BaseModel


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    actor_user_id: uuid.UUID | None
    actor_username: str | None
    actor_full_name: str | None
    action: str
    entity_type: str
    entity_id: uuid.UUID | None
    metadata: dict[str, object] | None
    created_at: datetime
