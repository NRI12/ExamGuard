from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.audit import AuditLog
from app.db.models.user import User
from app.features.audit.schemas import AuditLogResponse


def list_audit_logs(
    db: Session,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[AuditLogResponse], int]:
    total = db.scalar(select(func.count()).select_from(AuditLog)) or 0
    rows = db.execute(
        select(AuditLog, User.username, User.full_name)
        .outerjoin(User, User.id == AuditLog.actor_user_id)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [
        AuditLogResponse(
            id=audit.id,
            actor_user_id=audit.actor_user_id,
            actor_username=username,
            actor_full_name=full_name,
            action=audit.action,
            entity_type=audit.entity_type,
            entity_id=audit.entity_id,
            metadata=audit.audit_metadata,
            created_at=audit.created_at,
        )
        for audit, username, full_name in rows
    ]
    return items, total
