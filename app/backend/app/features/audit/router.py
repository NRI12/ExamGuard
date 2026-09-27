from fastapi import APIRouter, Query

from app.features.audit.schemas import AuditLogResponse
from app.features.audit.service import list_audit_logs
from app.features.auth.dependencies import AuditReader, DatabaseSession
from app.shared.pagination import Page

router = APIRouter(prefix="/audit-logs", tags=["audit"])


@router.get("", response_model=Page[AuditLogResponse])
def get_audit_logs(
    _: AuditReader,
    db: DatabaseSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[AuditLogResponse]:
    entries, total = list_audit_logs(db, page, page_size)
    return Page(items=entries, page=page, page_size=page_size, total=total)
