import enum

from app.db.models.user import UserRole


class Permission(enum.StrEnum):
    DASHBOARD_READ = "dashboard.read"
    ROOM_READ = "room.read"
    ROOM_MANAGE = "room.manage"
    CANDIDATE_READ = "candidate.read"
    CANDIDATE_MANAGE = "candidate.manage"
    SESSION_READ = "session.read"
    SESSION_MANAGE = "session.manage"
    SESSION_MONITOR = "session.monitor"
    MEDIA_READ = "media.read"
    MEDIA_UPLOAD = "media.upload"
    TRACKING_READ = "tracking.read"
    EVENT_READ = "event.read"
    EVENT_CREATE_MANUAL = "event.create_manual"
    EVENT_REVIEW = "event.review"
    EVIDENCE_READ = "evidence.read"
    EVIDENCE_MANAGE = "evidence.manage"
    APPEAL_READ = "appeal.read"
    APPEAL_MANAGE = "appeal.manage"
    REPORT_READ = "report.read"
    REPORT_EXPORT = "report.export"
    USER_MANAGE = "user.manage"
    AUDIT_READ = "audit.read"
    AUDIT_READ_RELEVANT = "audit.read_relevant"
    SYSTEM_MANAGE = "system.manage"
    DIAGNOSTICS_READ = "diagnostics.read"


ALL_PERMISSIONS = frozenset(Permission)

ROLE_PERMISSIONS: dict[UserRole, frozenset[Permission]] = {
    UserRole.SUPERVISOR: frozenset(
        {
            Permission.DASHBOARD_READ,
            Permission.ROOM_READ,
            Permission.CANDIDATE_READ,
            Permission.CANDIDATE_MANAGE,
            Permission.SESSION_READ,
            Permission.SESSION_MANAGE,
            Permission.SESSION_MONITOR,
            Permission.MEDIA_READ,
            Permission.MEDIA_UPLOAD,
            Permission.TRACKING_READ,
            Permission.EVENT_READ,
            Permission.EVENT_CREATE_MANUAL,
            Permission.EVIDENCE_READ,
            Permission.REPORT_READ,
        }
    ),
    UserRole.REVIEWER: frozenset(
        {
            Permission.DASHBOARD_READ,
            Permission.ROOM_READ,
            Permission.CANDIDATE_READ,
            Permission.SESSION_READ,
            Permission.TRACKING_READ,
            Permission.EVENT_READ,
            Permission.EVENT_REVIEW,
            Permission.EVIDENCE_READ,
            Permission.EVIDENCE_MANAGE,
            Permission.APPEAL_READ,
            Permission.APPEAL_MANAGE,
            Permission.REPORT_READ,
            Permission.AUDIT_READ_RELEVANT,
        }
    ),
    UserRole.ADMIN: ALL_PERMISSIONS,
}


def permissions_for_role(role: str | UserRole) -> frozenset[Permission]:
    try:
        resolved_role = UserRole(role)
    except ValueError:
        return frozenset()
    return ROLE_PERMISSIONS[resolved_role]


def has_permission(role: str | UserRole, permission: Permission) -> bool:
    return permission in permissions_for_role(role)
