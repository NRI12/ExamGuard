import uuid
from collections.abc import Callable
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.permissions import Permission, has_permission
from app.core.security import decode_access_token
from app.db.models.user import User
from app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)
MEDIA_COOKIE_NAME = "examguard_media_access"
WEBSOCKET_COOKIE_NAME = "examguard_ws_access"
DatabaseSession = Annotated[Session, Depends(get_db)]
ApplicationSettings = Annotated[Settings, Depends(get_settings)]


def authentication_error() -> ApiError:
    return ApiError(
        status.HTTP_401_UNAUTHORIZED,
        "AUTHENTICATION_REQUIRED",
        "Authentication required",
    )


def resolve_user_from_token(db: Session, settings: Settings, token: str) -> User:
    try:
        user_id: uuid.UUID = decode_access_token(token, settings)
    except (jwt.InvalidTokenError, ValueError):
        raise authentication_error() from None
    user = db.get(User, user_id)
    if user is None:
        raise authentication_error()
    if not user.is_active:
        raise ApiError(status.HTTP_403_FORBIDDEN, "USER_INACTIVE", "User is inactive")
    return user


def get_current_user(
    db: DatabaseSession,
    settings: ApplicationSettings,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise authentication_error()
    return resolve_user_from_token(db, settings, credentials.credentials)


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_media_user(
    db: DatabaseSession,
    settings: ApplicationSettings,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    media_cookie: Annotated[str | None, Cookie(alias=MEDIA_COOKIE_NAME)] = None,
) -> User:
    token = credentials.credentials if credentials is not None else media_cookie
    if token is None:
        raise authentication_error()
    return resolve_user_from_token(db, settings, token)


MediaUser = Annotated[User, Depends(get_media_user)]


def require_permission(permission: Permission) -> Callable[[CurrentUser], User]:
    def permission_dependency(current_user: CurrentUser) -> User:
        if not has_permission(current_user.role, permission):
            raise ApiError(
                status.HTTP_403_FORBIDDEN,
                "FORBIDDEN",
                "You do not have permission to perform this action.",
            )
        return current_user

    return permission_dependency


RoomReader = Annotated[User, Depends(require_permission(Permission.ROOM_READ))]
RoomManager = Annotated[User, Depends(require_permission(Permission.ROOM_MANAGE))]
CandidateReader = Annotated[User, Depends(require_permission(Permission.CANDIDATE_READ))]
CandidateManager = Annotated[User, Depends(require_permission(Permission.CANDIDATE_MANAGE))]
SessionReader = Annotated[User, Depends(require_permission(Permission.SESSION_READ))]
SessionManager = Annotated[User, Depends(require_permission(Permission.SESSION_MANAGE))]
SessionMonitor = Annotated[User, Depends(require_permission(Permission.SESSION_MONITOR))]
MediaReader = Annotated[User, Depends(require_permission(Permission.MEDIA_READ))]
MediaUploader = Annotated[User, Depends(require_permission(Permission.MEDIA_UPLOAD))]
TrackingReader = Annotated[User, Depends(require_permission(Permission.TRACKING_READ))]
EventReader = Annotated[User, Depends(require_permission(Permission.EVENT_READ))]
EventReviewer = Annotated[User, Depends(require_permission(Permission.EVENT_REVIEW))]
UserManager = Annotated[User, Depends(require_permission(Permission.USER_MANAGE))]
AuditReader = Annotated[User, Depends(require_permission(Permission.AUDIT_READ))]
SystemManager = Annotated[User, Depends(require_permission(Permission.SYSTEM_MANAGE))]


def get_media_reader(media_user: MediaUser) -> User:
    if not has_permission(media_user.role, Permission.MEDIA_READ):
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            "FORBIDDEN",
            "You do not have permission to perform this action.",
        )
    return media_user


MediaReaderWithCookie = Annotated[User, Depends(get_media_reader)]
