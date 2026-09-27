import uuid

from fastapi import status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ApiError
from app.core.security import hash_password
from app.db.models.user import User, UserRole
from app.features.users.schemas import UserCreate, UserUpdate
from app.shared.audit import AuditAction, AuditService


def user_or_error(db: Session, user_id: uuid.UUID) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "USER_NOT_FOUND", "User was not found")
    return user


def list_users(
    db: Session,
    query: str | None = None,
    role: UserRole | None = None,
    is_active: bool | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[User], int]:
    statement = select(User)
    if query and (term := query.strip()):
        pattern = f"%{term}%"
        statement = statement.where(
            or_(User.username.ilike(pattern), User.full_name.ilike(pattern))
        )
    if role is not None:
        statement = statement.where(User.role == role.value)
    if is_active is not None:
        statement = statement.where(User.is_active.is_(is_active))
    total = db.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0
    users = list(
        db.scalars(
            statement.order_by(User.username).offset((page - 1) * page_size).limit(page_size)
        )
    )
    return users, total


def create_user(db: Session, payload: UserCreate, actor: User) -> User:
    if db.scalar(select(User.id).where(User.username == payload.username)):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "USERNAME_EXISTS",
            "Username already exists",
            field_name="username",
        )
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role.value,
        is_active=payload.is_active,
    )
    db.add(user)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.USER_CREATED,
            entity_type="USER",
            entity_id=user.id,
            metadata={
                "username": user.username,
                "role": user.role,
                "is_active": user.is_active,
            },
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "USERNAME_EXISTS",
            "Username already exists",
            field_name="username",
        ) from error
    db.refresh(user)
    return user


def update_user(
    db: Session,
    user_id: uuid.UUID,
    payload: UserUpdate,
    actor: User,
) -> User:
    user = user_or_error(db, user_id)
    requested = payload.model_dump(exclude_unset=True)
    requested_role = requested.get("role", user.role)
    new_role = requested_role.value if isinstance(requested_role, UserRole) else requested_role
    new_active = requested.get("is_active", user.is_active)

    if user.id == actor.id and new_role != user.role:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "USER_SELF_ROLE_CHANGE",
            "You cannot change your own role",
        )
    if user.id == actor.id and not new_active:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "USER_SELF_DEACTIVATION",
            "You cannot deactivate your own account",
        )

    removes_active_admin = (
        user.role == UserRole.ADMIN.value
        and user.is_active
        and (new_role != UserRole.ADMIN.value or not new_active)
    )
    if removes_active_admin:
        active_admins = (
            db.scalar(
                select(func.count())
                .select_from(User)
                .where(User.role == UserRole.ADMIN.value, User.is_active.is_(True))
            )
            or 0
        )
        if active_admins <= 1:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "LAST_ACTIVE_ADMIN",
                "The last active administrator cannot be changed",
            )

    old_role = user.role
    old_active = user.is_active
    changed_fields: list[str] = []
    if "full_name" in requested and requested["full_name"] != user.full_name:
        user.full_name = requested["full_name"]
        changed_fields.append("full_name")
    if "role" in requested and new_role != user.role:
        user.role = new_role
        changed_fields.append("role")
    if "is_active" in requested and new_active != user.is_active:
        user.is_active = new_active
        changed_fields.append("is_active")

    if not changed_fields:
        return user

    db.flush()
    if "full_name" in changed_fields:
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.USER_UPDATED,
            entity_type="USER",
            entity_id=user.id,
            metadata={"username": user.username, "changed_fields": ["full_name"]},
        )
    if "role" in changed_fields:
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.USER_ROLE_CHANGED,
            entity_type="USER",
            entity_id=user.id,
            metadata={
                "username": user.username,
                "previous_role": old_role,
                "new_role": user.role,
            },
        )
    if "is_active" in changed_fields:
        AuditService.record(
            db,
            actor=actor,
            action=(
                AuditAction.USER_REACTIVATED if user.is_active else AuditAction.USER_DEACTIVATED
            ),
            entity_type="USER",
            entity_id=user.id,
            metadata={
                "username": user.username,
                "previous_active": old_active,
                "new_active": user.is_active,
            },
        )
    db.commit()
    db.refresh(user)
    return user
