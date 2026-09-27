import uuid

from fastapi import APIRouter, Query, status

from app.db.models.user import UserRole
from app.features.auth.dependencies import DatabaseSession, UserManager
from app.features.users.schemas import UserCreate, UserResponse, UserUpdate
from app.features.users.service import create_user, list_users, update_user, user_or_error
from app.shared.pagination import Page

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=Page[UserResponse])
def get_users(
    _: UserManager,
    db: DatabaseSession,
    q: str | None = Query(default=None, max_length=255),
    role: UserRole | None = None,
    is_active: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[UserResponse]:
    users, total = list_users(db, q, role, is_active, page, page_size)
    return Page(
        items=[UserResponse.model_validate(user) for user in users],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def post_user(
    payload: UserCreate,
    actor: UserManager,
    db: DatabaseSession,
) -> UserResponse:
    return UserResponse.model_validate(create_user(db, payload, actor))


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: uuid.UUID,
    _: UserManager,
    db: DatabaseSession,
) -> UserResponse:
    return UserResponse.model_validate(user_or_error(db, user_id))


@router.patch("/{user_id}", response_model=UserResponse)
def patch_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    actor: UserManager,
    db: DatabaseSession,
) -> UserResponse:
    return UserResponse.model_validate(update_user(db, user_id, payload, actor))
