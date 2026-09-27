from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import hash_password
from app.db.models.user import User, UserRole


@dataclass(frozen=True)
class BootstrapResult:
    created: bool
    username: str


def bootstrap_admin(db: Session, settings: Settings) -> BootstrapResult:
    username = settings.admin_username
    password = settings.admin_password
    if username is None or password is None:
        raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD are required")

    existing = db.scalar(select(User).where(User.username == username))
    if existing is not None:
        return BootstrapResult(created=False, username=existing.username)

    user = User(
        username=username,
        password_hash=hash_password(password.get_secret_value()),
        full_name=settings.admin_full_name,
        role=UserRole.ADMIN,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return BootstrapResult(created=True, username=user.username)
