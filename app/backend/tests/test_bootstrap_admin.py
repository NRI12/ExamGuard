from sqlalchemy import select
from sqlalchemy.orm import Session

from app.bootstrap.admin import bootstrap_admin
from app.core.security import verify_password
from app.db.models.user import User, UserRole
from tests.conftest import make_settings


def test_admin_bootstrap_is_idempotent_and_does_not_modify_existing_user(db: Session) -> None:
    settings = make_settings(
        ADMIN_USERNAME="Admin",
        ADMIN_PASSWORD="initial-admin-password",
        ADMIN_FULL_NAME="Initial Administrator",
    )

    first = bootstrap_admin(db, settings)
    assert first.created

    user = db.scalar(select(User).where(User.username == "admin"))
    assert user is not None
    original_hash = user.password_hash
    user.role = UserRole.REVIEWER
    db.commit()

    second = bootstrap_admin(
        db,
        make_settings(
            ADMIN_USERNAME="admin",
            ADMIN_PASSWORD="different-password",
            ADMIN_FULL_NAME="Different Name",
        ),
    )
    db.refresh(user)

    assert not second.created
    assert user.role == UserRole.REVIEWER
    assert user.full_name == "Initial Administrator"
    assert user.password_hash == original_hash
    assert verify_password("initial-admin-password", user.password_hash)
