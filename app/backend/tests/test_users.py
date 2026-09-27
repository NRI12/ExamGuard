from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models.audit import AuditLog
from app.db.models.user import User, UserRole


def add_user(db: Session, username: str, role: UserRole) -> User:
    user = User(
        username=username,
        password_hash=hash_password("existing-user-password-1"),
        full_name=username.title(),
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return user


def auth(settings: Settings, user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id, settings)}"}


def test_admin_can_create_filter_and_audit_users(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, "admin-users", UserRole.ADMIN)
    headers = auth(settings, admin)

    created = client.post(
        "/api/v1/users",
        headers=headers,
        json={
            "username": " New.Supervisor ",
            "full_name": "Nguyen Van A",
            "role": "SUPERVISOR",
            "password": "strong-password-123",
            "is_active": True,
        },
    )
    assert created.status_code == 201
    payload = created.json()
    assert payload["username"] == "new.supervisor"
    assert payload["role"] == "SUPERVISOR"
    assert "password" not in payload
    assert "password_hash" not in payload
    stored = db.scalar(select(User).where(User.username == "new.supervisor"))
    assert stored is not None
    assert verify_password("strong-password-123", stored.password_hash)

    filtered = client.get(
        "/api/v1/users?role=SUPERVISOR&is_active=true&q=new",
        headers=headers,
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1

    duplicate = client.post(
        "/api/v1/users",
        headers=headers,
        json={
            "username": "new.supervisor",
            "role": "REVIEWER",
            "password": "another-password-123",
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "USERNAME_EXISTS"

    audit = client.get("/api/v1/audit-logs", headers=headers)
    assert audit.status_code == 200
    assert any(entry["action"] == "USER_CREATED" for entry in audit.json()["items"])


def test_admin_role_and_status_changes_are_audited(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, "admin-editor", UserRole.ADMIN)
    target = add_user(db, "account-target", UserRole.ADMIN)
    headers = auth(settings, admin)
    target_headers = auth(settings, target)
    assert client.get("/api/v1/users", headers=target_headers).status_code == 200

    changed = client.patch(
        f"/api/v1/users/{target.id}",
        headers=headers,
        json={"full_name": "Updated Name", "role": "REVIEWER"},
    )
    assert changed.status_code == 200
    assert changed.json()["role"] == "REVIEWER"
    refreshed_access = client.get("/api/v1/users", headers=target_headers)
    assert refreshed_access.status_code == 403

    deactivated = client.patch(
        f"/api/v1/users/{target.id}",
        headers=headers,
        json={"is_active": False},
    )
    assert deactivated.status_code == 200
    assert deactivated.json()["is_active"] is False
    inactive_access = client.get("/api/v1/auth/me", headers=target_headers)
    assert inactive_access.status_code == 403
    assert inactive_access.json()["error"]["code"] == "USER_INACTIVE"
    reactivated = client.patch(
        f"/api/v1/users/{target.id}",
        headers=headers,
        json={"is_active": True},
    )
    assert reactivated.status_code == 200

    actions = set(db.scalars(select(AuditLog.action).where(AuditLog.entity_id == target.id)))
    assert {
        "USER_UPDATED",
        "USER_ROLE_CHANGED",
        "USER_DEACTIVATED",
        "USER_REACTIVATED",
    }.issubset(actions)


def test_user_management_requires_admin_permission(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    supervisor = add_user(db, "users-supervisor", UserRole.SUPERVISOR)
    reviewer = add_user(db, "users-reviewer", UserRole.REVIEWER)
    assert client.get("/api/v1/users", headers=auth(settings, supervisor)).status_code == 403
    assert client.get("/api/v1/audit-logs", headers=auth(settings, reviewer)).status_code == 403


def test_admin_cannot_remove_own_access_or_last_active_admin(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, "only-admin", UserRole.ADMIN)
    headers = auth(settings, admin)

    self_role = client.patch(
        f"/api/v1/users/{admin.id}",
        headers=headers,
        json={"role": "REVIEWER"},
    )
    assert self_role.status_code == 409
    assert self_role.json()["error"]["code"] == "USER_SELF_ROLE_CHANGE"
    self_deactivate = client.patch(
        f"/api/v1/users/{admin.id}",
        headers=headers,
        json={"is_active": False},
    )
    assert self_deactivate.status_code == 409
    assert self_deactivate.json()["error"]["code"] == "USER_SELF_DEACTIVATION"

    second_admin = add_user(db, "second-admin", UserRole.ADMIN)
    db.delete(admin)
    db.commit()
    response = client.patch(
        f"/api/v1/users/{second_admin.id}",
        headers=auth(settings, second_admin),
        json={"is_active": False},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "USER_SELF_DEACTIVATION"


def test_user_password_policy_is_validated(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, "admin-password", UserRole.ADMIN)
    response = client.post(
        "/api/v1/users",
        headers=auth(settings, admin),
        json={"username": "weak-user", "role": "REVIEWER", "password": "onlylettersxx"},
    )
    assert response.status_code == 422
