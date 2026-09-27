import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.permissions import ALL_PERMISSIONS, Permission, has_permission, permissions_for_role
from app.core.security import create_access_token, hash_password
from app.db.models.user import User, UserRole


def add_user(db: Session, role: UserRole) -> User:
    user = User(
        username=f"rbac-{role.value.lower()}",
        password_hash=hash_password("rbac-test-password"),
        full_name=f"RBAC {role.value.title()}",
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return user


def auth(settings: Settings, user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id, settings)}"}


def test_role_permission_matrix() -> None:
    assert has_permission(UserRole.SUPERVISOR, Permission.SESSION_MONITOR)
    assert not has_permission(UserRole.SUPERVISOR, Permission.EVENT_REVIEW)
    assert has_permission(UserRole.REVIEWER, Permission.EVENT_REVIEW)
    assert not has_permission(UserRole.REVIEWER, Permission.SESSION_MONITOR)
    assert not has_permission(UserRole.REVIEWER, Permission.DIAGNOSTICS_READ)
    assert permissions_for_role(UserRole.ADMIN) == ALL_PERMISSIONS
    assert not has_permission("UNKNOWN", Permission.DASHBOARD_READ)


def test_supervisor_candidate_access_and_room_management_denial(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    supervisor = add_user(db, UserRole.SUPERVISOR)
    headers = auth(settings, supervisor)

    rooms = client.get("/api/v1/rooms", headers=headers)
    assert rooms.status_code == 200

    forbidden_room = client.post(
        "/api/v1/rooms",
        json={"code": "RBAC-ROOM", "name": "RBAC Room", "is_active": True},
        headers=headers,
    )
    assert forbidden_room.status_code == 403
    assert forbidden_room.json()["error"]["code"] == "FORBIDDEN"
    forbidden_layout = client.put(
        f"/api/v1/rooms/{uuid.uuid4()}/seats",
        json={"seats": []},
        headers=headers,
    )
    assert forbidden_layout.status_code == 403

    candidate = client.post(
        "/api/v1/candidates",
        json={"candidate_code": "RBAC-SV01", "full_name": "Supervisor Candidate"},
        headers=headers,
    )
    assert candidate.status_code == 201
    updated = client.patch(
        f"/api/v1/candidates/{candidate.json()['id']}",
        json={"class_name": "RBAC"},
        headers=headers,
    )
    assert updated.status_code == 200


def test_reviewer_is_read_only_for_setup_and_cannot_control_monitoring(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN)
    reviewer = add_user(db, UserRole.REVIEWER)
    admin_headers = auth(settings, admin)
    reviewer_headers = auth(settings, reviewer)
    candidate = client.post(
        "/api/v1/candidates",
        json={"candidate_code": "RBAC-RV01", "full_name": "Reviewer Candidate"},
        headers=admin_headers,
    )
    assert candidate.status_code == 201

    assert client.get("/api/v1/candidates", headers=reviewer_headers).status_code == 200
    forbidden_candidate = client.patch(
        f"/api/v1/candidates/{candidate.json()['id']}",
        json={"note": "not allowed"},
        headers=reviewer_headers,
    )
    assert forbidden_candidate.status_code == 403
    assert forbidden_candidate.json()["error"]["message"] == (
        "You do not have permission to perform this action."
    )

    session_id = uuid.uuid4()
    for action in ("start", "pause", "resume", "stop", "seek"):
        payload = {"timestamp_ms": 0} if action in {"start", "pause", "resume", "seek"} else None
        response = client.post(
            f"/api/v1/sessions/{session_id}/{action}",
            json=payload,
            headers=reviewer_headers,
        )
        assert response.status_code == 403


def test_admin_can_manage_rooms(client: TestClient, db: Session, settings: Settings) -> None:
    admin = add_user(db, UserRole.ADMIN)
    response = client.post(
        "/api/v1/rooms",
        json={"code": "RBAC-ADMIN", "name": "Admin Room", "is_active": True},
        headers=auth(settings, admin),
    )
    assert response.status_code == 201
