from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.models.user import User, UserRole


def create_user(db: Session, *, active: bool = True) -> User:
    user = User(
        username="supervisor",
        password_hash=hash_password("phase-one-password"),
        full_name="Phase One Supervisor",
        role=UserRole.SUPERVISOR,
        is_active=active,
    )
    db.add(user)
    db.commit()
    return user


def test_login_success_and_auth_me(client: TestClient, db: Session) -> None:
    user = create_user(db)

    login = client.post(
        "/api/v1/auth/login",
        json={"username": " SUPERVISOR ", "password": "phase-one-password"},
    )

    assert login.status_code == 200
    payload = login.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["id"] == str(user.id)
    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["username"] == "supervisor"


def test_login_rejects_invalid_credentials(client: TestClient, db: Session) -> None:
    create_user(db)

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "supervisor", "password": "incorrect"},
    )

    assert response.status_code == 401


def test_login_rejects_inactive_user(client: TestClient, db: Session) -> None:
    create_user(db, active=False)

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "supervisor", "password": "phase-one-password"},
    )

    assert response.status_code == 403


def test_auth_me_rejects_missing_and_invalid_tokens(client: TestClient) -> None:
    assert client.get("/api/v1/auth/me").status_code == 401
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401
