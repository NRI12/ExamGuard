from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.models.user import User, UserRole


def test_rtsp_check_requires_authenticated_monitor(client: TestClient) -> None:
    response = client.post(
        "/api/v1/monitoring/rtsp/check",
        json={"url": "rtsp://camera.local/live"},
    )
    assert response.status_code == 401


def test_supervisor_can_check_rtsp_connection(
    client: TestClient,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add(
        User(
            username="rtsp-supervisor",
            password_hash=hash_password("rtsp-password"),
            full_name="RTSP Supervisor",
            role=UserRole.SUPERVISOR.value,
            is_active=True,
        )
    )
    db.commit()
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "rtsp-supervisor", "password": "rtsp-password"},
    )
    token = login.json()["access_token"]

    def connected(_: str, *, timeout_seconds: int) -> dict[str, Any]:
        assert timeout_seconds == 10
        return {
            "connected": True,
            "codec": "h264",
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "message": "Kết nối camera thành công.",
        }

    monkeypatch.setattr("app.features.monitoring.rtsp_router.probe_rtsp", connected)
    response = client.post(
        "/api/v1/monitoring/rtsp/check",
        headers={"Authorization": f"Bearer {token}"},
        json={"url": "rtsp://camera.local/live"},
    )
    assert response.status_code == 200
    assert response.json()["connected"] is True
