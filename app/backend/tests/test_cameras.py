import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token, hash_password
from app.db.models.media import MediaAsset
from app.db.models.user import User, UserRole


def add_user(db: Session, role: UserRole, username: str) -> User:
    user = User(
        username=username,
        password_hash=hash_password("camera-test-password"),
        full_name=username.title(),
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return user


def auth(settings: Settings, user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id, settings)}"}


def add_media(db: Session, settings: Settings, user: User) -> MediaAsset:
    media_id = uuid.uuid4()
    directory = settings.upload_root / str(media_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.mp4").write_bytes(b"valid-source-placeholder")
    media = MediaAsset(
        id=media_id,
        original_filename="S00_exam.mp4",
        stored_filename="source.mp4",
        storage_path=f"{media_id}/source.mp4",
        mime_type="video/mp4",
        codec="h264",
        width=1920,
        height=1080,
        fps=25.0,
        duration_ms=60_000,
        size_bytes=24,
        sha256="a" * 64,
        created_by=user.id,
    )
    db.add(media)
    db.commit()
    return media


def test_admin_camera_crud_supervisor_selection_and_no_hard_delete(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "camera-admin")
    supervisor = add_user(db, UserRole.SUPERVISOR, "camera-supervisor")
    admin_headers = auth(settings, admin)
    supervisor_headers = auth(settings, supervisor)
    room = client.post(
        "/api/v1/rooms",
        json={"code": "CAM-A101", "name": "Phòng A101", "is_active": True},
        headers=admin_headers,
    ).json()
    media = add_media(db, settings, admin)

    forbidden = client.post(
        "/api/v1/cameras",
        json={
            "name": "Camera A101",
            "room_id": room["id"],
            "source_media_asset_id": str(media.id),
            "is_active": True,
        },
        headers=supervisor_headers,
    )
    assert forbidden.status_code == 403

    created = client.post(
        "/api/v1/cameras",
        json={
            "name": "  Camera A101 - Tổng quan  ",
            "room_id": room["id"],
            "source_media_asset_id": str(media.id),
            "description": "Camera mô phỏng",
            "is_active": True,
        },
        headers=admin_headers,
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Camera A101 - Tổng quan"
    assert body["status"] == "READY"
    assert body["source_media"]["fps"] == 25.0
    assert body["source_media"]["media_url"].endswith("/content")

    selectable = client.get(
        f"/api/v1/cameras?room_id={room['id']}&is_active=true",
        headers=supervisor_headers,
    )
    assert selectable.status_code == 200
    assert [item["id"] for item in selectable.json()["items"]] == [body["id"]]

    disabled = client.patch(
        f"/api/v1/cameras/{body['id']}",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "DISABLED"
    assert client.delete(f"/api/v1/cameras/{body['id']}", headers=admin_headers).status_code == 405


def test_session_camera_source_snapshots_media_and_requires_same_room(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "camera-session-admin")
    headers = auth(settings, admin)
    rooms = [
        client.post(
            "/api/v1/rooms",
            json={"code": code, "name": code, "is_active": True},
            headers=headers,
        ).json()
        for code in ("CAM-R1", "CAM-R2")
    ]
    media = add_media(db, settings, admin)
    camera = client.post(
        "/api/v1/cameras",
        json={
            "name": "Camera phòng 1",
            "room_id": rooms[0]["id"],
            "source_media_asset_id": str(media.id),
            "is_active": True,
        },
        headers=headers,
    ).json()
    session = client.post(
        "/api/v1/sessions",
        json={"session_code": "CAM-SESSION", "exam_name": "Thi camera", "room_id": rooms[0]["id"]},
        headers=headers,
    ).json()
    sourced = client.patch(
        f"/api/v1/sessions/{session['id']}",
        json={"source_type": "CAMERA", "camera_id": camera["id"]},
        headers=headers,
    )
    assert sourced.status_code == 200
    assert sourced.json()["source_type"] == "CAMERA"
    assert sourced.json()["camera"]["name"] == "Camera phòng 1"
    assert sourced.json()["video_asset_id"] == str(media.id)

    other_session = client.post(
        "/api/v1/sessions",
        json={
            "session_code": "CAM-OTHER",
            "exam_name": "Thi phòng khác",
            "room_id": rooms[1]["id"],
        },
        headers=headers,
    ).json()
    mismatch = client.patch(
        f"/api/v1/sessions/{other_session['id']}",
        json={"source_type": "CAMERA", "camera_id": camera["id"]},
        headers=headers,
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "CAMERA_NOT_IN_ROOM"
