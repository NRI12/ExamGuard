import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token, hash_password
from app.db.models.audit import AuditLog
from app.db.models.media import MediaAsset
from app.db.models.room import Room
from app.db.models.session import ExamSession, ExamSessionStatus
from app.db.models.user import User, UserRole
from app.video.probe import VideoMetadata, VideoProbeError


def add_user(db: Session, role: UserRole = UserRole.ADMIN) -> User:
    user = User(
        username=f"media-{role.value.lower()}-{uuid.uuid4().hex[:8]}",
        password_hash=hash_password("phase-three-password"),
        full_name="Media User",
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return user


def auth(settings: Settings, user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id, settings)}"}


def metadata() -> VideoMetadata:
    return VideoMetadata("h264", 1920, 1080, 25.0, 2520000, ("mov", "mp4"))


def stored_media(
    db: Session,
    settings: Settings,
    user: User,
    content: bytes = b"0123456789",
) -> MediaAsset:
    media_id = uuid.uuid4()
    directory = settings.upload_root / str(media_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.mp4").write_bytes(content)
    asset = MediaAsset(
        id=media_id,
        original_filename="exam.mp4",
        stored_filename="source.mp4",
        storage_path=f"{media_id}/source.mp4",
        mime_type="video/mp4",
        codec="h264",
        width=32,
        height=24,
        fps=25.0,
        duration_ms=1000,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        created_by=user.id,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def test_upload_stores_metadata_hash_file_and_audit(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = add_user(db)
    monkeypatch.setattr(
        "app.features.media.service.probe_video",
        lambda *_args, **_kwargs: metadata(),
    )
    content = b"small-mp4-fixture"
    response = client.post(
        "/api/v1/media/videos",
        headers=auth(settings, user),
        files={"file": ("session_S08.mp4", content, "application/octet-stream")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["codec"] == "h264"
    assert body["width"] == 1920
    assert body["sha256"] == hashlib.sha256(content).hexdigest()
    assert "storage_path" not in body
    asset = db.get(MediaAsset, uuid.UUID(body["id"]))
    assert asset is not None
    assert (settings.upload_root / asset.storage_path).read_bytes() == content
    assert db.scalar(
        select(func.count(AuditLog.id)).where(AuditLog.action == "MEDIA_VIDEO_UPLOADED")
    ) == 1


def test_upload_rejects_empty_extension_invalid_content_and_large_file(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = add_user(db)
    headers = auth(settings, user)
    empty = client.post(
        "/api/v1/media/videos",
        headers=headers,
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert empty.json()["error"]["code"] == "VIDEO_EMPTY"
    extension = client.post(
        "/api/v1/media/videos",
        headers=headers,
        files={"file": ("video.avi", b"data", "video/mp4")},
    )
    assert extension.json()["error"]["code"] == "INVALID_VIDEO_EXTENSION"
    monkeypatch.setattr(
        "app.features.media.service.probe_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(VideoProbeError("bad")),
    )
    invalid = client.post(
        "/api/v1/media/videos",
        headers=headers,
        files={"file": ("fake.mp4", b"not-video", "video/mp4")},
    )
    assert invalid.json()["error"]["code"] == "INVALID_VIDEO"
    settings.max_upload_bytes = 4
    large = client.post(
        "/api/v1/media/videos",
        headers=headers,
        files={"file": ("large.mp4", b"12345", "video/mp4")},
    )
    assert large.status_code == 413
    assert db.scalar(select(func.count(MediaAsset.id))) == 0


def test_upload_cleans_final_file_when_database_commit_fails(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = add_user(db)
    settings.upload_root.mkdir(parents=True, exist_ok=True)
    before = {path.relative_to(settings.upload_root) for path in settings.upload_root.rglob("*")}
    monkeypatch.setattr(
        "app.features.media.service.probe_video",
        lambda *_args, **_kwargs: metadata(),
    )

    def fail_commit() -> None:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(db, "commit", fail_commit)
    response = client.post(
        "/api/v1/media/videos",
        headers=auth(settings, user),
        files={"file": ("db-failure.mp4", b"video", "video/mp4")},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MEDIA_SAVE_FAILED"
    after = {path.relative_to(settings.upload_root) for path in settings.upload_root.rglob("*")}
    assert after == before


def test_content_supports_cookie_full_range_invalid_range_and_auth(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    user = add_user(db)
    asset = stored_media(db, settings, user)
    login = client.post(
        "/api/v1/auth/login",
        json={"username": user.username, "password": "phase-three-password"},
    )
    assert login.status_code == 200
    full = client.get(f"/api/v1/media/{asset.id}/content")
    assert full.status_code == 200
    assert full.content == b"0123456789"
    assert full.headers["accept-ranges"] == "bytes"
    partial = client.get(
        f"/api/v1/media/{asset.id}/content",
        headers={"Range": "bytes=2-5"},
    )
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.headers["content-length"] == "4"
    invalid = client.get(
        f"/api/v1/media/{asset.id}/content",
        headers={"Range": "bytes=20-30"},
    )
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == "bytes */10"
    client.cookies.clear()
    assert client.get(f"/api/v1/media/{asset.id}/content").status_code == 401


def test_calibration_frame_uses_media_reader_and_validates_timestamp(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = add_user(db, UserRole.SUPERVISOR)
    asset = stored_media(db, settings, supervisor)
    calls: list[tuple[str, int]] = []

    def fake_extract(path: object, timestamp_ms: int) -> bytes:
        calls.append((str(path), timestamp_ms))
        return b"jpeg-frame"

    monkeypatch.setattr("app.features.media.router.extract_video_frame", fake_extract)
    headers = auth(settings, supervisor)
    response = client.get(
        f"/api/v1/media/{asset.id}/frame?timestamp_ms=500",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.content == b"jpeg-frame"
    assert calls == [(str(settings.upload_root / asset.storage_path), 500)]
    invalid = client.get(
        f"/api/v1/media/{asset.id}/frame?timestamp_ms=1000",
        headers=headers,
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "VIDEO_FRAME_TIMESTAMP_INVALID"
    assert client.get(f"/api/v1/media/{asset.id}/frame?timestamp_ms=100").status_code == 401


def test_metadata_authorization_not_found_and_path_traversal(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    reviewer = add_user(db, UserRole.REVIEWER)
    supervisor = add_user(db, UserRole.SUPERVISOR)
    asset = stored_media(db, settings, supervisor)
    assert client.get(f"/api/v1/media/{asset.id}").status_code == 401
    assert client.get(
        f"/api/v1/media/{asset.id}", headers=auth(settings, reviewer)
    ).status_code == 403
    headers = auth(settings, supervisor)
    assert client.get(f"/api/v1/media/{asset.id}", headers=headers).status_code == 200
    assert client.get(f"/api/v1/media/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.get(
        f"/api/v1/media/{uuid.uuid4()}/content", headers=headers
    ).status_code == 404
    asset.storage_path = "../../etc/passwd"
    db.commit()
    traversal = client.get(f"/api/v1/media/{asset.id}/content", headers=headers)
    assert traversal.json()["error"]["code"] == "MEDIA_PATH_INVALID"


def test_attach_video_in_draft_ready_and_reject_active_state(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db)
    headers = auth(settings, admin)
    room = Room(code="MEDIA-ROOM", name="Media Room", is_active=True)
    db.add(room)
    db.commit()
    exam_session = ExamSession(
        session_code="MEDIA-SESSION",
        exam_name="Media Exam",
        room_id=room.id,
        status=ExamSessionStatus.DRAFT.value,
        created_by=admin.id,
    )
    db.add(exam_session)
    db.commit()
    first = stored_media(db, settings, admin, b"first")
    second = stored_media(db, settings, admin, b"second")
    attached = client.patch(
        f"/api/v1/sessions/{exam_session.id}",
        headers=headers,
        json={"video_asset_id": str(first.id)},
    )
    assert attached.status_code == 200
    assert attached.json()["video"]["original_filename"] == "exam.mp4"
    exam_session.status = ExamSessionStatus.READY.value
    db.commit()
    replaced = client.patch(
        f"/api/v1/sessions/{exam_session.id}",
        headers=headers,
        json={"video_asset_id": str(second.id)},
    )
    assert replaced.status_code == 200
    detached = client.patch(
        f"/api/v1/sessions/{exam_session.id}", headers=headers, json={"video_asset_id": None}
    )
    assert detached.json()["error"]["code"] == "SESSION_NOT_READY"
    missing = client.patch(
        f"/api/v1/sessions/{exam_session.id}",
        headers=headers,
        json={"video_asset_id": str(uuid.uuid4())},
    )
    assert missing.json()["error"]["code"] == "MEDIA_NOT_FOUND"
    exam_session.status = ExamSessionStatus.RUNNING.value
    db.commit()
    active = client.patch(
        f"/api/v1/sessions/{exam_session.id}",
        headers=headers,
        json={"video_asset_id": str(first.id)},
    )
    assert active.json()["error"]["code"] == "INVALID_SESSION_STATE"
    assert client.patch(
        f"/api/v1/sessions/{uuid.uuid4()}",
        headers=headers,
        json={"video_asset_id": str(first.id)},
    ).status_code == 404


def test_media_role_permissions(
    client: TestClient,
    db: Session,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer = add_user(db, UserRole.REVIEWER)
    supervisor = add_user(db, UserRole.SUPERVISOR)
    rejected = client.post(
        "/api/v1/media/videos",
        headers=auth(settings, reviewer),
        files={"file": ("reviewer.mp4", b"video", "video/mp4")},
    )
    assert rejected.status_code == 403
    monkeypatch.setattr(
        "app.features.media.service.probe_video",
        lambda *_args, **_kwargs: metadata(),
    )
    accepted = client.post(
        "/api/v1/media/videos",
        headers=auth(settings, supervisor),
        files={"file": ("supervisor.mp4", b"video", "video/mp4")},
    )
    assert accepted.status_code == 201
