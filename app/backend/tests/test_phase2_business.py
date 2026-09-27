import uuid

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token, hash_password
from app.db.models.audit import AuditLog
from app.db.models.candidate import Candidate
from app.db.models.media import MediaAsset
from app.db.models.room import Room, Seat
from app.db.models.session import ExamSession, SessionCandidate
from app.db.models.user import User, UserRole


def add_user(db: Session, role: UserRole, username: str) -> User:
    user = User(
        username=username,
        password_hash=hash_password("phase-two-password"),
        full_name=username.title(),
        role=role.value,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return user


def auth(settings: Settings, user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id, settings)}"}


def room_payload(code: str = "P101") -> dict[str, object]:
    return {"code": code, "name": f"Phòng {code}", "description": None, "is_active": True}


def seat_payload(code: str = "A01") -> dict[str, object]:
    return {
        "code": code,
        "x": 0.1,
        "y": 0.1,
        "width": 0.2,
        "height": 0.2,
        "sort_order": 1,
        "is_active": True,
    }


def test_rooms_crud_duplicate_deactivate_and_audit(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-rooms")
    headers = auth(settings, admin)
    created = client.post("/api/v1/rooms", json=room_payload(), headers=headers)
    assert created.status_code == 201
    room_id = created.json()["id"]

    duplicate = client.post("/api/v1/rooms", json=room_payload(), headers=headers)
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "ROOM_CODE_EXISTS"

    updated = client.patch(
        f"/api/v1/rooms/{room_id}",
        json={"name": "Phòng thi 101", "is_active": False},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Phòng thi 101"
    assert updated.json()["is_active"] is False
    room_page = client.get("/api/v1/rooms", headers=headers).json()
    assert len(room_page["items"]) == 1
    assert room_page["total"] == 1
    actions = set(db.scalars(select(AuditLog.action)))
    assert {"ROOM_CREATED", "ROOM_UPDATED"} <= actions


def test_seat_layout_validation_and_transactional_rollback(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-seats")
    headers = auth(settings, admin)
    room_id = client.post("/api/v1/rooms", json=room_payload(), headers=headers).json()["id"]
    valid = client.put(
        f"/api/v1/rooms/{room_id}/seats",
        json={"seats": [seat_payload("A01"), {**seat_payload("A02"), "x": 0.4}]},
        headers=headers,
    )
    assert valid.status_code == 200
    assert [item["code"] for item in valid.json()] == ["A01", "A02"]

    duplicate = client.put(
        f"/api/v1/rooms/{room_id}/seats",
        json={"seats": [seat_payload(), seat_payload()]},
        headers=headers,
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["error"]["code"] == "SEAT_CODE_DUPLICATE"
    outside = client.put(
        f"/api/v1/rooms/{room_id}/seats",
        json={"seats": [{**seat_payload(), "x": 0.9, "width": 0.2}]},
        headers=headers,
    )
    assert outside.status_code == 422
    assert outside.json()["error"]["code"] == "SEAT_OUTSIDE_FRAME"
    persisted = client.get(f"/api/v1/rooms/{room_id}/seats", headers=headers).json()
    assert [item["code"] for item in persisted] == ["A01", "A02"]
    layout_audits = db.scalar(
        select(func.count(AuditLog.id)).where(AuditLog.action == "SEAT_LAYOUT_UPDATED")
    )
    assert layout_audits == 1


def test_candidate_create_search_duplicate_update(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-candidates")
    headers = auth(settings, admin)
    payload = {
        "candidate_code": "sv103",
        "full_name": "Nguyễn Văn An",
        "class_name": "CNTT1",
        "note": None,
    }
    created = client.post("/api/v1/candidates", json=payload, headers=headers)
    assert created.status_code == 201
    assert created.json()["candidate_code"] == "SV103"
    duplicate = client.post("/api/v1/candidates", json=payload, headers=headers)
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["field"] == "candidate_code"
    found = client.get("/api/v1/candidates?q=SV103", headers=headers)
    assert [item["id"] for item in found.json()["items"]] == [created.json()["id"]]
    updated = client.patch(
        f"/api/v1/candidates/{created.json()['id']}",
        json={"class_name": "CNTT2"},
        headers=headers,
    )
    assert updated.json()["class_name"] == "CNTT2"
    assert set(db.scalars(select(AuditLog.action))) >= {"CANDIDATE_CREATED", "CANDIDATE_UPDATED"}


def test_session_create_duplicate_invalid_and_inactive_room(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-sessions")
    headers = auth(settings, admin)
    active_id = client.post(
        "/api/v1/rooms", json=room_payload("P201"), headers=headers
    ).json()["id"]
    inactive = client.post(
        "/api/v1/rooms",
        json={**room_payload("P202"), "is_active": False},
        headers=headers,
    ).json()
    payload = {"session_code": "ca-01", "exam_name": "Cơ sở dữ liệu", "room_id": active_id}
    created = client.post("/api/v1/sessions", json=payload, headers=headers)
    assert created.status_code == 201
    assert created.json()["status"] == "DRAFT"
    assert client.post("/api/v1/sessions", json=payload, headers=headers).status_code == 409
    missing = client.post(
        "/api/v1/sessions",
        json={
            **payload,
            "session_code": "CA-02",
            "room_id": "00000000-0000-0000-0000-000000000099",
        },
        headers=headers,
    )
    assert missing.json()["error"]["code"] == "ROOM_NOT_FOUND"
    inactive_response = client.post(
        "/api/v1/sessions",
        json={**payload, "session_code": "CA-03", "room_id": inactive["id"]},
        headers=headers,
    )
    assert inactive_response.json()["error"]["code"] == "ROOM_INACTIVE"
    updated = client.patch(
        f"/api/v1/sessions/{created.json()['id']}",
        json={"exam_name": "Cơ sở dữ liệu nâng cao"},
        headers=headers,
    )
    assert updated.json()["exam_name"].endswith("nâng cao")


def _prepare_assignment_data(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> tuple[dict[str, str], str, list[str], list[str], str]:
    admin = add_user(db, UserRole.ADMIN, "admin-assignment")
    headers = auth(settings, admin)
    room_id = client.post("/api/v1/rooms", json=room_payload("P301"), headers=headers).json()["id"]
    seats = client.put(
        f"/api/v1/rooms/{room_id}/seats",
        json={"seats": [seat_payload("A01"), {**seat_payload("A02"), "x": 0.4}]},
        headers=headers,
    ).json()
    candidates = [
        client.post(
            "/api/v1/candidates",
            json={"candidate_code": f"SV30{index}", "full_name": f"Thí sinh {index}"},
            headers=headers,
        ).json()
        for index in (1, 2)
    ]
    session_id = client.post(
        "/api/v1/sessions",
        json={"session_code": "CA-ASSIGN", "exam_name": "Lập trình", "room_id": room_id},
        headers=headers,
    ).json()["id"]
    other_room_id = client.post(
        "/api/v1/rooms", json=room_payload("P302"), headers=headers
    ).json()["id"]
    other_seat_id = client.put(
        f"/api/v1/rooms/{other_room_id}/seats",
        json={"seats": [seat_payload("B01")]},
        headers=headers,
    ).json()[0]["id"]
    candidate_ids = [item["id"] for item in candidates]
    seat_ids = [item["id"] for item in seats]
    return headers, session_id, candidate_ids, seat_ids, other_seat_id


def test_session_assignments_constraints_readiness_and_transaction(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    headers, session_id, candidates, seats, other_seat = _prepare_assignment_data(
        client, db, settings
    )
    valid_payload = {"assignments": [{"candidate_id": candidates[0], "seat_id": seats[0]}]}
    valid = client.put(
        f"/api/v1/sessions/{session_id}/candidates",
        json=valid_payload,
        headers=headers,
    )
    assert valid.status_code == 200
    assert valid.json()["readiness"]["can_mark_ready"] is False
    assert valid.json()["readiness"]["video_configured"] is False

    cases = [
        (
            [
                {"candidate_id": candidates[0], "seat_id": seats[0]},
                {"candidate_id": candidates[0], "seat_id": seats[1]},
            ],
            "CANDIDATE_ALREADY_ASSIGNED",
        ),
        (
            [
                {"candidate_id": candidates[0], "seat_id": seats[0]},
                {"candidate_id": candidates[1], "seat_id": seats[0]},
            ],
            "SEAT_ALREADY_ASSIGNED",
        ),
        ([{"candidate_id": candidates[1], "seat_id": other_seat}], "SEAT_NOT_IN_SESSION_ROOM"),
    ]
    for assignments, expected_code in cases:
        response = client.put(
            f"/api/v1/sessions/{session_id}/candidates",
            json={"assignments": assignments},
            headers=headers,
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == expected_code
        persisted = list(
            db.scalars(
                select(SessionCandidate).where(
                    SessionCandidate.session_id == uuid.UUID(session_id)
                )
            )
        )
        assert len(persisted) == 1
        assert str(persisted[0].candidate_id) == candidates[0]

    inactive_seat = db.get(Seat, uuid.UUID(seats[1]))
    assert inactive_seat is not None
    inactive_seat.is_active = False
    db.commit()
    response = client.put(
        f"/api/v1/sessions/{session_id}/candidates",
        json={"assignments": [{"candidate_id": candidates[1], "seat_id": seats[1]}]},
        headers=headers,
    )
    assert response.json()["error"]["code"] == "SEAT_INACTIVE"
    exam_session = db.get(ExamSession, uuid.UUID(session_id))
    assert exam_session is not None
    media_id = uuid.uuid4()
    media_directory = settings.upload_root / str(media_id)
    media_directory.mkdir(parents=True, exist_ok=True)
    (media_directory / "source.mp4").write_bytes(b"video")
    media = MediaAsset(
        id=media_id,
        original_filename="phase2-ready.mp4",
        stored_filename="source.mp4",
        storage_path=f"{media_id}/source.mp4",
        mime_type="video/mp4",
        codec="h264",
        width=32,
        height=24,
        fps=25.0,
        duration_ms=1000,
        size_bytes=5,
        sha256="a" * 64,
        created_by=exam_session.created_by,
    )
    db.add(media)
    exam_session.video_asset_id = media.id
    db.commit()
    ready = client.patch(
        f"/api/v1/sessions/{session_id}", json={"status": "READY"}, headers=headers
    )
    assert ready.status_code == 200
    assert ready.json()["status"] == "READY"
    assert ready.json()["readiness"]["can_mark_ready"] is True
    assert db.scalar(
        select(func.count(AuditLog.id)).where(AuditLog.action == "SESSION_CANDIDATES_UPDATED")
    ) == 1


def test_role_authorization(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-roles")
    supervisor = add_user(db, UserRole.SUPERVISOR, "supervisor-roles")
    reviewer = add_user(db, UserRole.REVIEWER, "reviewer-roles")
    admin_headers = auth(settings, admin)
    supervisor_headers = auth(settings, supervisor)
    reviewer_headers = auth(settings, reviewer)
    room = client.post(
        "/api/v1/rooms", json=room_payload("P401"), headers=admin_headers
    ).json()

    assert client.get("/api/v1/rooms", headers=reviewer_headers).status_code == 200
    forbidden_room = client.post(
        "/api/v1/rooms", json=room_payload("P402"), headers=supervisor_headers
    )
    assert forbidden_room.status_code == 403
    assert forbidden_room.json()["error"]["code"] == "FORBIDDEN"
    session = client.post(
        "/api/v1/sessions",
        json={"session_code": "CA-SUP", "exam_name": "Mạng", "room_id": room["id"]},
        headers=supervisor_headers,
    )
    assert session.status_code == 201
    supervisor_update = client.patch(
        f"/api/v1/sessions/{session.json()['id']}",
        json={"exam_name": "Mạng máy tính"},
        headers=supervisor_headers,
    )
    assert supervisor_update.status_code == 200
    assert supervisor_update.json()["exam_name"] == "Mạng máy tính"
    assert client.put(
        f"/api/v1/sessions/{session.json()['id']}/candidates",
        json={"assignments": []},
        headers=reviewer_headers,
    ).status_code == 403


def test_inactive_room_rejects_new_assignments(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    headers, session_id, candidates, seats, _ = _prepare_assignment_data(client, db, settings)
    exam_session = db.get(ExamSession, uuid.UUID(session_id))
    assert exam_session is not None
    room = db.get(Room, exam_session.room_id)
    assert room is not None
    room.is_active = False
    db.commit()
    response = client.put(
        f"/api/v1/sessions/{session_id}/candidates",
        json={"assignments": [{"candidate_id": candidates[0], "seat_id": seats[0]}]},
        headers=headers,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ROOM_INACTIVE"
    assert db.scalar(select(func.count(SessionCandidate.id))) == 0


def test_list_pagination_search_and_session_cancellation(
    client: TestClient,
    db: Session,
    settings: Settings,
) -> None:
    admin = add_user(db, UserRole.ADMIN, "admin-pagination")
    headers = auth(settings, admin)
    db.add_all(
        [
            Candidate(candidate_code=f"PAGE-{index:02d}", full_name=f"Thí sinh {index:02d}")
            for index in range(25)
        ]
    )
    db.commit()

    first = client.get("/api/v1/candidates?page=1&page_size=10&q=PAGE", headers=headers)
    second = client.get("/api/v1/candidates?page=2&page_size=10&q=PAGE", headers=headers)
    assert first.status_code == 200
    assert first.json()["total"] == 25
    assert len(first.json()["items"]) == 10
    assert first.json()["items"][0]["id"] != second.json()["items"][0]["id"]
    assert client.get("/api/v1/candidates?page=0", headers=headers).status_code == 422

    room = client.post(
        "/api/v1/rooms", json=room_payload("P-PAGE"), headers=headers
    ).json()
    exam_session = client.post(
        "/api/v1/sessions",
        json={
            "session_code": "CANCEL-ME",
            "exam_name": "Phiên sẽ hủy",
            "room_id": room["id"],
        },
        headers=headers,
    ).json()
    filtered = client.get(
        f"/api/v1/sessions?room_id={room['id']}", headers=headers
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["id"] == exam_session["id"]
    cancelled = client.patch(
        f"/api/v1/sessions/{exam_session['id']}",
        json={"status": "CANCELLED"},
        headers=headers,
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    locked = client.patch(
        f"/api/v1/sessions/{exam_session['id']}",
        json={"exam_name": "Không được đổi"},
        headers=headers,
    )
    assert locked.status_code == 409
    assert "SESSION_CANCELLED" in set(db.scalars(select(AuditLog.action)))
