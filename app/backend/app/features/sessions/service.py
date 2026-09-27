import uuid
from datetime import UTC, date, datetime, time, timedelta

from fastapi import status
from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models.camera import Camera
from app.db.models.candidate import Candidate
from app.db.models.media import MediaAsset
from app.db.models.room import Room, Seat
from app.db.models.session import (
    ExamSession,
    ExamSessionStatus,
    SessionCandidate,
    SessionSourceType,
)
from app.db.models.user import User
from app.features.media.service import media_file_path, media_response
from app.features.sessions.schemas import (
    CameraSummary,
    CandidateSummary,
    MediaSummary,
    RoomSummary,
    SeatSummary,
    SessionAssignmentResponse,
    SessionCandidatesUpdate,
    SessionCreate,
    SessionReadiness,
    SessionResponse,
    SessionUpdate,
    UserSummary,
)
from app.shared.audit import AuditAction, AuditService


def session_or_error(db: Session, session_id: uuid.UUID) -> ExamSession:
    exam_session = db.get(ExamSession, session_id)
    if exam_session is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "SESSION_NOT_FOUND",
            "Exam session was not found",
        )
    return exam_session


def _active_room_or_error(db: Session, room_id: uuid.UUID) -> Room:
    room = db.get(Room, room_id)
    if room is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ROOM_NOT_FOUND",
            "Room was not found",
            field_name="room_id",
        )
    if not room.is_active:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ROOM_INACTIVE",
            "Inactive room cannot be selected",
            field_name="room_id",
        )
    return room


def _validate_schedule(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and end <= start:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_SCHEDULE",
            "Scheduled end must be after scheduled start",
            field_name="scheduled_end",
        )


def _readiness(
    db: Session,
    exam_session: ExamSession,
    room: Room,
    video_asset_id: uuid.UUID | None,
) -> SessionReadiness:
    seat_count = (
        db.scalar(
            select(func.count(Seat.id)).where(
                Seat.room_id == room.id,
                Seat.is_active.is_(True),
            )
        )
        or 0
    )
    candidate_count = (
        db.scalar(
            select(func.count(SessionCandidate.id)).where(
                SessionCandidate.session_id == exam_session.id
            )
        )
        or 0
    )
    return SessionReadiness(
        room_selected=True,
        room_active=room.is_active,
        seat_layout_available=seat_count > 0,
        active_seats=seat_count,
        candidates_assigned=candidate_count,
        video_configured=video_asset_id is not None,
        monitoring_status=(
            "NOT_STARTED"
            if exam_session.status in {ExamSessionStatus.DRAFT.value, ExamSessionStatus.READY.value}
            else exam_session.status
        ),
        can_mark_ready=room.is_active and video_asset_id is not None,
    )


def session_response(db: Session, exam_session: ExamSession) -> SessionResponse:
    room = db.get(Room, exam_session.room_id)
    if room is None:
        raise ApiError(status.HTTP_409_CONFLICT, "ROOM_NOT_FOUND", "Session room was not found")
    rows = db.execute(
        select(SessionCandidate, Candidate, Seat)
        .join(Candidate, Candidate.id == SessionCandidate.candidate_id)
        .join(Seat, Seat.id == SessionCandidate.seat_id)
        .where(SessionCandidate.session_id == exam_session.id)
        .order_by(Seat.sort_order.nulls_last(), Seat.code)
    ).all()
    assignments = [
        SessionAssignmentResponse(
            id=assignment.id,
            candidate=CandidateSummary(
                id=candidate.id,
                candidate_code=candidate.candidate_code,
                full_name=candidate.full_name,
                class_name=candidate.class_name,
            ),
            seat=SeatSummary(id=seat.id, code=seat.code, is_active=seat.is_active),
        )
        for assignment, candidate, seat in rows
    ]
    readiness = _readiness(db, exam_session, room, exam_session.video_asset_id)
    camera = db.get(Camera, exam_session.camera_id) if exam_session.camera_id else None
    creator = db.get(User, exam_session.created_by)
    if creator is None:
        raise ApiError(status.HTTP_409_CONFLICT, "USER_NOT_FOUND", "Session creator was not found")
    video: MediaSummary | None = None
    if exam_session.video_asset_id is not None:
        asset = db.get(MediaAsset, exam_session.video_asset_id)
        if asset is None:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "MEDIA_NOT_FOUND",
                "Session video was not found",
            )
        response = media_response(asset)
        video = MediaSummary(
            id=response.id,
            original_filename=response.original_filename,
            media_url=response.media_url,
            codec=response.codec,
            width=response.width,
            height=response.height,
            fps=response.fps,
            duration_ms=response.duration_ms,
            size_bytes=response.size_bytes,
        )
    return SessionResponse(
        id=exam_session.id,
        session_code=exam_session.session_code,
        exam_name=exam_session.exam_name,
        room_id=exam_session.room_id,
        room=RoomSummary(id=room.id, code=room.code, name=room.name, is_active=room.is_active),
        source_type=SessionSourceType(exam_session.source_type),
        camera_id=exam_session.camera_id,
        camera=(
            CameraSummary(id=camera.id, name=camera.name, is_active=camera.is_active)
            if camera is not None
            else None
        ),
        video_asset_id=exam_session.video_asset_id,
        video=video,
        status=ExamSessionStatus(exam_session.status),
        scheduled_start=exam_session.scheduled_start,
        scheduled_end=exam_session.scheduled_end,
        actual_start=exam_session.actual_start,
        actual_end=exam_session.actual_end,
        runtime_profile=exam_session.runtime_profile,
        created_by=exam_session.created_by,
        created_by_user=UserSummary(
            id=creator.id, username=creator.username, full_name=creator.full_name
        ),
        candidate_count=readiness.candidates_assigned,
        assignments=assignments,
        readiness=readiness,
        created_at=exam_session.created_at,
        updated_at=exam_session.updated_at,
    )


def list_sessions(
    db: Session,
    query: str | None = None,
    session_status: ExamSessionStatus | None = None,
    page: int = 1,
    page_size: int = 20,
    room_id: uuid.UUID | None = None,
    scheduled_date: date | None = None,
) -> tuple[list[SessionResponse], int]:
    statement = select(ExamSession).join(Room, Room.id == ExamSession.room_id)
    if query and (term := query.strip()):
        pattern = f"%{term}%"
        statement = statement.where(
            or_(
                ExamSession.session_code.ilike(pattern),
                ExamSession.exam_name.ilike(pattern),
                Room.code.ilike(pattern),
                Room.name.ilike(pattern),
            )
        )
    if session_status is not None:
        statement = statement.where(ExamSession.status == session_status.value)
    if room_id is not None:
        statement = statement.where(ExamSession.room_id == room_id)
    if scheduled_date is not None:
        day_start = datetime.combine(scheduled_date, time.min, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        statement = statement.where(
            ExamSession.scheduled_start >= day_start,
            ExamSession.scheduled_start < day_end,
        )
    total = db.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0
    sessions = list(
        db.scalars(
            statement.order_by(ExamSession.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return [session_response(db, item) for item in sessions], total


def create_session(db: Session, payload: SessionCreate, actor: User) -> SessionResponse:
    if payload.status != ExamSessionStatus.DRAFT:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_SESSION_STATE",
            "A session must be created in DRAFT state",
            field_name="status",
        )
    _active_room_or_error(db, payload.room_id)
    if db.scalar(select(ExamSession.id).where(ExamSession.session_code == payload.session_code)):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_CODE_EXISTS",
            "Session code already exists",
            field_name="session_code",
        )
    values = payload.model_dump(exclude={"duration_minutes"})
    values["status"] = payload.status.value
    exam_session = ExamSession(**values, created_by=actor.id)
    db.add(exam_session)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.SESSION_CREATED,
            entity_type="EXAM_SESSION",
            entity_id=exam_session.id,
            metadata={
                "session_code": exam_session.session_code,
                "room_id": str(exam_session.room_id),
            },
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_CODE_EXISTS",
            "Session code already exists",
            field_name="session_code",
        ) from error
    db.refresh(exam_session)
    return session_response(db, exam_session)


def update_session(
    db: Session,
    session_id: uuid.UUID,
    payload: SessionUpdate,
    actor: User,
    settings: Settings,
) -> SessionResponse:
    exam_session = session_or_error(db, session_id)
    changes = payload.model_dump(exclude_unset=True)
    current_status = ExamSessionStatus(exam_session.status)
    desired_status = changes.get("status")
    cancelling = desired_status in {
        ExamSessionStatus.CANCELLED,
        ExamSessionStatus.CANCELLED.value,
    }
    if current_status not in {ExamSessionStatus.DRAFT, ExamSessionStatus.READY}:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "INVALID_SESSION_STATE",
            "A session can only be edited or cancelled before monitoring starts",
        )
    new_code = changes.get("session_code")
    if isinstance(new_code, str) and db.scalar(
        select(ExamSession.id).where(
            ExamSession.session_code == new_code,
            ExamSession.id != exam_session.id,
        )
    ):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_CODE_EXISTS",
            "Session code already exists",
            field_name="session_code",
        )

    room_id = changes.get("room_id", exam_session.room_id)
    if not isinstance(room_id, uuid.UUID):
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ROOM_NOT_FOUND",
            "Room was not found",
            field_name="room_id",
        )
    if cancelling and set(changes) == {"status"}:
        room = db.get(Room, room_id)
        if room is None:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "ROOM_NOT_FOUND",
                "Session room was not found",
                field_name="room_id",
            )
    else:
        room = _active_room_or_error(db, room_id)
    if room_id != exam_session.room_id:
        assignment_count = (
            db.scalar(
                select(func.count(SessionCandidate.id)).where(
                    SessionCandidate.session_id == exam_session.id
                )
            )
            or 0
        )
        if assignment_count:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "SESSION_HAS_ASSIGNMENTS",
                "Clear candidate assignments before changing the room",
                field_name="room_id",
            )

    if not (cancelling and set(changes) == {"status"}):
        requested_source = SessionSourceType(
            changes.get("source_type", exam_session.source_type)
        )
        if requested_source == SessionSourceType.CAMERA:
            camera_id = changes.get("camera_id", exam_session.camera_id)
            if not isinstance(camera_id, uuid.UUID):
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "CAMERA_REQUIRED",
                    "A camera is required for camera monitoring",
                    field_name="camera_id",
                )
            camera = db.get(Camera, camera_id)
            if camera is None:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "CAMERA_NOT_FOUND",
                    "Camera was not found",
                    field_name="camera_id",
                )
            if not camera.is_active:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "CAMERA_DISABLED",
                    "Disabled camera cannot be selected",
                    field_name="camera_id",
                )
            if camera.room_id != room_id:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "CAMERA_NOT_IN_ROOM",
                    "Camera does not belong to the selected room",
                    field_name="camera_id",
                )
            media = db.get(MediaAsset, camera.source_media_asset_id)
            if media is None:
                raise ApiError(
                    status.HTTP_409_CONFLICT,
                    "CAMERA_SOURCE_MISSING",
                    "Camera source video was not found",
                )
            media_response(media)
            media_file_path(media, settings)
            changes["source_type"] = SessionSourceType.CAMERA.value
            changes["camera_id"] = camera.id
            changes["video_asset_id"] = camera.source_media_asset_id
        else:
            if changes.get("camera_id") is not None:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "INVALID_SESSION_SOURCE",
                    "Uploaded video sessions cannot reference a camera",
                    field_name="camera_id",
                )
            video_asset_id = changes.get("video_asset_id", exam_session.video_asset_id)
            if video_asset_id is not None:
                media = db.get(MediaAsset, video_asset_id)
                if media is None:
                    raise ApiError(
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "MEDIA_NOT_FOUND",
                        "Media was not found",
                        field_name="video_asset_id",
                    )
                media_response(media)
                media_file_path(media, settings)
            if video_asset_id is None and current_status == ExamSessionStatus.READY:
                raise ApiError(
                    status.HTTP_409_CONFLICT,
                    "SESSION_NOT_READY",
                    "A READY session must keep a valid source video",
                )
            changes["source_type"] = SessionSourceType.VIDEO_UPLOAD.value
            changes["camera_id"] = None

    scheduled_start = changes.get("scheduled_start", exam_session.scheduled_start)
    scheduled_end = changes.get("scheduled_end", exam_session.scheduled_end)
    _validate_schedule(scheduled_start, scheduled_end)

    if desired_status is not None:
        desired_status = ExamSessionStatus(desired_status)
        if desired_status != current_status:
            valid_transition = desired_status == ExamSessionStatus.CANCELLED or (
                current_status == ExamSessionStatus.DRAFT
                and desired_status == ExamSessionStatus.READY
            )
            if not valid_transition:
                raise ApiError(
                    status.HTTP_409_CONFLICT,
                    "INVALID_SESSION_STATE",
                    "The requested session status transition is not allowed",
                    field_name="status",
                )
            if desired_status == ExamSessionStatus.READY:
                readiness = _readiness(
                    db,
                    exam_session,
                    room,
                    changes.get("video_asset_id", exam_session.video_asset_id),
                )
                if not readiness.can_mark_ready:
                    raise ApiError(
                        status.HTTP_409_CONFLICT,
                        "SESSION_NOT_READY",
                        "An active room and source video are required",
                        readiness.model_dump(),
                        field_name="status",
                    )
        changes["status"] = desired_status.value

    for field, value in changes.items():
        setattr(exam_session, field, value)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=(
                AuditAction.SESSION_CANCELLED
                if changes.get("status") == ExamSessionStatus.CANCELLED.value
                else AuditAction.SESSION_UPDATED
            ),
            entity_type="EXAM_SESSION",
            entity_id=exam_session.id,
            metadata={
                "session_code": exam_session.session_code,
                "changed_fields": sorted(changes),
            },
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SESSION_CODE_EXISTS",
            "Session code already exists",
            field_name="session_code",
        ) from error
    db.refresh(exam_session)
    return session_response(db, exam_session)


def replace_assignments(
    db: Session,
    session_id: uuid.UUID,
    payload: SessionCandidatesUpdate,
    actor: User,
) -> SessionResponse:
    exam_session = session_or_error(db, session_id)
    if exam_session.status not in {ExamSessionStatus.DRAFT.value, ExamSessionStatus.READY.value}:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "INVALID_SESSION_STATE",
            "Assignments can only be changed before monitoring starts",
        )
    room = _active_room_or_error(db, exam_session.room_id)
    candidate_ids: set[uuid.UUID] = set()
    seat_ids: set[uuid.UUID] = set()
    validated: list[tuple[Candidate, Seat]] = []
    for item in payload.assignments:
        if item.candidate_id in candidate_ids:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "CANDIDATE_ALREADY_ASSIGNED",
                "Candidate can only be assigned once in a session",
                {"candidate_id": str(item.candidate_id)},
            )
        if item.seat_id in seat_ids:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_ALREADY_ASSIGNED",
                "Seat can only be assigned once in a session",
                {"seat_id": str(item.seat_id)},
            )
        candidate = db.get(Candidate, item.candidate_id)
        if candidate is None:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "CANDIDATE_NOT_FOUND",
                "Candidate was not found",
                {"candidate_id": str(item.candidate_id)},
            )
        seat = db.get(Seat, item.seat_id)
        if seat is None:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_NOT_FOUND",
                "Seat was not found",
                {"seat_id": str(item.seat_id)},
            )
        if seat.room_id != room.id:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_NOT_IN_SESSION_ROOM",
                "Seat does not belong to the session room",
                {"seat_id": str(seat.id)},
            )
        if not seat.is_active:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_INACTIVE",
                "Inactive seat cannot be assigned",
                {"seat_id": str(seat.id)},
            )
        candidate_ids.add(candidate.id)
        seat_ids.add(seat.id)
        validated.append((candidate, seat))

    try:
        db.execute(delete(SessionCandidate).where(SessionCandidate.session_id == exam_session.id))
        db.add_all(
            [
                SessionCandidate(
                    session_id=exam_session.id,
                    candidate_id=candidate.id,
                    seat_id=seat.id,
                )
                for candidate, seat in validated
            ]
        )
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.SESSION_CANDIDATES_UPDATED,
            entity_type="EXAM_SESSION",
            entity_id=exam_session.id,
            metadata={
                "session_code": exam_session.session_code,
                "assignment_count": len(validated),
                "candidate_codes": [candidate.candidate_code for candidate, _ in validated],
                "seat_codes": [seat.code for _, seat in validated],
            },
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "ASSIGNMENT_CONFLICT",
            "Candidate assignments could not be saved",
        ) from error
    return session_response(db, exam_session)
