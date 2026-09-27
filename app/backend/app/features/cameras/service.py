import uuid

from fastapi import status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models.camera import Camera
from app.db.models.media import MediaAsset
from app.db.models.room import Room
from app.db.models.session import ExamSession, ExamSessionStatus
from app.db.models.user import User
from app.features.cameras.schemas import (
    CameraCreate,
    CameraMediaSummary,
    CameraResponse,
    CameraRoomSummary,
    CameraUpdate,
)
from app.features.media.service import media_file_path, media_response
from app.shared.audit import AuditAction, AuditService


def camera_or_error(db: Session, camera_id: uuid.UUID) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "CAMERA_NOT_FOUND", "Camera was not found")
    return camera


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


def _validated_media(
    db: Session,
    media_id: uuid.UUID,
    settings: Settings,
) -> MediaAsset:
    media = db.get(MediaAsset, media_id)
    if media is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "MEDIA_NOT_FOUND",
            "Media was not found",
            field_name="source_media_asset_id",
        )
    media_response(media)
    media_file_path(media, settings)
    return media


def camera_response(db: Session, camera: Camera, settings: Settings) -> CameraResponse:
    room = db.get(Room, camera.room_id)
    if room is None:
        raise ApiError(status.HTTP_409_CONFLICT, "ROOM_NOT_FOUND", "Camera room was not found")
    in_use = bool(
        db.scalar(
            select(ExamSession.id).where(
                ExamSession.camera_id == camera.id,
                ExamSession.status.in_(
                    [ExamSessionStatus.RUNNING.value, ExamSessionStatus.PAUSED.value]
                ),
            )
        )
    )
    source: CameraMediaSummary | None = None
    source_ok = False
    media = db.get(MediaAsset, camera.source_media_asset_id)
    if media is not None:
        try:
            value = media_response(media)
            media_file_path(media, settings)
            source = CameraMediaSummary(
                id=value.id,
                original_filename=value.original_filename,
                media_url=value.media_url,
                mime_type=value.mime_type,
                codec=value.codec,
                width=value.width,
                height=value.height,
                fps=value.fps,
                duration_ms=value.duration_ms,
                size_bytes=value.size_bytes,
                sha256=value.sha256,
                created_at=value.created_at,
            )
            source_ok = True
        except ApiError:
            source_ok = False
    if not camera.is_active:
        camera_status = "DISABLED"
    elif in_use:
        camera_status = "IN_USE"
    elif source_ok:
        camera_status = "READY"
    else:
        camera_status = "ERROR"
    return CameraResponse(
        id=camera.id,
        name=camera.name,
        room_id=camera.room_id,
        room=CameraRoomSummary(id=room.id, code=room.code, name=room.name),
        source_media_asset_id=camera.source_media_asset_id,
        source_media=source,
        description=camera.description,
        is_active=camera.is_active,
        status=camera_status,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
    )


def list_cameras(
    db: Session,
    settings: Settings,
    *,
    query: str | None = None,
    room_id: uuid.UUID | None = None,
    is_active: bool | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[CameraResponse], int]:
    statement = select(Camera).join(Room, Room.id == Camera.room_id)
    if query and (term := query.strip()):
        pattern = f"%{term}%"
        statement = statement.where(
            or_(Camera.name.ilike(pattern), Room.code.ilike(pattern), Room.name.ilike(pattern))
        )
    if room_id is not None:
        statement = statement.where(Camera.room_id == room_id)
    if is_active is not None:
        statement = statement.where(Camera.is_active.is_(is_active))
    total = db.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0
    cameras = list(
        db.scalars(statement.order_by(Camera.name).offset((page - 1) * page_size).limit(page_size))
    )
    return [camera_response(db, item, settings) for item in cameras], total


def create_camera(
    db: Session,
    payload: CameraCreate,
    actor: User,
    settings: Settings,
) -> CameraResponse:
    _active_room_or_error(db, payload.room_id)
    _validated_media(db, payload.source_media_asset_id, settings)
    camera = Camera(**payload.model_dump())
    db.add(camera)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.CAMERA_CREATED,
            entity_type="CAMERA",
            entity_id=camera.id,
            metadata={"room_id": str(camera.room_id), "name": camera.name},
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "CAMERA_SAVE_FAILED",
            "Camera could not be saved",
        ) from error
    db.refresh(camera)
    return camera_response(db, camera, settings)


def update_camera(
    db: Session,
    camera_id: uuid.UUID,
    payload: CameraUpdate,
    actor: User,
    settings: Settings,
) -> CameraResponse:
    camera = camera_or_error(db, camera_id)
    changes = payload.model_dump(exclude_unset=True)
    if "room_id" in changes:
        _active_room_or_error(db, changes["room_id"])
    if "source_media_asset_id" in changes:
        _validated_media(db, changes["source_media_asset_id"], settings)
    in_use = bool(
        db.scalar(
            select(ExamSession.id).where(
                ExamSession.camera_id == camera.id,
                ExamSession.status.in_(
                    [ExamSessionStatus.RUNNING.value, ExamSessionStatus.PAUSED.value]
                ),
            )
        )
    )
    if in_use and any(
        field in changes for field in {"room_id", "source_media_asset_id", "is_active"}
    ):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "CAMERA_IN_USE",
            "Camera source cannot be changed while it is in use",
        )
    for field, value in changes.items():
        setattr(camera, field, value)
    AuditService.record(
        db,
        actor=actor,
        action=AuditAction.CAMERA_UPDATED,
        entity_type="CAMERA",
        entity_id=camera.id,
        metadata={"name": camera.name, "changed_fields": sorted(changes)},
    )
    db.commit()
    db.refresh(camera)
    return camera_response(db, camera, settings)
