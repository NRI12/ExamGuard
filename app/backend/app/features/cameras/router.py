import uuid

from fastapi import APIRouter, Query, status

from app.features.auth.dependencies import (
    ApplicationSettings,
    DatabaseSession,
    SessionMonitor,
    SystemManager,
)
from app.features.cameras.schemas import CameraCreate, CameraResponse, CameraUpdate
from app.features.cameras.service import (
    camera_or_error,
    camera_response,
    create_camera,
    list_cameras,
    update_camera,
)
from app.shared.pagination import Page

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=Page[CameraResponse])
def get_cameras(
    _: SessionMonitor,
    db: DatabaseSession,
    settings: ApplicationSettings,
    q: str | None = Query(default=None, max_length=255),
    room_id: uuid.UUID | None = None,
    is_active: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[CameraResponse]:
    cameras, total = list_cameras(
        db,
        settings,
        query=q,
        room_id=room_id,
        is_active=is_active,
        page=page,
        page_size=page_size,
    )
    return Page(items=cameras, page=page, page_size=page_size, total=total)


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def post_camera(
    payload: CameraCreate,
    actor: SystemManager,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> CameraResponse:
    return create_camera(db, payload, actor, settings)


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: uuid.UUID,
    _: SessionMonitor,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> CameraResponse:
    return camera_response(db, camera_or_error(db, camera_id), settings)


@router.patch("/{camera_id}", response_model=CameraResponse)
def patch_camera(
    camera_id: uuid.UUID,
    payload: CameraUpdate,
    actor: SystemManager,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> CameraResponse:
    return update_camera(db, camera_id, payload, actor, settings)
