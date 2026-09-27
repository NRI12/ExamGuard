import uuid

from fastapi import APIRouter, Query, status

from app.features.auth.dependencies import DatabaseSession, RoomManager, RoomReader
from app.features.rooms.schemas import (
    RoomCreate,
    RoomResponse,
    RoomUpdate,
    SeatLayoutUpdate,
    SeatResponse,
)
from app.features.rooms.service import (
    create_room,
    list_active_seats,
    list_rooms,
    replace_seat_layout,
    room_or_error,
    update_room,
)
from app.shared.pagination import Page

router = APIRouter(prefix="/rooms", tags=["rooms"])


@router.get("", response_model=Page[RoomResponse])
def get_rooms(
    _: RoomReader,
    db: DatabaseSession,
    q: str | None = Query(default=None, max_length=255),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[RoomResponse]:
    rooms, total = list_rooms(db, q, page, page_size)
    return Page(
        items=[RoomResponse.model_validate(room) for room in rooms],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
def post_room(payload: RoomCreate, actor: RoomManager, db: DatabaseSession) -> RoomResponse:
    return RoomResponse.model_validate(create_room(db, payload, actor))


@router.get("/{room_id}", response_model=RoomResponse)
def get_room(room_id: uuid.UUID, _: RoomReader, db: DatabaseSession) -> RoomResponse:
    return RoomResponse.model_validate(room_or_error(db, room_id))


@router.patch("/{room_id}", response_model=RoomResponse)
def patch_room(
    room_id: uuid.UUID,
    payload: RoomUpdate,
    actor: RoomManager,
    db: DatabaseSession,
) -> RoomResponse:
    return RoomResponse.model_validate(update_room(db, room_id, payload, actor))


@router.get("/{room_id}/seats", response_model=list[SeatResponse])
def get_seats(
    room_id: uuid.UUID,
    _: RoomReader,
    db: DatabaseSession,
) -> list[SeatResponse]:
    return [SeatResponse.model_validate(seat) for seat in list_active_seats(db, room_id)]


@router.put("/{room_id}/seats", response_model=list[SeatResponse])
def put_seats(
    room_id: uuid.UUID,
    payload: SeatLayoutUpdate,
    actor: RoomManager,
    db: DatabaseSession,
) -> list[SeatResponse]:
    seats = replace_seat_layout(db, room_id, payload, actor)
    return [SeatResponse.model_validate(seat) for seat in seats]
