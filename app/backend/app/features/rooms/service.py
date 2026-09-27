import math
import uuid

from fastapi import status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ApiError
from app.db.models.room import Room, Seat
from app.db.models.user import User
from app.features.rooms.schemas import RoomCreate, RoomUpdate, SeatLayoutUpdate, SeatWrite
from app.shared.audit import AuditAction, AuditService


def room_or_error(db: Session, room_id: uuid.UUID) -> Room:
    room = db.get(Room, room_id)
    if room is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "ROOM_NOT_FOUND", "Room was not found")
    return room


def list_rooms(
    db: Session,
    query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Room], int]:
    statement = select(Room)
    if query and (term := query.strip()):
        pattern = f"%{term}%"
        statement = statement.where(
            or_(Room.code.ilike(pattern), Room.name.ilike(pattern))
        )
    total = db.scalar(
        select(func.count()).select_from(statement.order_by(None).subquery())
    ) or 0
    items = list(
        db.scalars(
            statement.order_by(Room.code).offset((page - 1) * page_size).limit(page_size)
        )
    )
    return items, total


def create_room(db: Session, payload: RoomCreate, actor: User) -> Room:
    if db.scalar(select(Room.id).where(Room.code == payload.code)) is not None:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "ROOM_CODE_EXISTS",
            "Room code already exists",
            field_name="code",
        )

    room = Room(**payload.model_dump())
    db.add(room)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.ROOM_CREATED,
            entity_type="ROOM",
            entity_id=room.id,
            metadata={"room_code": room.code},
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "ROOM_CODE_EXISTS",
            "Room code already exists",
            field_name="code",
        ) from error
    db.refresh(room)
    return room


def update_room(db: Session, room_id: uuid.UUID, payload: RoomUpdate, actor: User) -> Room:
    room = room_or_error(db, room_id)
    changes = payload.model_dump(exclude_unset=True)
    new_code = changes.get("code")
    if isinstance(new_code, str):
        duplicate = db.scalar(
            select(Room.id).where(Room.code == new_code, Room.id != room.id)
        )
        if duplicate is not None:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "ROOM_CODE_EXISTS",
                "Room code already exists",
                field_name="code",
            )

    for field, value in changes.items():
        setattr(room, field, value)
    try:
        db.flush()
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.ROOM_UPDATED,
            entity_type="ROOM",
            entity_id=room.id,
            metadata={"room_code": room.code, "changed_fields": sorted(changes)},
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "ROOM_CODE_EXISTS",
            "Room code already exists",
            field_name="code",
        ) from error
    db.refresh(room)
    return room


def list_active_seats(db: Session, room_id: uuid.UUID) -> list[Seat]:
    room_or_error(db, room_id)
    statement = (
        select(Seat)
        .where(Seat.room_id == room_id, Seat.is_active.is_(True))
        .order_by(Seat.sort_order.nulls_last(), Seat.code)
    )
    return list(db.scalars(statement))


def _validate_layout(seats: list[SeatWrite]) -> None:
    codes: set[str] = set()
    ids: set[uuid.UUID] = set()
    for seat in seats:
        if seat.code in codes:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_CODE_DUPLICATE",
                "Seat codes must be unique inside a room",
                {"seat_code": seat.code},
                field_name="seats",
            )
        codes.add(seat.code)
        if seat.id is not None:
            if seat.id in ids:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "SEAT_ID_DUPLICATE",
                    "A seat cannot appear more than once",
                )
            ids.add(seat.id)

        geometry = (seat.x, seat.y, seat.width, seat.height)
        valid = (
            all(math.isfinite(value) for value in geometry)
            and 0 <= seat.x <= 1
            and 0 <= seat.y <= 1
            and 0 < seat.width <= 1
            and 0 < seat.height <= 1
            and seat.x + seat.width <= 1
            and seat.y + seat.height <= 1
        )
        if not valid:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "SEAT_OUTSIDE_FRAME",
                "Seat geometry must remain inside the normalized frame",
                {"seat_code": seat.code},
                field_name="seats",
            )


def replace_seat_layout(
    db: Session,
    room_id: uuid.UUID,
    payload: SeatLayoutUpdate,
    actor: User,
) -> list[Seat]:
    room = room_or_error(db, room_id)
    _validate_layout(payload.seats)
    existing = list(db.scalars(select(Seat).where(Seat.room_id == room.id)))
    by_id = {seat.id: seat for seat in existing}
    by_code = {seat.code: seat for seat in existing}

    targets: list[tuple[SeatWrite, Seat | None]] = []
    target_ids: set[uuid.UUID] = set()
    for item in payload.seats:
        target: Seat | None
        if item.id is not None:
            target = by_id.get(item.id)
            if target is None:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "SEAT_NOT_IN_ROOM",
                    "Seat does not belong to this room",
                    {"seat_id": str(item.id)},
                )
        else:
            target = by_code.get(item.code)

        conflicting = by_code.get(item.code)
        if conflicting is not None and conflicting is not target:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "SEAT_CODE_DUPLICATE",
                "Seat codes must be unique inside a room",
                {"seat_code": item.code},
                field_name="seats",
            )
        if target is not None:
            if target.id in target_ids:
                raise ApiError(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "SEAT_ID_DUPLICATE",
                    "A seat cannot appear more than once",
                )
            target_ids.add(target.id)
        targets.append((item, target))

    for seat in existing:
        if seat.id not in target_ids:
            seat.is_active = False

    for item, target in targets:
        values = item.model_dump(exclude={"id"})
        if target is None:
            target = Seat(room_id=room.id, **values)
            db.add(target)
        else:
            for field, value in values.items():
                setattr(target, field, value)

    try:
        db.flush()
        active_seats = list_active_seats(db, room.id)
        AuditService.record(
            db,
            actor=actor,
            action=AuditAction.SEAT_LAYOUT_UPDATED,
            entity_type="ROOM",
            entity_id=room.id,
            metadata={
                "room_code": room.code,
                "seat_count": len(active_seats),
                "seat_codes": [seat.code for seat in active_seats],
            },
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "SEAT_CODE_DUPLICATE",
            "Seat codes must be unique inside a room",
            field_name="seats",
        ) from error
    return list_active_seats(db, room.id)
