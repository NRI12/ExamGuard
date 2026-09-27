import uuid

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.room import Room, Seat

DOMAIN_TABLES = {
    "appeal_cases",
    "appeal_events",
    "audit_logs",
    "cameras",
    "candidates",
    "event_actors",
    "event_reviews",
    "events",
    "evidence_assets",
    "exam_sessions",
    "media_assets",
    "rooms",
    "seats",
    "session_candidates",
    "users",
}


def test_model_metadata_contains_exact_domain_tables() -> None:
    assert set(Base.metadata.tables) == DOMAIN_TABLES


def test_database_contains_exact_domain_tables(db: Session) -> None:
    assert set(inspect(db.get_bind()).get_table_names()) == DOMAIN_TABLES


def test_session_rollback_does_not_persist_data(db: Session) -> None:
    room = Room(code="ROLLBACK", name="Rollback room")
    db.add(room)
    db.flush()
    db.rollback()

    assert db.scalar(select(Room).where(Room.code == "ROLLBACK")) is None


def test_seat_geometry_constraint_rejects_out_of_bounds_region(db: Session) -> None:
    room = Room(code="A", name="Room A")
    db.add(room)
    db.flush()
    db.add(
        Seat(
            room_id=room.id,
            code="A01",
            x=0.8,
            y=0.1,
            width=0.3,
            height=0.3,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()


def test_critical_unique_and_check_constraints_exist() -> None:
    seats = Base.metadata.tables["seats"]
    seat_constraint_names = {constraint.name for constraint in seats.constraints}
    assert "uq_seats_room_id_code" in seat_constraint_names
    assert "ck_seats_horizontal_bounds" in seat_constraint_names
    assert "ck_seats_vertical_bounds" in seat_constraint_names

    assignments = Base.metadata.tables["session_candidates"]
    assignment_constraints = {constraint.name for constraint in assignments.constraints}
    assert "uq_session_candidates_session_candidate" in assignment_constraints
    assert "uq_session_candidates_session_seat" in assignment_constraints

    event = Base.metadata.tables["events"]
    event_constraints = {constraint.name for constraint in event.constraints}
    assert "ck_events_time_range" in event_constraints
    assert "ck_events_ai_confidence_range" in event_constraints


def test_uuid_primary_keys_are_generated(db: Session) -> None:
    room = Room(code="UUID", name="UUID room")
    db.add(room)
    db.flush()

    assert isinstance(room.id, uuid.UUID)
