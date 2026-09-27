import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.features.sessions.schemas import SessionCreate


def test_session_create_calculates_scheduled_end_from_duration() -> None:
    start = datetime(2026, 9, 24, 8, 30, tzinfo=UTC)
    payload = SessionCreate(
        session_code="MORNING-01",
        exam_name="Morning exam",
        room_id=uuid.uuid4(),
        scheduled_start=start,
        duration_minutes=90,
    )
    assert payload.scheduled_end == start + timedelta(minutes=90)


def test_session_duration_requires_scheduled_start() -> None:
    with pytest.raises(ValidationError):
        SessionCreate(
            session_code="MORNING-02",
            exam_name="Morning exam",
            room_id=uuid.uuid4(),
            duration_minutes=90,
        )
