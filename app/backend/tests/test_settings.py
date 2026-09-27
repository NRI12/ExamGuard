from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tests.conftest import make_settings


def test_settings_accept_clean_phase_one_configuration() -> None:
    settings = make_settings()

    assert settings.app_env == "test"
    assert settings.jwt_algorithm == "HS256"
    assert settings.database_url.startswith("postgresql+psycopg://")


@pytest.mark.parametrize(
    ("overrides", "error_fragment"),
    [
        ({"DATABASE_URL": "sqlite:///examguard.db"}, "postgresql"),
        ({"JWT_SECRET": "too-short"}, "at least 32"),
        (
            {"APP_ENV": "production", "JWT_SECRET": "change-me-with-more-than-32-characters"},
            "placeholder",
        ),
    ],
)
def test_settings_reject_invalid_configuration(
    overrides: dict[str, object],
    error_fragment: str,
) -> None:
    with pytest.raises(ValidationError, match=error_fragment):
        make_settings(**overrides)


def test_old_secret_key_alias_is_not_accepted() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg,arg-type]
            _env_file=None,
            APP_ENV="test",
            DATABASE_URL="postgresql+psycopg://examguard:examguard@postgres/examguard",
            SECRET_KEY="old-alias-must-not-work",
            UPLOAD_ROOT=Path("/tmp/uploads"),
            EVIDENCE_ROOT=Path("/tmp/evidence"),
            MODEL_ROOT=Path("/tmp/models"),
        )
