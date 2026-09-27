import uuid
from datetime import timedelta

import jwt
import pytest

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification() -> None:
    encoded = hash_password("correct horse battery staple")

    assert encoded != "correct horse battery staple"
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_jwt_round_trip(settings: Settings) -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id, settings)

    assert decode_access_token(token, settings) == user_id


def test_expired_jwt_is_rejected(settings: Settings) -> None:
    token = create_access_token(uuid.uuid4(), settings, expires_delta=timedelta(seconds=-1))

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, settings)


def test_invalid_jwt_is_rejected(settings: Settings) -> None:
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token("not-a-jwt", settings)
