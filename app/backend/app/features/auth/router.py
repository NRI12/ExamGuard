from fastapi import APIRouter, Response, status

from app.core.errors import ApiError
from app.core.security import create_access_token
from app.features.auth.dependencies import (
    MEDIA_COOKIE_NAME,
    WEBSOCKET_COOKIE_NAME,
    ApplicationSettings,
    CurrentUser,
    DatabaseSession,
)
from app.features.auth.schemas import LoginRequest, LoginResponse, UserResponse
from app.features.auth.service import authenticate

router = APIRouter(prefix="/auth", tags=["auth"])


def set_media_cookie(response: Response, access_token: str, settings: ApplicationSettings) -> None:
    response.set_cookie(
        key=MEDIA_COOKIE_NAME,
        value=access_token,
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
        path="/api/v1/media",
    )
    response.set_cookie(
        key=WEBSOCKET_COOKIE_NAME,
        value=access_token,
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
        path="/ws",
    )


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    response: Response,
    db: DatabaseSession,
    settings: ApplicationSettings,
) -> LoginResponse:
    user = authenticate(db, payload.username, payload.password)
    if user is None:
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "INVALID_CREDENTIALS",
            "Invalid username or password",
        )
    if not user.is_active:
        raise ApiError(status.HTTP_403_FORBIDDEN, "USER_INACTIVE", "User is inactive")

    access_token = create_access_token(user.id, settings)
    set_media_cookie(response, access_token, settings)
    return LoginResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(
        key=MEDIA_COOKIE_NAME,
        path="/api/v1/media",
        httponly=True,
        samesite="strict",
    )
    response.delete_cookie(
        key=WEBSOCKET_COOKIE_NAME,
        path="/ws",
        httponly=True,
        samesite="strict",
    )


@router.get("/me", response_model=UserResponse)
def me(
    current_user: CurrentUser,
    response: Response,
    settings: ApplicationSettings,
) -> UserResponse:
    set_media_cookie(response, create_access_token(current_user.id, settings), settings)
    return UserResponse.model_validate(current_user)
