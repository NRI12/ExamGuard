from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.features.auth.dependencies import SessionMonitor
from app.features.monitoring.rtsp import probe_rtsp, validate_rtsp_url

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class RtspProbeRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)

    _validate_url = field_validator("url")(validate_rtsp_url)


class RtspProbeResponse(BaseModel):
    connected: bool
    codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    message: str


@router.post("/rtsp/check", response_model=RtspProbeResponse)
def check_rtsp_connection(
    payload: RtspProbeRequest,
    _: SessionMonitor,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    return probe_rtsp(payload.url, timeout_seconds=min(settings.ffprobe_timeout_seconds, 10))
