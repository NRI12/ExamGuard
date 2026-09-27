import json
import subprocess

import pytest

from app.core.errors import ApiError
from app.features.monitoring.rtsp import probe_rtsp, validate_rtsp_url


def test_validate_rtsp_url_accepts_only_rtsp_sources() -> None:
    assert validate_rtsp_url(" rtsp://camera.local/live ") == "rtsp://camera.local/live"
    assert validate_rtsp_url("rtsps://camera.local/secure") == "rtsps://camera.local/secure"
    with pytest.raises(ValueError):
        validate_rtsp_url("https://camera.local/live")
    with pytest.raises(ValueError):
        validate_rtsp_url("rtsp:///missing-host")


def test_probe_rtsp_returns_video_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "25/1",
            }
        ]
    }

    def completed(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ffprobe"], 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", completed)
    result = probe_rtsp("rtsp://camera.local/live", timeout_seconds=3)
    assert result == {
        "connected": True,
        "codec": "h264",
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "message": "Kết nối camera thành công.",
    }


def test_probe_rtsp_reports_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ffprobe"], 1, "", "connection refused")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ApiError) as captured:
        probe_rtsp("rtsp://camera.local/live", timeout_seconds=3)
    assert captured.value.code == "RTSP_CONNECTION_FAILED"
