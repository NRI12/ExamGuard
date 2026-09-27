from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from typing import Any
from urllib.parse import urlsplit

from fastapi import status

from app.core.errors import ApiError


def validate_rtsp_url(value: str) -> str:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise ValueError("URL must use rtsp:// or rtsps:// and include a host")
    return normalized


def _fps(stream: dict[str, Any]) -> float | None:
    value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        return None
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def probe_rtsp(url: str, *, timeout_seconds: int) -> dict[str, object]:
    timeout_microseconds = str(timeout_seconds * 1_000_000)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        timeout_microseconds,
        "-show_streams",
        "-of",
        "json",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 1,
        )
    except subprocess.TimeoutExpired as error:
        raise ApiError(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "RTSP_TIMEOUT",
            "Không thể kết nối camera trong thời gian cho phép.",
        ) from error
    except OSError as error:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "RTSP_PROBE_UNAVAILABLE",
            "Dịch vụ kiểm tra camera chưa sẵn sàng.",
        ) from error
    if completed.returncode != 0:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "RTSP_CONNECTION_FAILED",
            "Không thể kết nối hoặc đọc luồng RTSP. Hãy kiểm tra URL, tài khoản và mạng camera.",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ApiError(
            status.HTTP_502_BAD_GATEWAY,
            "RTSP_INVALID_RESPONSE",
            "Camera trả về thông tin luồng không hợp lệ.",
        ) from error
    streams = payload.get("streams") if isinstance(payload, dict) else None
    stream = next(
        (
            item
            for item in streams or []
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        None,
    )
    if stream is None:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "RTSP_NO_VIDEO",
            "Nguồn RTSP không có luồng hình ảnh.",
        )
    return {
        "connected": True,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": _fps(stream),
        "message": "Kết nối camera thành công.",
    }
