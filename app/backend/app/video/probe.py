import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


class VideoProbeError(ValueError):
    """Raised when ffprobe cannot produce valid video metadata."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    codec: str
    width: int
    height: int
    fps: float
    duration_ms: int
    format_names: tuple[str, ...]


def parse_fps(value: object) -> float:
    if not isinstance(value, str) or not value or value in {"0/0", "N/A"}:
        raise VideoProbeError("Video frame rate is missing")
    try:
        fps = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        raise VideoProbeError("Video frame rate is invalid") from error
    if not math.isfinite(fps) or fps <= 0:
        raise VideoProbeError("Video frame rate must be positive")
    return fps


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise VideoProbeError(f"Video {field} is invalid") from error
    if parsed <= 0:
        raise VideoProbeError(f"Video {field} must be positive")
    return parsed


def _duration_ms(stream: dict[str, Any], format_data: dict[str, Any]) -> int:
    raw_duration = stream.get("duration") or format_data.get("duration")
    if not isinstance(raw_duration, (str, int, float)):
        raise VideoProbeError("Video duration is invalid")
    try:
        seconds = float(raw_duration)
    except (TypeError, ValueError) as error:
        raise VideoProbeError("Video duration is invalid") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise VideoProbeError("Video duration must be positive")
    duration_ms = round(seconds * 1000)
    if duration_ms <= 0:
        raise VideoProbeError("Video duration must be positive")
    return duration_ms


def parse_probe_output(payload: dict[str, Any]) -> VideoMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoProbeError("ffprobe did not return a stream list")
    stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(stream, dict):
        raise VideoProbeError("Media does not contain a video stream")
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        format_data = {}
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec.strip():
        raise VideoProbeError("Video codec is missing")
    formats = tuple(
        value.strip().lower()
        for value in str(format_data.get("format_name", "")).split(",")
        if value.strip()
    )
    fps_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    return VideoMetadata(
        codec=codec.strip().lower(),
        width=_positive_int(stream.get("width"), "width"),
        height=_positive_int(stream.get("height"), "height"),
        fps=parse_fps(fps_value),
        duration_ms=_duration_ms(stream, format_data),
        format_names=formats,
    )


def probe_video(path: Path, *, timeout_seconds: int = 30) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VideoProbeError("ffprobe could not inspect the media") from error
    if completed.returncode != 0:
        raise VideoProbeError("ffprobe rejected the media")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise VideoProbeError("ffprobe returned invalid output") from error
    if not isinstance(payload, dict):
        raise VideoProbeError("ffprobe returned invalid output")
    return parse_probe_output(payload)
