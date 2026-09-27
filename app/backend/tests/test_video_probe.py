import subprocess
from pathlib import Path

import pytest

from app.video.probe import VideoProbeError, parse_fps, parse_probe_output, probe_video


def payload(**overrides: object) -> dict[str, object]:
    stream: dict[str, object] = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30000/1001",
        "duration": "10.5",
    }
    stream.update(overrides)
    return {
        "streams": [stream],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.5"},
    }


def test_parse_rational_fps() -> None:
    assert parse_fps("25/1") == 25.0
    assert parse_fps("30000/1001") == pytest.approx(29.97003, rel=1e-5)


def test_rejects_no_video_stream() -> None:
    with pytest.raises(VideoProbeError, match="video stream"):
        parse_probe_output({"streams": [{"codec_type": "audio"}], "format": {}})


@pytest.mark.parametrize("duration", ["0", "-1", "N/A"])
def test_rejects_invalid_duration(duration: str) -> None:
    with pytest.raises(VideoProbeError, match="duration"):
        parse_probe_output(payload(duration=duration))


def test_handles_ffprobe_command_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["ffprobe"], 1, "", "bad")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(VideoProbeError, match="rejected"):
        probe_video(Path("invalid.mp4"))


def test_reads_small_valid_mp4(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "color=black:s=32x24:r=25", "-t", "0.2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-an", str(path),
        ],
        check=True,
        capture_output=True,
    )
    metadata = probe_video(path)
    assert metadata.codec == "h264"
    assert (metadata.width, metadata.height) == (32, 24)
    assert metadata.fps == 25.0
    assert metadata.duration_ms > 0
    assert "mp4" in metadata.format_names
