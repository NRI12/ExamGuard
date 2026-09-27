"""Measure the Phase 4 realtime monitoring pipeline on a representative video."""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2

from app.monitoring.config import load_runtime_profile_from_paths
from app.monitoring.worker import VideoAnalysisWorker


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO11n + ByteTrack realtime runtime")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--profile", choices=("gtx1650", "rtx3060"), default="gtx1650")
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--config-root", type=Path, default=Path("/app/configs"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracking-output", type=Path)
    parser.add_argument("--annotated-video", type=Path)
    return parser.parse_args()


def _pixel_bbox(
    bbox_norm: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    return (
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    )


def write_tracking_jsonl(path: Path, tracking: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for message in tracking:
            output.write(json.dumps(message, ensure_ascii=False) + "\n")


def write_annotated_video(
    source: Path,
    destination: Path,
    tracking: list[dict[str, Any]],
    *,
    tolerance_ms: int = 250,
) -> None:
    if not tracking:
        raise RuntimeError("Không có tracking frame để tạo video kiểm tra")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Không thể mở video để vẽ tracking: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Metadata video không hợp lệ khi tạo video kiểm tra")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Không thể tạo video kiểm tra: {destination}")

    timestamps = [int(message["timestamp_ms"]) for message in tracking]
    final_timestamp = timestamps[-1]
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = round(frame_id / fps * 1000)
            if timestamp_ms > final_timestamp:
                break
            insertion = bisect.bisect_left(timestamps, timestamp_ms)
            candidate_indexes = (insertion - 1, insertion)
            nearest_index = min(
                (index for index in candidate_indexes if 0 <= index < len(tracking)),
                key=lambda index: abs(timestamps[index] - timestamp_ms),
            )
            message = tracking[nearest_index]
            if abs(int(message["timestamp_ms"]) - timestamp_ms) <= tolerance_ms:
                for track in message["tracks"]:
                    x1, y1, x2, y2 = _pixel_bbox(track["bbox_norm"], width, height)
                    color = (238, 211, 34)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {int(track['track_id']):02d}"
                    cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 82, y1), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (15, 23, 42),
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            frame_id += 1
    finally:
        writer.release()
        capture.release()


def main() -> None:
    args = parse_args()
    if args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    profile = load_runtime_profile_from_paths(
        config_root=args.config_root,
        model_root=args.model_root,
        profile_name=args.profile,
    )
    ready = threading.Event()
    completed = threading.Event()
    errors: list[str] = []
    messages: list[tuple[float, dict[str, Any]]] = []

    def publish(message: dict[str, Any]) -> None:
        messages.append((time.monotonic(), message))

    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=video,
        profile=profile,
        start_timestamp_ms=0,
        publish=publish,
        on_ready=ready.set,
        on_complete=completed.set,
        on_error=lambda error: (errors.append(error), completed.set()),
    )
    wall_started = time.monotonic()
    worker.start()
    if not ready.wait(timeout=60):
        worker.stop()
        raise RuntimeError(errors[0] if errors else "AI runtime initialization timed out")
    inference_started = time.monotonic()
    completed.wait(timeout=args.duration_seconds)
    elapsed = min(time.monotonic() - inference_started, args.duration_seconds)
    worker.stop()
    if errors:
        raise RuntimeError(errors[0])

    tracking = [(at, item) for at, item in messages if item.get("type") == "tracking"]
    tracking_messages = [item for _, item in tracking]
    diagnostics = [item for _, item in messages if item.get("type") == "diagnostics"]
    measured_window = tracking[-1][0] - tracking[0][0] if len(tracking) > 1 else elapsed
    analysis_fps = (len(tracking) - 1) / measured_window if measured_window > 0 else 0.0
    lags = [float(item["analysis_lag_ms"]) for item in diagnostics]
    gpu_values = [
        float(item["gpu_util_pct"])
        for item in diagnostics
        if item["gpu_util_pct"] is not None
    ]
    vram_values = [
        float(item["vram_used_mb"])
        for item in diagnostics
        if item["vram_used_mb"] is not None
    ]
    report = {
        "video": str(video),
        "profile": profile.profile,
        "device": profile.device,
        "precision": profile.precision,
        "requested_duration_seconds": args.duration_seconds,
        "initialization_seconds": inference_started - wall_started,
        "measured_seconds": elapsed,
        "source_fps": diagnostics[-1]["source_fps"] if diagnostics else None,
        "target_analysis_fps": profile.analysis.target_fps,
        "actual_analysis_fps": analysis_fps,
        "effective_analysis_fps_including_cold_start": len(tracking) / elapsed,
        "tracking_frames": len(tracking),
        "detector_ms_average": average([float(item["detector_ms"]) for item in diagnostics]),
        "detector_ms_steady_average": average(
            [float(item["detector_ms"]) for item in diagnostics[3:]]
        ),
        "tracker_ms_average": average([float(item["tracker_ms"]) for item in diagnostics]),
        "pipeline_ms_average": average([float(item["pipeline_ms"]) for item in diagnostics]),
        "pipeline_ms_steady_average": average(
            [float(item["pipeline_ms"]) for item in diagnostics[3:]]
        ),
        "analysis_lag_ms_average": average(lags),
        "analysis_lag_ms_first": lags[0] if lags else None,
        "analysis_lag_ms_last": lags[-1] if lags else None,
        "analysis_lag_growth_ms": lags[-1] - lags[0] if len(lags) > 1 else None,
        "gpu_util_pct_average": average(gpu_values),
        "gpu_util_pct_peak": max(gpu_values, default=None),
        "vram_used_mb_peak": max(vram_values, default=None),
        "cpu_util_pct_average": average([float(item["cpu_util_pct"]) for item in diagnostics]),
        "ram_used_mb_average": average([float(item["ram_used_mb"]) for item in diagnostics]),
        "dropped_analysis_frames": worker.dropped_analysis_frames,
        "queue_size_maximum": max((int(item["queue_size"]) for item in diagnostics), default=0),
        "active_tracks_average": average([float(len(item["tracks"])) for _, item in tracking]),
        "active_tracks_peak": max((len(item["tracks"]) for _, item in tracking), default=0),
        "tracking_output": str(args.tracking_output) if args.tracking_output else None,
        "annotated_video": str(args.annotated_video) if args.annotated_video else None,
    }
    if args.tracking_output:
        write_tracking_jsonl(args.tracking_output, tracking_messages)
    if args.annotated_video:
        write_annotated_video(video, args.annotated_video, tracking_messages)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
