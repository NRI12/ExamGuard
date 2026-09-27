"""Real-video Phase 6 integration/throughput diagnostic using production components."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]
import torch
import yaml  # type: ignore[import-untyped]

from app.ai.action_recognition.adapter import ActionModelAdapter
from app.ai.action_recognition.runtime import ActionRecognitionRuntime
from app.ai.detector.yolo import PersonDetector
from app.ai.domain import Track, normalize_bbox
from app.ai.seat_identity.assignment import SeatAssignmentEngine
from app.ai.seat_identity.types import AssignmentState, TrackIdentity
from app.ai.tracker.bytetrack import ByteTrackAdapter
from app.cli.validate_seat_identity import load_context
from app.monitoring.config import RuntimeProfile, load_runtime_profile_from_paths
from app.monitoring.decoder import VideoDecoder
from app.monitoring.metrics import read_system_metrics


def percentile(values: list[float], point: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * point))]


class ResourceSampler:
    def __init__(self, interval_s: float = 0.25) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._gpu: list[float] = []
        self._vram: list[float] = []
        self._cpu: list[float] = []
        self._rss: list[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return {
            "gpu_util_pct_mean": statistics.fmean(self._gpu) if self._gpu else None,
            "gpu_util_pct_p95": percentile(self._gpu, 0.95),
            "gpu_util_pct_max": max(self._gpu, default=None),
            "vram_used_mb_mean": statistics.fmean(self._vram) if self._vram else None,
            "vram_used_mb_peak": max(self._vram, default=None),
            "cpu_util_pct_mean": statistics.fmean(self._cpu) if self._cpu else None,
            "process_rss_mb_peak": max(self._rss, default=None),
        }

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            metrics = read_system_metrics()
            if metrics.gpu_util_pct is not None:
                self._gpu.append(metrics.gpu_util_pct)
            if metrics.vram_used_mb is not None:
                self._vram.append(metrics.vram_used_mb)
            if metrics.cpu_util_pct is not None:
                self._cpu.append(metrics.cpu_util_pct)
            self._rss.append(process.memory_info().rss / (1024 * 1024))
            self._stop.wait(self._interval_s)


def validate_clip(
    clip: dict[str, Any],
    profile: RuntimeProfile,
    *,
    sample_fps: float,
    duration_ms: int,
    action_model: ActionModelAdapter | None,
    realtime: bool,
    experiment_id: str,
) -> dict[str, Any]:
    clip_id = str(clip["id"])
    context, _ = load_context(clip_id, clip)
    source = Path(str(clip["source"]))
    detector = PersonDetector(profile.detector)
    tracker = ByteTrackAdapter(profile.tracker)
    seat_engine = SeatAssignmentEngine(context, profile.seat_assignment)
    decoder = VideoDecoder(source)
    messages: list[dict[str, Any]] = []
    action_runtime = (
        ActionRecognitionRuntime(
            session_id=context.session_id,
            runtime_instance_id=uuid.uuid4(),
            config=profile.action_recognition,
            identity_context=context,
            model=action_model,
            publish=messages.append,
        )
        if action_model is not None
        else None
    )
    start_ms = int(clip.get("start_ms", 0))
    end_ms = start_ms + min(duration_ms, int(clip["duration_ms"]))
    target_ms = float(start_ms)
    interval_ms = 1000.0 / sample_fps
    frames = 0
    assigned_count = 0
    tracking_durations: list[float] = []
    lag_samples: list[float] = []
    active_single_samples: list[float] = []
    active_pair_samples: list[float] = []
    ready_samples: list[float] = []
    active_candidate_samples: list[float] = []
    started = time.perf_counter()
    resources = ResourceSampler()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resources.start()
    decoder.seek(start_ms)
    try:
        while target_ms <= end_ms:
            if realtime:
                target_wall = started + (target_ms - start_ms) / 1000
                remaining = target_wall - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            packet, _ = decoder.read_for_timestamp(round(target_ms), 0)
            if packet is None:
                break
            tracking_started = time.perf_counter()
            detections = detector.detect(packet.frame)
            tracked = tracker.update(detections, packet.frame.shape[:2], packet.timestamp_ms)
            tracks = tuple(
                Track(
                    track_id=item.track_id,
                    bbox_norm=normalize_bbox(
                        item.bbox_xyxy, packet.source_width, packet.source_height
                    ),
                    confidence=item.confidence,
                    identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
                )
                for item in tracked
            )
            snapshot = seat_engine.update(tracks, packet.timestamp_ms)
            resolved = tuple(
                Track(
                    track_id=track.track_id,
                    bbox_norm=track.bbox_norm,
                    confidence=track.confidence,
                    identity=snapshot.identities[track.track_id],
                )
                for track in tracks
            )
            tracking_durations.append((time.perf_counter() - tracking_started) * 1000)
            assigned_count += snapshot.assigned_tracks
            active_candidate_samples.append(float(snapshot.assigned_tracks))
            if action_runtime is not None:
                action_runtime.update(packet.frame, resolved, packet.timestamp_ms, 0)
                current = action_runtime.diagnostics()
                active_single_samples.append(float(current.active_single_proposals))
                active_pair_samples.append(float(current.active_pair_proposals))
                ready_samples.append(float(current.scheduler_ready_proposals))
            elapsed_ms = (time.perf_counter() - started) * 1000
            lag_samples.append(max(0.0, elapsed_ms - (packet.timestamp_ms - start_ms)))
            frames += 1
            target_ms += interval_ms
        processing_elapsed_s = time.perf_counter() - started
        idle = action_runtime.wait_idle(timeout=30.0) if action_runtime else True
        action_diagnostics = action_runtime.diagnostics() if action_runtime else None
    finally:
        decoder.release()
        if action_runtime is not None:
            action_runtime.close(timeout=30.0)
        resource_metrics = resources.stop()
    predictions = [
        prediction
        for message in messages
        if message.get("type") == "action_prediction"
        for prediction in message["predictions"]
    ]
    errors = [message["error"] for message in messages if message.get("type") == "action_error"]
    single_ids = {
        prediction["proposal_id"]
        for prediction in predictions
        if prediction["proposal_type"] == "SINGLE"
    }
    pair_ids = {
        prediction["proposal_id"]
        for prediction in predictions
        if prediction["proposal_type"] == "PAIR"
    }
    source_elapsed_s = (
        max(0, frames - 1) / sample_fps if frames else 0.0
    )
    first_lag = lag_samples[: max(1, len(lag_samples) // 5)]
    last_lag = lag_samples[-max(1, len(lag_samples) // 5) :]
    return {
        "experiment_id": experiment_id,
        "clip_id": clip_id,
        "source": str(source),
        "mode": "tracking_action" if action_model else "tracking_only",
        "realtime_pacing": realtime,
        "frames": frames,
        "assigned_track_samples": assigned_count,
        "processing_elapsed_s": processing_elapsed_s,
        "tracking_fps": frames / processing_elapsed_s if processing_elapsed_s > 0 else None,
        "tracking_frame_ms_mean": statistics.fmean(tracking_durations)
        if tracking_durations
        else None,
        "analysis_lag_ms_mean": statistics.fmean(lag_samples) if lag_samples else None,
        "analysis_lag_ms_p95": percentile(lag_samples, 0.95),
        "analysis_lag_ms_max": max(lag_samples, default=None),
        "analysis_lag_growth_ms": (
            statistics.fmean(last_lag) - statistics.fmean(first_lag)
            if lag_samples
            else None
        ),
        "active_candidates_mean": (
            statistics.fmean(active_candidate_samples) if active_candidate_samples else 0.0
        ),
        "active_candidates_max": max(active_candidate_samples, default=0.0),
        "active_single_proposals_mean": (
            statistics.fmean(active_single_samples) if active_single_samples else 0.0
        ),
        "active_single_proposals_max": max(active_single_samples, default=0.0),
        "active_pair_proposals_mean": (
            statistics.fmean(active_pair_samples) if active_pair_samples else 0.0
        ),
        "active_pair_proposals_max": max(active_pair_samples, default=0.0),
        "ready_proposals_mean": statistics.fmean(ready_samples) if ready_samples else 0.0,
        "ready_proposals_max": max(ready_samples, default=0.0),
        "single_proposals_with_predictions": len(single_ids),
        "pair_proposals_with_predictions": len(pair_ids),
        "action_predictions": len(predictions),
        "action_predictions_per_second": (
            len(predictions) / source_elapsed_s if source_elapsed_s > 0 else None
        ),
        "sample_predictions": predictions[:3],
        "action_diagnostics": (
            asdict(action_diagnostics) if action_diagnostics is not None else None
        ),
        "action_errors": errors,
        "action_idle_before_stop": idle,
        "resources": {
            **resource_metrics,
            "torch_allocated_mb": (
                torch.cuda.memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
            "torch_reserved_mb": (
                torch.cuda.memory_reserved() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
            "torch_peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("/app/configs"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--profile", choices=("gtx1650", "rtx3060"), default="gtx1650")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--duration-ms", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--clip-id")
    parser.add_argument("--experiment-id", default="ACTION-VALIDATION")
    parser.add_argument("--mode", choices=("paired", "tracking", "action"), default="paired")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--stride-ms", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-prediction-age-ms", type=int)
    parser.add_argument("--min-inference-interval-ms", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.duration_ms <= 0 or args.limit <= 0:
        parser.error("sample-fps, duration-ms and limit must be positive")
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("clips"), list):
        parser.error("manifest must contain clips")
    profile = load_runtime_profile_from_paths(
        config_root=args.config_root,
        model_root=args.model_root,
        profile_name=args.profile,
    )
    scheduling = profile.action_recognition.scheduling.model_copy(
        update={
            key: value
            for key, value in {
                "prediction_stride_ms": args.stride_ms,
                "max_batch_size": args.batch_size,
                "max_prediction_age_ms": args.max_prediction_age_ms,
                "min_inference_interval_ms": args.min_inference_interval_ms,
            }.items()
            if value is not None
        }
    )
    action_config = profile.action_recognition.model_copy(
        update={
            "precision": args.precision
            or ("fp32" if args.device.startswith("cuda") else "fp32"),
            "scheduling": scheduling,
        }
    )
    profile = profile.model_copy(
        update={
            "device": args.device,
            "detector": profile.detector.model_copy(
                update={"device": args.device, "half": args.device.startswith("cuda")}
            ),
            "action_recognition": action_config,
        }
    )
    clips: list[dict[str, Any]] = []
    selected_clips = [
        raw_clip
        for raw_clip in manifest["clips"]
        if args.clip_id is None or str(raw_clip.get("id")) == args.clip_id
    ][: args.limit]
    if not selected_clips:
        parser.error(f"clip-id not found in manifest: {args.clip_id}")
    for raw_clip in selected_clips:
        clip = dict(raw_clip)
        source = Path(str(clip["source"]))
        if not source.is_absolute():
            clip["source"] = str((args.manifest.parent / source).resolve())
        clips.append(clip)
    results: list[dict[str, Any]] = []
    modes = ("tracking", "action") if args.mode == "paired" else (args.mode,)
    model: ActionModelAdapter | None = None
    for mode in modes:
        if mode == "action" and model is None:
            model = ActionModelAdapter(action_config, args.device)
            model.ensure_loaded()
        for clip in clips:
            result = validate_clip(
                clip,
                profile,
                sample_fps=args.sample_fps,
                duration_ms=args.duration_ms,
                action_model=model if mode == "action" else None,
                realtime=args.realtime,
                experiment_id=args.experiment_id,
            )
            results.append(result)
            if not args.quiet:
                print(json.dumps(result, ensure_ascii=False), flush=True)
    output = {
        "experiment_id": args.experiment_id,
        "device": args.device,
        "precision": action_config.precision,
        "scheduling": action_config.scheduling.model_dump(),
        "model_load_count": model.load_count if model else 0,
        "results": results,
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "experiment_id": args.experiment_id,
                    "output": str(args.output) if args.output else None,
                    "result_count": len(results),
                }
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
