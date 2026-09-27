from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any

from app.ai.logical_tracking.types import ActorState
from app.ai.seat_identity.types import SeatRuntimeSnapshot, TrackIdentity


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    identity: TrackIdentity
    actor_id: str = ""
    actor_state: ActorState = ActorState.ACTIVE
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    session_id: uuid.UUID
    runtime_instance_id: uuid.UUID
    runtime_generation: int
    tracker_instance_id: uuid.UUID
    tracking_seq: int
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    tracks: tuple[Track, ...]
    seats: tuple[SeatRuntimeSnapshot, ...]

    def as_message(self) -> dict[str, Any]:
        return {
            "type": "tracking",
            "session_id": str(self.session_id),
            "runtime_instance_id": str(self.runtime_instance_id),
            "runtime_generation": self.runtime_generation,
            "tracker_instance_id": str(self.tracker_instance_id),
            "tracking_seq": self.tracking_seq,
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "tracks": [
                {
                    "actor_id": track.actor_id,
                    "actor_state": track.actor_state.value,
                    "recovered": track.recovered,
                    "track_id": track.track_id,
                    "bbox_norm": track.bbox_norm,
                    "confidence": track.confidence,
                    "identity": {
                        "state": track.identity.state.value,
                        "seat_id": (
                            str(track.identity.seat_id) if track.identity.seat_id else None
                        ),
                        "seat_code": track.identity.seat_code,
                        "session_candidate_id": (
                            str(track.identity.session_candidate_id)
                            if track.identity.session_candidate_id
                            else None
                        ),
                        "score": track.identity.score,
                    },
                }
                for track in self.tracks
            ],
            "seats": [
                {
                    "seat_id": str(seat.seat_id),
                    "seat_code": seat.seat_code,
                    "session_candidate_id": (
                        str(seat.session_candidate_id) if seat.session_candidate_id else None
                    ),
                    "state": seat.state.value,
                    "track_id": seat.track_id,
                }
                for seat in self.seats
            ],
        }


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    session_id: uuid.UUID
    runtime_instance_id: uuid.UUID
    runtime_generation: int
    worker_instance_id: uuid.UUID
    tracker_instance_id: uuid.UUID
    tracking_seq: int
    latest_frame_id: int
    latest_timestamp_ms: int
    raw_detection_count: int
    active_track_count: int
    source_fps: float
    target_analysis_fps: float
    analysis_fps: float
    detector_ms: float | None
    tracker_ms: float | None
    pipeline_ms: float | None
    seat_assignment_ms: float | None
    analysis_lag_ms: float
    gpu_util_pct: float | None
    vram_used_mb: float | None
    cpu_util_pct: float | None
    ram_used_mb: float | None
    dropped_analysis_frames: int
    queue_size: int
    assigned_tracks: int
    tentative_tracks: int
    unassigned_tracks: int
    occupied_seats: int
    grace_seats: int
    empty_seats: int
    seat_switches: int
    identity_recoveries: int
    profile: str
    active_logical_actors: int = 0
    lost_logical_actors: int = 0
    raw_track_count: int = 0
    recoveries_total: int = 0
    motion_recoveries: int = 0
    reid_recoveries: int = 0
    ambiguous_recoveries: int = 0
    reid_requests_total: int = 0
    reid_batches_total: int = 0
    reid_dropped_stale: int = 0
    logical_tracking_ms: float | None = None
    reid_latency_ms_mean: float | None = None
    reid_latency_ms_p95: float | None = None
    dynamic_pairs: int = 0
    active_single_proposals: int = 0
    active_pair_proposals: int = 0
    ready_action_buffers: int = 0
    active_action_buffers: int = 0
    buffered_roi_frames: int = 0
    action_predictions_total: int = 0
    action_predictions_per_second: float = 0.0
    tsm_preprocess_ms_mean: float | None = None
    tsm_preprocess_ms_p95: float | None = None
    tsm_inference_ms_mean: float | None = None
    tsm_inference_ms_p95: float | None = None
    action_pipeline_ms_mean: float | None = None
    action_pipeline_ms_p95: float | None = None
    action_batch_size_mean: float | None = None
    action_batch_size_p95: float | None = None
    action_queue_depth: int = 0
    stale_action_requests_dropped: int = 0
    action_device: str | None = None
    scheduler_ready_proposals: int = 0
    scheduler_in_flight_proposals: int = 0
    expired_ready_requests: int = 0
    replaced_ready_requests: int = 0
    action_batches_total: int = 0
    single_predictions_per_second: float = 0.0
    pair_predictions_per_second: float = 0.0
    single_prediction_interval_ms_mean: float | None = None
    single_prediction_interval_ms_p95: float | None = None
    single_prediction_interval_ms_max: float | None = None
    pair_prediction_interval_ms_mean: float | None = None
    pair_prediction_interval_ms_p95: float | None = None
    pair_prediction_interval_ms_max: float | None = None
    action_prediction_age_ms_mean: float | None = None
    action_prediction_age_ms_p95: float | None = None
    tsm_forward_ms_mean: float | None = None
    tsm_forward_ms_p95: float | None = None
    candidate_fsms: int = 0
    active_fsms: int = 0
    cooldown_fsms: int = 0
    events_created_total: int = 0
    events_suppressed_total: int = 0
    events_deduplicated_total: int = 0
    per_behavior_event_count: dict[str, int] | None = None

    def as_message(self) -> dict[str, Any]:
        message = asdict(self)
        message.update(
            session_id=str(self.session_id),
            runtime_instance_id=str(self.runtime_instance_id),
            worker_instance_id=str(self.worker_instance_id),
            tracker_instance_id=str(self.tracker_instance_id),
        )
        return {"type": "diagnostics", **message}


def normalize_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Source dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    values = (x1 / width, y1 / height, x2 / width, y2 / height)
    clamped = tuple(max(0.0, min(1.0, float(value))) for value in values)
    return clamped  # type: ignore[return-value]


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def suspicious_detection_overlaps(
    detections: list[Detection],
    threshold: float,
) -> list[dict[str, float | int]]:
    overlaps: list[dict[str, float | int]] = []
    for left_index, left in enumerate(detections):
        for right_index in range(left_index + 1, len(detections)):
            iou = bbox_iou(left.bbox_xyxy, detections[right_index].bbox_xyxy)
            if iou >= threshold:
                overlaps.append({"left": left_index, "right": right_index, "iou": round(iou, 4)})
    return overlaps
