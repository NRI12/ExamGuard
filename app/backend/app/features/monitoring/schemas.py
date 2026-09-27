from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class MonitoringPosition(BaseModel):
    timestamp_ms: int = Field(default=0, ge=0)


class MonitoringStatusResponse(BaseModel):
    session_id: uuid.UUID
    state: Literal["INACTIVE", "INITIALIZING", "RUNNING", "PAUSED", "COMPLETED", "ERROR"]
    profile: str | None
    error: str | None
    subscriber_count: int = Field(ge=0)
    queue_size: int = Field(ge=0, le=1)
    dropped_analysis_frames: int = Field(ge=0)
    diagnostics: dict[str, Any] | None
    runtime_instance_id: uuid.UUID | None = None
    runtime_generation: int | None = Field(default=None, ge=0)
    worker_instance_id: uuid.UUID | None = None
    tracker_instance_id: uuid.UUID | None = None
    tracking_seq: int = Field(default=0, ge=0)


class TrackIdentityMessage(BaseModel):
    state: Literal["UNASSIGNED", "TENTATIVE", "ASSIGNED"]
    seat_id: uuid.UUID | None
    seat_code: str | None
    session_candidate_id: uuid.UUID | None
    score: float | None = Field(default=None, ge=0)


class TrackingTrackMessage(BaseModel):
    actor_id: str = Field(min_length=1)
    actor_state: Literal["ACTIVE", "LOST", "EXPIRED"]
    recovered: bool = False
    track_id: int
    bbox_norm: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    identity: TrackIdentityMessage


class SeatRuntimeMessage(BaseModel):
    seat_id: uuid.UUID
    seat_code: str
    session_candidate_id: uuid.UUID | None
    state: Literal["EMPTY", "OCCUPIED", "GRACE"]
    track_id: int | None


class TrackingMessage(BaseModel):
    type: Literal["tracking"]
    session_id: uuid.UUID
    runtime_instance_id: uuid.UUID
    runtime_generation: int = Field(ge=0)
    tracker_instance_id: uuid.UUID
    tracking_seq: int = Field(gt=0)
    timestamp_ms: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    tracks: list[TrackingTrackMessage]
    seats: list[SeatRuntimeMessage]


class DiagnosticsMessage(BaseModel):
    type: Literal["diagnostics"]
    session_id: uuid.UUID
    runtime_instance_id: uuid.UUID
    runtime_generation: int = Field(ge=0)
    worker_instance_id: uuid.UUID
    tracker_instance_id: uuid.UUID
    tracking_seq: int = Field(ge=0)
    latest_frame_id: int = Field(ge=0)
    latest_timestamp_ms: int = Field(ge=0)
    raw_detection_count: int = Field(ge=0)
    active_track_count: int = Field(ge=0)
    source_fps: float = Field(gt=0)
    target_analysis_fps: float = Field(gt=0)
    analysis_fps: float = Field(ge=0)
    detector_ms: float | None = Field(default=None, ge=0)
    tracker_ms: float | None = Field(default=None, ge=0)
    pipeline_ms: float | None = Field(default=None, ge=0)
    seat_assignment_ms: float | None = Field(default=None, ge=0)
    analysis_lag_ms: float = Field(ge=0)
    gpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    vram_used_mb: float | None = Field(default=None, ge=0)
    cpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    ram_used_mb: float | None = Field(default=None, ge=0)
    dropped_analysis_frames: int = Field(ge=0)
    queue_size: int = Field(ge=0, le=1)
    assigned_tracks: int = Field(ge=0)
    tentative_tracks: int = Field(ge=0)
    unassigned_tracks: int = Field(ge=0)
    occupied_seats: int = Field(ge=0)
    grace_seats: int = Field(ge=0)
    active_logical_actors: int = Field(default=0, ge=0)
    lost_logical_actors: int = Field(default=0, ge=0)
    raw_track_count: int = Field(default=0, ge=0)
    recoveries_total: int = Field(default=0, ge=0)
    motion_recoveries: int = Field(default=0, ge=0)
    reid_recoveries: int = Field(default=0, ge=0)
    ambiguous_recoveries: int = Field(default=0, ge=0)
    reid_requests_total: int = Field(default=0, ge=0)
    reid_batches_total: int = Field(default=0, ge=0)
    reid_dropped_stale: int = Field(default=0, ge=0)
    logical_tracking_ms: float | None = Field(default=None, ge=0)
    reid_latency_ms_mean: float | None = Field(default=None, ge=0)
    reid_latency_ms_p95: float | None = Field(default=None, ge=0)
    dynamic_pairs: int = Field(default=0, ge=0)
    empty_seats: int = Field(ge=0)
    seat_switches: int = Field(ge=0)
    identity_recoveries: int = Field(ge=0)
    active_single_proposals: int = Field(default=0, ge=0)
    active_pair_proposals: int = Field(default=0, ge=0)
    ready_action_buffers: int = Field(default=0, ge=0)
    active_action_buffers: int = Field(default=0, ge=0)
    buffered_roi_frames: int = Field(default=0, ge=0)
    action_predictions_total: int = Field(default=0, ge=0)
    action_predictions_per_second: float = Field(default=0, ge=0)
    tsm_preprocess_ms_mean: float | None = Field(default=None, ge=0)
    tsm_preprocess_ms_p95: float | None = Field(default=None, ge=0)
    tsm_inference_ms_mean: float | None = Field(default=None, ge=0)
    tsm_inference_ms_p95: float | None = Field(default=None, ge=0)
    action_pipeline_ms_mean: float | None = Field(default=None, ge=0)
    action_pipeline_ms_p95: float | None = Field(default=None, ge=0)
    action_batch_size_mean: float | None = Field(default=None, ge=0)
    action_batch_size_p95: float | None = Field(default=None, ge=0)
    action_queue_depth: int = Field(default=0, ge=0, le=2)
    stale_action_requests_dropped: int = Field(default=0, ge=0)
    action_device: str | None = None
    scheduler_ready_proposals: int = Field(default=0, ge=0)
    scheduler_in_flight_proposals: int = Field(default=0, ge=0)
    expired_ready_requests: int = Field(default=0, ge=0)
    replaced_ready_requests: int = Field(default=0, ge=0)
    action_batches_total: int = Field(default=0, ge=0)
    single_predictions_per_second: float = Field(default=0, ge=0)
    pair_predictions_per_second: float = Field(default=0, ge=0)
    single_prediction_interval_ms_mean: float | None = Field(default=None, ge=0)
    single_prediction_interval_ms_p95: float | None = Field(default=None, ge=0)
    single_prediction_interval_ms_max: float | None = Field(default=None, ge=0)
    pair_prediction_interval_ms_mean: float | None = Field(default=None, ge=0)
    pair_prediction_interval_ms_p95: float | None = Field(default=None, ge=0)
    pair_prediction_interval_ms_max: float | None = Field(default=None, ge=0)
    action_prediction_age_ms_mean: float | None = Field(default=None, ge=0)
    action_prediction_age_ms_p95: float | None = Field(default=None, ge=0)
    tsm_forward_ms_mean: float | None = Field(default=None, ge=0)
    tsm_forward_ms_p95: float | None = Field(default=None, ge=0)
    candidate_fsms: int = Field(default=0, ge=0)
    active_fsms: int = Field(default=0, ge=0)
    cooldown_fsms: int = Field(default=0, ge=0)
    events_created_total: int = Field(default=0, ge=0)
    events_suppressed_total: int = Field(default=0, ge=0)
    events_deduplicated_total: int = Field(default=0, ge=0)
    per_behavior_event_count: dict[str, int] | None = None
    profile: str


class RuntimeStateMessage(BaseModel):
    type: Literal["state"]
    session_id: uuid.UUID
    state: Literal["INACTIVE", "INITIALIZING", "RUNNING", "PAUSED", "COMPLETED", "ERROR"]
    synchronizing: bool
    error: str | None
    runtime_instance_id: uuid.UUID
    runtime_generation: int = Field(ge=0)
    worker_instance_id: uuid.UUID | None = None
    tracker_instance_id: uuid.UUID | None = None
    tracking_seq: int = Field(ge=0)
