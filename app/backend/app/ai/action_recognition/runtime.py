from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from app.ai.action_recognition.adapter import (
    R3_CLASS_NAMES,
    ActionModelAdapter,
    ActionModelResult,
)
from app.ai.action_recognition.buffer import TimestampedRoiBuffer
from app.ai.action_recognition.proposals import (
    adjacent_seat_pairs,
    build_action_proposals,
    build_logical_action_proposals,
)
from app.ai.action_recognition.roi import (
    RoiPreparationProfile,
    extract_training_roi_profiled,
)
from app.ai.action_recognition.scheduler import ActionScheduler
from app.ai.action_recognition.types import ActionClip, ActionPrediction, ProposalType
from app.ai.domain import Track
from app.ai.logical_tracking.neighbors import dynamic_neighbor_pairs
from app.ai.seat_identity.types import SeatIdentityContext
from app.monitoring.config import (
    ActionRecognitionConfig,
    ActionRoiGeometryConfig,
    DynamicNeighborsConfig,
)


class ActionPredictor(Protocol):
    @property
    def device(self) -> str: ...

    def predict(self, clips: tuple[ActionClip, ...]) -> ActionModelResult: ...


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    runtime_generation: int
    timestamp_ms: int
    clips: tuple[ActionClip, ...]
    submitted_monotonic: float


@dataclass(frozen=True, slots=True)
class ActionFrame:
    runtime_generation: int
    timestamp_ms: int
    frame_bgr: np.ndarray
    tracks: tuple[Track, ...]


@dataclass(frozen=True, slots=True)
class ActionRuntimeDiagnostics:
    active_single_proposals: int
    active_pair_proposals: int
    dynamic_pairs: int
    ready_action_buffers: int
    active_action_buffers: int
    buffered_roi_frames: int
    action_predictions_total: int
    action_predictions_per_second: float
    tsm_preprocess_ms_mean: float | None
    tsm_preprocess_ms_p95: float | None
    tsm_inference_ms_mean: float | None
    tsm_inference_ms_p95: float | None
    action_pipeline_ms_mean: float | None
    action_pipeline_ms_p95: float | None
    action_batch_size_mean: float | None
    action_batch_size_p95: float | None
    action_queue_depth: int
    stale_action_requests_dropped: int
    action_device: str
    scheduler_ready_proposals: int
    scheduler_in_flight_proposals: int
    expired_ready_requests: int
    replaced_ready_requests: int
    action_batches_total: int
    single_predictions_per_second: float
    pair_predictions_per_second: float
    single_prediction_interval_ms_mean: float | None
    single_prediction_interval_ms_p95: float | None
    single_prediction_interval_ms_max: float | None
    pair_prediction_interval_ms_mean: float | None
    pair_prediction_interval_ms_p95: float | None
    pair_prediction_interval_ms_max: float | None
    action_prediction_age_ms_mean: float | None
    action_prediction_age_ms_p95: float | None
    timing_profiles: dict[str, dict[str, float | None]]


class ActionRecognitionRuntime:
    def __init__(
        self,
        *,
        session_id: uuid.UUID,
        runtime_instance_id: uuid.UUID,
        config: ActionRecognitionConfig,
        identity_context: SeatIdentityContext,
        model: ActionModelAdapter,
        publish: Callable[[dict[str, Any]], None],
        on_predictions: Callable[[tuple[ActionPrediction, ...], int], None] | None = None,
        identity_mode: str = "seat",
        dynamic_neighbors_config: DynamicNeighborsConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.runtime_instance_id = runtime_instance_id
        self.config = config
        self._model: ActionPredictor = model
        self._publish = publish
        self._on_predictions = on_predictions
        self._identity_mode = identity_mode
        self._dynamic_neighbors = dynamic_neighbors_config or DynamicNeighborsConfig()
        self._dynamic_pair_count = 0
        self._adjacent_pairs = identity_context.adjacent_seat_pairs or adjacent_seat_pairs(
            identity_context.seats,
            row_tolerance_ratio=config.adjacency.row_tolerance_ratio,
            max_gap_ratio=config.adjacency.max_horizontal_gap_ratio,
        )
        self._buffers = TimestampedRoiBuffer(
            clip_span_ms=config.clip_span_ms,
            num_segments=config.num_segments,
            capture_interval_ms=config.capture_interval_ms,
            sample_tolerance_ms=config.sample_tolerance_ms,
            max_gap_ms=config.max_gap_ms,
        )
        scheduling = config.scheduling
        self._scheduler = ActionScheduler(
            prediction_stride_ms=scheduling.prediction_stride_ms,
            max_batch_size=scheduling.max_batch_size,
            max_prediction_age_ms=scheduling.max_prediction_age_ms,
            min_inference_interval_ms=scheduling.min_inference_interval_ms,
        )
        self._frame_queue: queue.Queue[ActionFrame | None] = queue.Queue(
            maxsize=scheduling.max_queue_size
        )
        self._inference_queue: queue.Queue[InferenceBatch | None] = queue.Queue(
            maxsize=scheduling.max_queue_size
        )
        self._lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._queue_drops = 0
        self._stale_drops = 0
        self._active_single = 0
        self._active_pair = 0
        self._ready_buffers = 0
        self._prediction_count = 0
        self._batch_count = 0
        self._preprocess_samples: deque[float] = deque(maxlen=256)
        self._inference_samples: deque[float] = deque(maxlen=256)
        self._pipeline_samples: deque[float] = deque(maxlen=256)
        self._batch_samples: deque[float] = deque(maxlen=256)
        self._prediction_age_samples: deque[float] = deque(maxlen=1024)
        self._timing_samples: dict[str, deque[float]] = {
            name: deque(maxlen=256)
            for name in (
                "roi_prepare_ms",
                "roi_crop_ms",
                "roi_resize_ms",
                "roi_color_conversion_ms",
                "tensor_assembly_ms",
                "input_resize_ms",
                "normalization_ms",
                "h2d_ms",
                "forward_ms",
                "postprocess_ms",
                "model_total_ms",
            )
        }
        self._prediction_times: deque[float] = deque(maxlen=256)
        self._started_monotonic = time.monotonic()
        self._latest_timestamp_ms = 0
        self._inflight = False
        self._model_inflight = False
        self._buffer_inflight = False
        self._buffer_thread = threading.Thread(
            target=self._buffer_loop,
            name=f"action-buffer-{session_id}",
            daemon=True,
        )
        self._inference_thread = threading.Thread(
            target=self._infer_loop,
            name=f"action-inference-{session_id}",
            daemon=True,
        )
        self._buffer_thread.start()
        self._inference_thread.start()

    def update(
        self,
        frame_bgr: np.ndarray,
        tracks: tuple[Track, ...],
        timestamp_ms: int,
        runtime_generation: int,
    ) -> None:
        with self._lock:
            if self._closed or runtime_generation != self._generation:
                return
            self._latest_timestamp_ms = max(self._latest_timestamp_ms, timestamp_ms)
        self._put_latest(
            self._frame_queue,
            ActionFrame(runtime_generation, timestamp_ms, frame_bgr, tracks),
        )

    def _process_frame(self, item: ActionFrame) -> None:
        frame_bgr = item.frame_bgr
        tracks = item.tracks
        timestamp_ms = item.timestamp_ms
        runtime_generation = item.runtime_generation
        if self._identity_mode == "logical_track":
            neighbor_pairs = dynamic_neighbor_pairs(tracks, self._dynamic_neighbors)
            proposals = build_logical_action_proposals(
                tracks,
                neighbor_pairs,
                timestamp_ms,
            )
            self._dynamic_pair_count = len(neighbor_pairs)
        else:
            proposals = build_action_proposals(tracks, self._adjacent_pairs, timestamp_ms)
        with self._lock:
            if self._closed or runtime_generation != self._generation:
                return
            self._buffers.prune({proposal.proposal_id for proposal in proposals}, timestamp_ms)
            self._active_single = sum(
                proposal.proposal_type is ProposalType.SINGLE for proposal in proposals
            )
            self._active_pair = len(proposals) - self._active_single
        ready: list[ActionClip] = []
        for proposal in proposals:
            with self._lock:
                if self._closed or runtime_generation != self._generation:
                    return
                if not self._buffers.should_capture(proposal.proposal_id, timestamp_ms):
                    continue
            geometry = (
                self.config.single_roi
                if proposal.proposal_type is ProposalType.SINGLE
                else self.config.pair_roi
            )
            try:
                roi, roi_profile = self._extract(frame_bgr, proposal.bbox_norm, geometry)
            except (ValueError, cv2.error):
                continue
            with self._lock:
                if self._closed or runtime_generation != self._generation:
                    return
                self._record_roi_profile(roi_profile)
                self._buffers.append(proposal, timestamp_ms, roi)
                clip = self._buffers.clip_if_ready(proposal, timestamp_ms)
            if clip is not None:
                ready.append(clip)
        with self._lock:
            if self._closed or runtime_generation != self._generation:
                return
            self._scheduler.observe(tuple(ready), timestamp_ms)
            if self._inflight:
                self._ready_buffers = self._scheduler.snapshot().ready_proposals
                return
            selected = self._scheduler.next_batch(timestamp_ms)
            self._ready_buffers = self._scheduler.snapshot().ready_proposals
            if not selected:
                return
            self._inflight = True
        try:
            self._inference_queue.put_nowait(
                InferenceBatch(
                    runtime_generation=runtime_generation,
                    timestamp_ms=timestamp_ms,
                    clips=selected,
                    submitted_monotonic=time.monotonic(),
                )
            )
        except queue.Full:  # pragma: no cover - guarded by the single in-flight budget
            with self._lock:
                self._scheduler.fail(selected)
                self._inflight = False
                self._queue_drops += 1

    @staticmethod
    def _extract(
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        geometry: ActionRoiGeometryConfig,
    ) -> tuple[np.ndarray, RoiPreparationProfile]:
        return extract_training_roi_profiled(
            frame_bgr,
            bbox,
            expand_x=geometry.expand_x,
            expand_top=geometry.expand_top,
            expand_bottom=geometry.expand_bottom,
            min_width_px=geometry.min_crop_width_px,
            min_height_px=geometry.min_crop_height_px,
        )

    def _record_roi_profile(self, profile: RoiPreparationProfile) -> None:
        self._timing_samples["roi_prepare_ms"].append(profile.total_ms)
        self._timing_samples["roi_crop_ms"].append(profile.crop_ms)
        self._timing_samples["roi_resize_ms"].append(profile.resize_ms)
        self._timing_samples["roi_color_conversion_ms"].append(profile.color_conversion_ms)

    def _put_latest(self, target: queue.Queue[Any], item: Any) -> None:
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:  # pragma: no cover - another consumer won the race
                pass
            with self._lock:
                self._queue_drops += 1
            target.put_nowait(item)

    def reset(self, runtime_generation: int) -> None:
        with self._lock:
            self._generation = runtime_generation
            self._buffers.clear()
            self._drain(self._frame_queue)
            self._drain(self._inference_queue)
            self._scheduler.reset()
            self._active_single = 0
            self._active_pair = 0
            self._ready_buffers = 0
            self._prediction_count = 0
            self._batch_count = 0
            self._preprocess_samples.clear()
            self._inference_samples.clear()
            self._pipeline_samples.clear()
            self._batch_samples.clear()
            self._prediction_age_samples.clear()
            for samples in self._timing_samples.values():
                samples.clear()
            self._prediction_times.clear()
            self._started_monotonic = time.monotonic()
            self._latest_timestamp_ms = 0
            if not self._model_inflight:
                self._inflight = False

    def close(self, timeout: float = 5.0) -> bool:
        with self._lock:
            if self._closed:
                return not self._buffer_thread.is_alive() and not self._inference_thread.is_alive()
            self._closed = True
            self._buffers.clear()
            self._scheduler.reset()
        self._drain(self._frame_queue)
        self._drain(self._inference_queue)
        self._frame_queue.put_nowait(None)
        self._inference_queue.put_nowait(None)
        deadline = time.monotonic() + timeout
        self._buffer_thread.join(max(0.0, deadline - time.monotonic()))
        self._inference_thread.join(max(0.0, deadline - time.monotonic()))
        return not self._buffer_thread.is_alive() and not self._inference_thread.is_alive()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                idle = (
                    self._frame_queue.empty()
                    and self._inference_queue.empty()
                    and not self._inflight
                    and not self._model_inflight
                    and not self._buffer_inflight
                )
            if idle:
                return True
            time.sleep(0.01)
        return False

    @staticmethod
    def _drain(target: queue.Queue[Any]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def diagnostics(self) -> ActionRuntimeDiagnostics:
        with self._lock:
            scheduler = self._scheduler.snapshot()
            elapsed = max(time.monotonic() - self._started_monotonic, 1.0)
            return ActionRuntimeDiagnostics(
                active_single_proposals=self._active_single,
                active_pair_proposals=self._active_pair,
                dynamic_pairs=self._dynamic_pair_count,
                ready_action_buffers=self._ready_buffers,
                active_action_buffers=self._buffers.proposal_count,
                buffered_roi_frames=self._buffers.stored_frame_count,
                action_predictions_total=self._prediction_count,
                action_predictions_per_second=self._prediction_rate(),
                tsm_preprocess_ms_mean=self._mean(self._preprocess_samples),
                tsm_preprocess_ms_p95=self._p95(self._preprocess_samples),
                tsm_inference_ms_mean=self._mean(self._inference_samples),
                tsm_inference_ms_p95=self._p95(self._inference_samples),
                action_pipeline_ms_mean=self._mean(self._pipeline_samples),
                action_pipeline_ms_p95=self._p95(self._pipeline_samples),
                action_batch_size_mean=self._mean(self._batch_samples),
                action_batch_size_p95=self._p95(self._batch_samples),
                action_queue_depth=self._frame_queue.qsize() + self._inference_queue.qsize(),
                stale_action_requests_dropped=(
                    self._queue_drops + self._stale_drops + scheduler.stale_drops
                ),
                action_device=self._model.device,
                scheduler_ready_proposals=scheduler.ready_proposals,
                scheduler_in_flight_proposals=scheduler.in_flight_proposals,
                expired_ready_requests=scheduler.stale_drops,
                replaced_ready_requests=scheduler.replaced_ready_requests,
                action_batches_total=self._batch_count,
                single_predictions_per_second=scheduler.single_predictions / elapsed,
                pair_predictions_per_second=scheduler.pair_predictions / elapsed,
                single_prediction_interval_ms_mean=scheduler.single_interval_ms_mean,
                single_prediction_interval_ms_p95=scheduler.single_interval_ms_p95,
                single_prediction_interval_ms_max=scheduler.single_interval_ms_max,
                pair_prediction_interval_ms_mean=scheduler.pair_interval_ms_mean,
                pair_prediction_interval_ms_p95=scheduler.pair_interval_ms_p95,
                pair_prediction_interval_ms_max=scheduler.pair_interval_ms_max,
                action_prediction_age_ms_mean=self._mean(self._prediction_age_samples),
                action_prediction_age_ms_p95=self._p95(self._prediction_age_samples),
                timing_profiles={
                    name: self._summary(samples) for name, samples in self._timing_samples.items()
                },
            )

    @staticmethod
    def _mean(samples: deque[float]) -> float | None:
        return sum(samples) / len(samples) if samples else None

    @staticmethod
    def _p95(samples: deque[float]) -> float | None:
        return float(np.percentile(np.asarray(samples), 95)) if samples else None

    @staticmethod
    def _summary(samples: deque[float]) -> dict[str, float | None]:
        return {
            "mean": ActionRecognitionRuntime._mean(samples),
            "p50": float(np.percentile(np.asarray(samples), 50)) if samples else None,
            "p95": ActionRecognitionRuntime._p95(samples),
        }

    def _prediction_rate(self) -> float:
        if not self._prediction_times:
            return 0.0
        now = time.monotonic()
        while self._prediction_times and self._prediction_times[0] < now - 10.0:
            self._prediction_times.popleft()
        window = max(now - self._prediction_times[0], 1.0)
        return len(self._prediction_times) / window

    def _infer_loop(self) -> None:
        while True:
            batch = self._inference_queue.get()
            if batch is None:
                return
            with self._lock:
                self._model_inflight = True
            try:
                result = self._model.predict(batch.clips)
            except Exception as error:
                with self._lock:
                    current = not self._closed and batch.runtime_generation == self._generation
                    self._scheduler.fail(batch.clips)
                    self._model_inflight = False
                    self._inflight = False
                if current:
                    self._publish(
                        {
                            "type": "action_error",
                            "session_id": str(self.session_id),
                            "runtime_instance_id": str(self.runtime_instance_id),
                            "runtime_generation": batch.runtime_generation,
                            "timestamp_ms": batch.timestamp_ms,
                            "error": str(error),
                        }
                    )
                continue
            with self._lock:
                if self._closed or batch.runtime_generation != self._generation:
                    self._stale_drops += 1
                    self._scheduler.fail(batch.clips)
                    self._model_inflight = False
                    self._inflight = False
                    continue
                completed = time.monotonic()
                pipeline_ms = (completed - batch.submitted_monotonic) * 1000
                self._scheduler.complete(result.predictions)
                self._preprocess_samples.append(result.preprocessing_ms)
                self._inference_samples.append(result.inference_ms)
                self._pipeline_samples.append(pipeline_ms)
                self._batch_samples.append(float(len(batch.clips)))
                self._timing_samples["tensor_assembly_ms"].append(result.tensor_assembly_ms)
                self._timing_samples["input_resize_ms"].append(result.resize_ms)
                self._timing_samples["normalization_ms"].append(result.normalization_ms)
                self._timing_samples["h2d_ms"].append(result.h2d_ms)
                self._timing_samples["forward_ms"].append(result.forward_ms)
                self._timing_samples["postprocess_ms"].append(result.postprocess_ms)
                self._timing_samples["model_total_ms"].append(result.total_ms)
                self._prediction_count += len(result.predictions)
                self._batch_count += 1
                self._prediction_times.extend(completed for _ in result.predictions)
                self._prediction_age_samples.extend(
                    float(max(0, self._latest_timestamp_ms - prediction.timestamp_ms))
                    for prediction in result.predictions
                )
            self._publish(
                {
                    "type": "action_prediction",
                    "session_id": str(self.session_id),
                    "runtime_instance_id": str(self.runtime_instance_id),
                    "runtime_generation": batch.runtime_generation,
                    "timestamp_ms": batch.timestamp_ms,
                    "predictions": [item.as_dict(R3_CLASS_NAMES) for item in result.predictions],
                }
            )
            if self._on_predictions is not None:
                try:
                    self._on_predictions(result.predictions, batch.runtime_generation)
                except Exception as error:
                    self._publish(
                        {
                            "type": "action_error",
                            "session_id": str(self.session_id),
                            "runtime_instance_id": str(self.runtime_instance_id),
                            "runtime_generation": batch.runtime_generation,
                            "timestamp_ms": batch.timestamp_ms,
                            "error": f"Event aggregation/persistence failure: {error}",
                        }
                    )
            with self._lock:
                self._model_inflight = False
                self._inflight = False

    def _buffer_loop(self) -> None:
        while True:
            item = self._frame_queue.get()
            if item is None:
                return
            with self._lock:
                self._buffer_inflight = True
            try:
                self._process_frame(item)
            except Exception as error:
                with self._lock:
                    current = not self._closed and item.runtime_generation == self._generation
                if current:
                    self._publish(
                        {
                            "type": "action_error",
                            "session_id": str(self.session_id),
                            "runtime_instance_id": str(self.runtime_instance_id),
                            "runtime_generation": item.runtime_generation,
                            "timestamp_ms": item.timestamp_ms,
                            "error": f"Action ROI/buffer failure: {error}",
                        }
                    )
            finally:
                with self._lock:
                    self._buffer_inflight = False
