from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.ai.action_recognition.adapter import ActionModelRegistry
from app.ai.cheating_classifier.model import CheatModelRegistry
from app.ai.detector.yolo import PersonDetector
from app.ai.event_aggregation.types import AggregatedEvent
from app.ai.seat_identity.types import SeatIdentityContext
from app.monitoring.config import RuntimeProfile
from app.monitoring.publisher import LatestWebSocketPublisher, Subscriber
from app.monitoring.worker import VideoAnalysisWorker

logger = logging.getLogger(__name__)


class RuntimeState(StrEnum):
    INACTIVE = "INACTIVE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass(slots=True)
class RuntimeHandle:
    worker: VideoAnalysisWorker
    profile: RuntimeProfile
    publisher: LatestWebSocketPublisher
    runtime_instance_id: uuid.UUID
    state: RuntimeState = RuntimeState.INITIALIZING
    synchronizing: bool = True
    error: str | None = None
    latest_diagnostics: dict[str, Any] | None = None
    latest_tracking_seq: int = 0


class MonitoringRuntimeManager:
    def __init__(
        self,
        *,
        worker_factory: Callable[..., VideoAnalysisWorker] = VideoAnalysisWorker,
        terminal_callback: Callable[[uuid.UUID, RuntimeState, str | None], None] | None = None,
        event_callback: Callable[[AggregatedEvent], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._handles: dict[uuid.UUID, RuntimeHandle] = {}
        self._worker_factory = worker_factory
        self._terminal_callback = terminal_callback
        self._event_callback = event_callback
        self._action_models = ActionModelRegistry()
        self._cheat_models = CheatModelRegistry()

    def preload(self, profile: RuntimeProfile) -> None:
        """Load (and compile) the X3D classifier before the first session starts."""
        if profile.cheating_classifier.enabled:
            self._cheat_models.get(profile.cheating_classifier, profile.device).ensure_loaded()

    def start(
        self,
        session_id: uuid.UUID,
        video_path: Path,
        profile: RuntimeProfile,
        timestamp_ms: int,
        loop_source: bool = False,
        seat_identity_context: SeatIdentityContext | None = None,
    ) -> dict[str, Any]:
        PersonDetector.validate_environment(profile.detector)
        with self._lock:
            existing = self._handles.get(session_id)
            if existing and existing.state in {
                RuntimeState.INITIALIZING,
                RuntimeState.RUNNING,
                RuntimeState.PAUSED,
            }:
                raise RuntimeError("Monitoring runtime đã hoạt động")
            publisher = LatestWebSocketPublisher()
            runtime_instance_id = uuid.uuid4()
            worker_arguments: dict[str, Any] = dict(
                session_id=session_id,
                video_path=video_path,
                profile=profile,
                start_timestamp_ms=timestamp_ms,
                loop_source=loop_source,
                publish=lambda message: self._publish(session_id, runtime_instance_id, message),
                on_ready=lambda: self._mark_ready(session_id, runtime_instance_id),
                on_complete=lambda: self._terminal(
                    session_id, runtime_instance_id, RuntimeState.COMPLETED, None
                ),
                on_error=lambda error: self._terminal(
                    session_id, runtime_instance_id, RuntimeState.ERROR, error
                ),
                runtime_instance_id=runtime_instance_id,
                seat_identity_context=seat_identity_context,
            )
            if profile.action_recognition.enabled:
                action_model = self._action_models.get(
                    profile.action_recognition,
                    profile.device,
                )
                action_model.ensure_loaded()
                worker_arguments["action_model"] = action_model
            if profile.event_detection.enabled:
                if self._event_callback is None:
                    raise RuntimeError("AI Event persistence is not configured")
                worker_arguments["event_callback"] = self._event_callback
            if profile.cheating_classifier.enabled:
                # loaded (and compiled: about a minute after a restart) by the worker thread
                # while the session is INITIALIZING, so this request returns at once
                cheat_model = self._cheat_models.get(profile.cheating_classifier, profile.device)
                worker_arguments["cheat_model"] = cheat_model
                if self._event_callback is not None:
                    worker_arguments["event_callback"] = self._event_callback
            worker = self._worker_factory(**worker_arguments)
            handle = RuntimeHandle(
                worker=worker,
                profile=profile,
                publisher=publisher,
                runtime_instance_id=runtime_instance_id,
            )
            self._handles[session_id] = handle
            worker.start()
            self._publish_state(session_id)
            return self.status(session_id)

    def pause(self, session_id: uuid.UUID, timestamp_ms: int | None = None) -> dict[str, Any]:
        handle = self._require_state(session_id, {RuntimeState.RUNNING})
        handle.worker.pause(timestamp_ms)
        self._set_state(session_id, RuntimeState.PAUSED)
        return self.status(session_id)

    def resume(self, session_id: uuid.UUID, timestamp_ms: int | None = None) -> dict[str, Any]:
        handle = self._require_state(session_id, {RuntimeState.PAUSED})
        handle.worker.resume(timestamp_ms)
        self._set_state(
            session_id,
            RuntimeState.RUNNING,
            synchronizing=timestamp_ms is not None,
        )
        return self.status(session_id)

    def seek(self, session_id: uuid.UUID, timestamp_ms: int) -> dict[str, Any]:
        handle = self._require_state(session_id, {RuntimeState.RUNNING, RuntimeState.PAUSED})
        handle.worker.seek(timestamp_ms)
        self._publish_state(session_id, synchronizing=True)
        return self.status(session_id)

    def stop(self, session_id: uuid.UUID) -> dict[str, Any]:
        handle = self._require_state(
            session_id,
            {RuntimeState.INITIALIZING, RuntimeState.RUNNING, RuntimeState.PAUSED},
        )
        if handle.worker.stop() is False:
            raise RuntimeError("Monitoring runtime không thể dừng sạch trong thời gian cho phép")
        self._set_state(session_id, RuntimeState.COMPLETED, synchronizing=False)
        return self.status(session_id)

    def status(self, session_id: uuid.UUID) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None:
                return {
                    "session_id": str(session_id),
                    "state": RuntimeState.INACTIVE.value,
                    "profile": None,
                    "error": None,
                    "subscriber_count": 0,
                    "queue_size": 0,
                    "dropped_analysis_frames": 0,
                    "diagnostics": None,
                    "runtime_instance_id": None,
                    "runtime_generation": None,
                    "worker_instance_id": None,
                    "tracker_instance_id": None,
                    "tracking_seq": 0,
                }
            return {
                "session_id": str(session_id),
                "state": handle.state.value,
                "profile": handle.profile.profile,
                "error": handle.error,
                "subscriber_count": handle.publisher.subscriber_count,
                "queue_size": handle.worker.queue_size,
                "dropped_analysis_frames": handle.worker.dropped_analysis_frames,
                "diagnostics": handle.latest_diagnostics,
                "runtime_instance_id": str(handle.runtime_instance_id),
                "runtime_generation": getattr(handle.worker, "runtime_generation", 0),
                "worker_instance_id": str(getattr(handle.worker, "worker_instance_id", "")) or None,
                "tracker_instance_id": (
                    handle.latest_diagnostics.get("tracker_instance_id")
                    if handle.latest_diagnostics
                    else None
                ),
                "tracking_seq": handle.latest_tracking_seq,
            }

    def subscribe(self, session_id: uuid.UUID) -> Subscriber:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None:
                raise RuntimeError("Monitoring runtime chưa được khởi tạo")
            subscriber = handle.publisher.subscribe()
        self._publish_state(session_id)
        return subscriber

    def unsubscribe(self, session_id: uuid.UUID, subscriber_id: uuid.UUID) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is not None:
                handle.publisher.unsubscribe(subscriber_id)

    def shutdown(self) -> None:
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            if handle.worker.stop() is False:
                logger.error("Monitoring worker did not stop cleanly during application shutdown")

    def _require_state(
        self,
        session_id: uuid.UUID,
        allowed: set[RuntimeState],
    ) -> RuntimeHandle:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None or handle.state not in allowed:
                current = handle.state.value if handle else RuntimeState.INACTIVE.value
                raise RuntimeError(f"Trạng thái runtime không hợp lệ: {current}")
            return handle

    def _publish(
        self,
        session_id: uuid.UUID,
        runtime_instance_id: uuid.UUID,
        message: dict[str, Any],
    ) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None or handle.runtime_instance_id != runtime_instance_id:
                return
            message_generation = message.get("runtime_generation")
            if message_generation is not None and message_generation != getattr(
                handle.worker, "runtime_generation", 0
            ):
                return
            if message.get("type") == "diagnostics":
                handle.latest_diagnostics = message
            elif message.get("type") == "tracking":
                handle.synchronizing = False
                handle.latest_tracking_seq = int(message.get("tracking_seq", 0))
            publisher = handle.publisher
        publisher.publish(message)

    def _set_state(
        self,
        session_id: uuid.UUID,
        state: RuntimeState,
        *,
        synchronizing: bool | None = None,
    ) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None:
                return
            handle.state = state
            if synchronizing is not None:
                handle.synchronizing = synchronizing
        self._publish_state(session_id)

    def _mark_ready(self, session_id: uuid.UUID, runtime_instance_id: uuid.UUID) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if (
                handle is None
                or handle.runtime_instance_id != runtime_instance_id
                or handle.state != RuntimeState.INITIALIZING
            ):
                return
            handle.state = RuntimeState.RUNNING
        self._publish_state(session_id)

    def _terminal(
        self,
        session_id: uuid.UUID,
        runtime_instance_id: uuid.UUID,
        state: RuntimeState,
        error: str | None,
    ) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if (
                handle is None
                or handle.runtime_instance_id != runtime_instance_id
                or handle.state in {RuntimeState.COMPLETED, RuntimeState.ERROR}
            ):
                return
            handle.state = state
            handle.synchronizing = False
            handle.error = error
        self._publish_state(session_id)
        if self._terminal_callback is not None:
            self._terminal_callback(session_id, state, error)

    def _publish_state(self, session_id: uuid.UUID, *, synchronizing: bool = False) -> None:
        with self._lock:
            handle = self._handles.get(session_id)
            if handle is None:
                return
            if synchronizing:
                handle.synchronizing = True
            message = {
                "type": "state",
                "session_id": str(session_id),
                "state": handle.state.value,
                "synchronizing": handle.synchronizing,
                "error": handle.error,
                "runtime_instance_id": str(handle.runtime_instance_id),
                "runtime_generation": getattr(handle.worker, "runtime_generation", 0),
                "worker_instance_id": str(getattr(handle.worker, "worker_instance_id", "")) or None,
                "tracker_instance_id": (
                    handle.latest_diagnostics.get("tracker_instance_id")
                    if handle.latest_diagnostics
                    else None
                ),
                "tracking_seq": handle.latest_tracking_seq,
            }
            publisher = handle.publisher
        publisher.publish(message)
