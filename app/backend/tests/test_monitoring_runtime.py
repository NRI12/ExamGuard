from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.ai.domain import Detection, RuntimeDiagnostics, TrackedObject
from app.ai.seat_identity.assignment import SeatAssignmentEngine
from app.monitoring.buffer import LatestValueBuffer
from app.monitoring.clock import AnalysisClock
from app.monitoring.config import (
    AnalysisConfig,
    ByteTrackConfig,
    DetectorConfig,
    DiagnosticsConfig,
    RuntimeProfile,
)
from app.monitoring.decoder import FramePacket
from app.monitoring.manager import MonitoringRuntimeManager, RuntimeState
from app.monitoring.publisher import LatestWebSocketPublisher
from app.monitoring.worker import VideoAnalysisWorker


def profile() -> RuntimeProfile:
    return RuntimeProfile(
        profile="gtx1650",
        device="cuda:0",
        precision="fp16",
        analysis=AnalysisConfig(
            target_fps=12.5,
            minimum_fps=8,
            queue_size=1,
            drop_stale_frames=True,
        ),
        detector=DetectorConfig(
            model=Path("/models/detection/yolo11n.pt"),
            imgsz=640,
            conf=0.1,
            iou=0.7,
            classes=[0],
            max_det=32,
            half=True,
            device="cuda:0",
        ),
        tracker=ByteTrackConfig(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=20,
            match_thresh=0.8,
            fuse_score=True,
        ),
        diagnostics=DiagnosticsConfig(publish_hz=2),
    )


class FakeDecoder:
    source_fps = 25.0
    source_width = 100
    source_height = 80

    def __init__(self, _: Path) -> None:
        self.frame_id = 0
        self.released = False

    def seek(self, timestamp_ms: int) -> None:
        self.frame_id = round(timestamp_ms / 40)

    def read_for_timestamp(self, target_ms: int, generation: int) -> tuple[FramePacket, int]:
        next_frame = max(self.frame_id + 1, round(target_ms / 40))
        dropped = max(0, next_frame - self.frame_id - 1)
        self.frame_id = next_frame
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[0, 0, 0] = next_frame % 256
        return FramePacket(
            frame=frame,
            frame_id=next_frame,
            timestamp_ms=next_frame * 40,
            source_width=100,
            source_height=80,
            generation=generation,
        ), dropped

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def __init__(self, _: object, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.frame_tokens: list[int] = []

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        self.frame_tokens.append(int(frame[0, 0, 0]))
        time.sleep(self.delay)
        return [Detection((10, 10, 30, 70), 0.9, 0)]


class FakeTracker:
    def __init__(self, _: object) -> None:
        self.instance_id = uuid.uuid4()
        self.reset_count = 0
        self.timestamps: list[int] = []

    def update(
        self,
        _: list[Detection],
        __: tuple[int, int],
        timestamp_ms: int,
    ) -> list[TrackedObject]:
        self.timestamps.append(timestamp_ms)
        return [TrackedObject(1, (10, 10, 30, 70), 0.9)]

    def reset(self) -> None:
        self.reset_count += 1


class BlockingDetector(FakeDetector):
    def __init__(self, config: object, entered: threading.Event, release: threading.Event) -> None:
        super().__init__(config)
        self.entered = entered
        self.release = release

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.calls == 0:
            self.entered.set()
            assert self.release.wait(1)
        return super().detect(frame)


class FiniteDecoder(FakeDecoder):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.finished = False

    def read_for_timestamp(
        self,
        target_ms: int,
        generation: int,
    ) -> tuple[FramePacket | None, int]:
        if self.finished:
            return None, 0
        self.finished = True
        return super().read_for_timestamp(target_ms, generation)


class OutOfOrderDecoder(FakeDecoder):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._timestamps = iter((80, 40, 120))

    def read_for_timestamp(
        self,
        target_ms: int,
        generation: int,
    ) -> tuple[FramePacket | None, int]:
        try:
            timestamp_ms = next(self._timestamps)
        except StopIteration:
            return None, 0
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        return (
            FramePacket(
                frame=frame,
                frame_id=timestamp_ms,
                timestamp_ms=timestamp_ms,
                source_width=100,
                source_height=80,
                generation=generation,
            ),
            0,
        )


def test_latest_buffer_never_grows_and_drops_stale_values() -> None:
    buffer: LatestValueBuffer[int] = LatestValueBuffer()
    buffer.put(1)
    buffer.put(2)
    buffer.put(3)
    assert buffer.size == 1
    assert buffer.dropped == 2
    assert buffer.get() == 3
    assert buffer.size == 0


def test_analysis_clock_pause_resume_and_seek() -> None:
    clock = AnalysisClock(1000)
    clock.pause(1250)
    time.sleep(0.01)
    assert clock.timestamp_ms() == 1250
    clock.seek(5000)
    assert clock.timestamp_ms() == 5000
    clock.resume(5000)
    time.sleep(0.01)
    assert clock.timestamp_ms() >= 5005


def test_websocket_publisher_keeps_only_latest_message_for_slow_client() -> None:
    async def scenario() -> None:
        publisher = LatestWebSocketPublisher()
        subscriber = publisher.subscribe()
        publisher.publish({"sequence": 1})
        publisher.publish({"sequence": 2})
        await asyncio.sleep(0)
        assert await subscriber.queue.get() == {"sequence": 2}

    asyncio.run(scenario())


def test_websocket_publisher_preserves_low_frequency_action_message() -> None:
    async def scenario() -> None:
        publisher = LatestWebSocketPublisher()
        subscriber = publisher.subscribe()
        publisher.publish({"type": "tracking", "sequence": 1})
        publisher.publish({"type": "action_prediction", "sequence": 2})
        publisher.publish({"type": "tracking", "sequence": 3})
        await asyncio.sleep(0)
        first = await subscriber.queue.get()
        second = await subscriber.queue.get()
        assert {first["type"], second["type"]} == {"tracking", "action_prediction"}
        assert next(item for item in (first, second) if item["type"] == "tracking")["sequence"] == 3

    asyncio.run(scenario())


def test_websocket_publisher_removes_subscriber_with_closed_loop() -> None:
    publisher = LatestWebSocketPublisher()
    loop = asyncio.new_event_loop()

    async def subscribe() -> None:
        publisher.subscribe()

    loop.run_until_complete(subscribe())
    loop.close()
    publisher.publish({"type": "tracking"})
    assert publisher.subscriber_count == 0


def test_runtime_diagnostics_message_is_json_serializable() -> None:
    session_id = uuid.uuid4()
    runtime_instance_id = uuid.uuid4()
    worker_instance_id = uuid.uuid4()
    tracker_instance_id = uuid.uuid4()
    message = RuntimeDiagnostics(
        session_id=session_id,
        runtime_instance_id=runtime_instance_id,
        runtime_generation=2,
        worker_instance_id=worker_instance_id,
        tracker_instance_id=tracker_instance_id,
        tracking_seq=3,
        latest_frame_id=4,
        latest_timestamp_ms=160,
        raw_detection_count=1,
        active_track_count=1,
        source_fps=25.0,
        target_analysis_fps=12.5,
        analysis_fps=12.0,
        detector_ms=18.0,
        tracker_ms=1.0,
        pipeline_ms=19.0,
        seat_assignment_ms=0.4,
        analysis_lag_ms=40.0,
        gpu_util_pct=50.0,
        vram_used_mb=512.0,
        cpu_util_pct=25.0,
        ram_used_mb=1024.0,
        dropped_analysis_frames=0,
        queue_size=0,
        assigned_tracks=1,
        tentative_tracks=0,
        unassigned_tracks=0,
        occupied_seats=1,
        grace_seats=0,
        empty_seats=0,
        seat_switches=0,
        identity_recoveries=0,
        profile="gtx1650",
    ).as_message()

    json.dumps(message)
    assert message["session_id"] == str(session_id)
    assert message["runtime_instance_id"] == str(runtime_instance_id)
    assert message["worker_instance_id"] == str(worker_instance_id)
    assert message["tracker_instance_id"] == str(tracker_instance_id)


def test_worker_cadence_pause_seek_latest_drop_and_cleanup() -> None:
    messages: list[dict[str, Any]] = []
    ready = threading.Event()
    detectors: list[FakeDetector] = []
    trackers: list[FakeTracker] = []
    decoders: list[FakeDecoder] = []
    seat_engines: list[SeatAssignmentEngine] = []

    def tracker_factory(config: object) -> FakeTracker:
        tracker = FakeTracker(config)
        trackers.append(tracker)
        return tracker

    def decoder_factory(path: Path) -> FakeDecoder:
        decoder = FakeDecoder(path)
        decoders.append(decoder)
        return decoder

    def detector_factory(config: object) -> FakeDetector:
        detector = FakeDetector(config, delay=0.12)
        detectors.append(detector)
        return detector

    def seat_assignment_factory(context: object, config: object) -> SeatAssignmentEngine:
        engine = SeatAssignmentEngine(context, config)  # type: ignore[arg-type]
        original_reset = engine.reset
        engine.reset_count = 0  # type: ignore[attr-defined]

        def reset() -> None:
            engine.reset_count += 1  # type: ignore[attr-defined]
            original_reset()

        engine.reset = reset  # type: ignore[method-assign]
        seat_engines.append(engine)
        return engine

    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=Path("video.mp4"),
        profile=profile(),
        start_timestamp_ms=0,
        publish=messages.append,
        on_ready=ready.set,
        on_complete=lambda: None,
        on_error=lambda error: (_ for _ in ()).throw(AssertionError(error)),
        detector_factory=detector_factory,
        tracker_factory=tracker_factory,
        decoder_factory=decoder_factory,
        seat_assignment_factory=seat_assignment_factory,
    )
    worker.start()
    assert ready.wait(0.5)
    time.sleep(0.42)
    tracking_before_pause = len([item for item in messages if item["type"] == "tracking"])
    assert 2 <= tracking_before_pause <= 5
    assert worker.queue_size <= 1
    assert worker.dropped_analysis_frames > 0
    worker.pause(1000)
    time.sleep(0.18)
    tracking_while_paused = len([item for item in messages if item["type"] == "tracking"])
    assert tracking_while_paused <= tracking_before_pause + 1
    worker.resume(1000)
    time.sleep(0.12)
    assert trackers[0].reset_count == 0
    worker.pause(1120)
    worker.seek(5000)
    worker.resume(5000)
    time.sleep(0.22)
    assert worker.stop() is True
    assert worker.alive is False
    assert len(detectors) == 1
    assert len(trackers) == 1
    published_count = len([item for item in messages if item["type"] == "tracking"])
    assert detectors[0].calls >= published_count
    assert len(detectors[0].frame_tokens) == len(set(detectors[0].frame_tokens))
    assert decoders[0].released is True
    assert trackers[0].reset_count >= 1
    assert seat_engines[0].reset_count >= 1  # type: ignore[attr-defined]
    timestamps = [item["timestamp_ms"] for item in messages if item["type"] == "tracking"]
    assert any(timestamp >= 5000 for timestamp in timestamps)
    tracking = [item for item in messages if item["type"] == "tracking"]
    assert tracking
    assert all(item["seats"] == [] for item in tracking)
    assert all(track["actor_id"].startswith("A") for item in tracking for track in item["tracks"])
    assert all(
        track["identity"]["seat_id"] is None for item in tracking for track in item["tracks"]
    )
    assert len({item["runtime_instance_id"] for item in tracking}) == 1
    assert len({item["tracker_instance_id"] for item in tracking}) == 1
    post_seek = [item for item in tracking if item["runtime_generation"] == 1]
    assert post_seek and post_seek[0]["tracking_seq"] == 1
    assert post_seek[0]["tracks"][0]["actor_id"] == "A0001"


def test_worker_discards_inflight_result_from_before_seek() -> None:
    messages: list[dict[str, Any]] = []
    entered = threading.Event()
    release = threading.Event()
    ready = threading.Event()
    errors: list[str] = []
    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=Path("video.mp4"),
        profile=profile(),
        start_timestamp_ms=0,
        publish=messages.append,
        on_ready=ready.set,
        on_complete=lambda: None,
        on_error=errors.append,
        detector_factory=lambda config: BlockingDetector(config, entered, release),
        tracker_factory=FakeTracker,
        decoder_factory=FakeDecoder,
    )
    worker.start()
    assert ready.wait(0.5)
    assert entered.wait(0.5)
    worker.seek(5000)
    release.set()
    time.sleep(0.25)
    assert worker.stop() is True
    assert errors == []
    tracking = [item for item in messages if item["type"] == "tracking"]
    assert tracking
    assert all(item["timestamp_ms"] >= 5000 for item in tracking)


def test_worker_drops_non_increasing_tracker_input_before_inference() -> None:
    messages: list[dict[str, Any]] = []
    completed = threading.Event()
    detectors: list[FakeDetector] = []
    trackers: list[FakeTracker] = []

    def detector_factory(config: object) -> FakeDetector:
        detector = FakeDetector(config)
        detectors.append(detector)
        return detector

    def tracker_factory(config: object) -> FakeTracker:
        tracker = FakeTracker(config)
        trackers.append(tracker)
        return tracker

    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=Path("video.mp4"),
        profile=profile(),
        start_timestamp_ms=0,
        publish=messages.append,
        on_ready=lambda: None,
        on_complete=completed.set,
        on_error=lambda error: (_ for _ in ()).throw(AssertionError(error)),
        detector_factory=detector_factory,
        tracker_factory=tracker_factory,
        decoder_factory=OutOfOrderDecoder,
    )
    worker.start()
    assert completed.wait(1)
    assert worker.stop() is True
    assert detectors[0].calls == 2
    assert trackers[0].timestamps == [80, 120]
    assert worker.dropped_analysis_frames >= 1
    tracking = [item for item in messages if item["type"] == "tracking"]
    assert [item["timestamp_ms"] for item in tracking] == [80, 120]


def test_worker_completes_after_analyzing_final_frame_and_releasing_decoder() -> None:
    messages: list[dict[str, Any]] = []
    completed = threading.Event()
    decoders: list[FiniteDecoder] = []

    def decoder_factory(path: Path) -> FiniteDecoder:
        decoder = FiniteDecoder(path)
        decoders.append(decoder)
        return decoder

    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=Path("video.mp4"),
        profile=profile(),
        start_timestamp_ms=0,
        publish=messages.append,
        on_ready=lambda: None,
        on_complete=completed.set,
        on_error=lambda error: (_ for _ in ()).throw(AssertionError(error)),
        detector_factory=FakeDetector,
        tracker_factory=FakeTracker,
        decoder_factory=decoder_factory,
    )
    worker.start()
    assert completed.wait(1)
    for _ in range(20):
        if not worker.alive:
            break
        time.sleep(0.01)
    assert worker.alive is False
    assert decoders[0].released is True
    assert len([item for item in messages if item["type"] == "tracking"]) == 1


def test_terminal_state_is_not_overwritten_by_late_worker_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.monitoring.manager.PersonDetector.validate_environment",
        lambda _: None,
    )
    terminal_states: list[RuntimeState] = []

    class CompletingBeforeReadyWorker:
        queue_size = 0
        dropped_analysis_frames = 0

        def __init__(self, **kwargs: Any) -> None:
            self.on_ready = kwargs["on_ready"]
            self.on_complete = kwargs["on_complete"]

        def start(self) -> None:
            self.on_complete()
            self.on_ready()

        def stop(self, timeout: float = 5.0) -> bool:
            return True

    manager = MonitoringRuntimeManager(
        worker_factory=CompletingBeforeReadyWorker,  # type: ignore[arg-type]
        terminal_callback=lambda _, state, __: terminal_states.append(state),
    )
    session_id = uuid.uuid4()
    result = manager.start(session_id, Path("video.mp4"), profile(), 0)
    assert result["state"] == RuntimeState.COMPLETED.value
    assert terminal_states == [RuntimeState.COMPLETED]


def test_manager_rejects_duplicate_start_and_ignores_replaced_worker_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.monitoring.manager.PersonDetector.validate_environment",
        lambda _: None,
    )
    workers: list[Any] = []

    class ControlledWorker:
        queue_size = 0
        dropped_analysis_frames = 0
        runtime_generation = 0
        tracking_seq = 0

        def __init__(self, **kwargs: Any) -> None:
            self.worker_instance_id = uuid.uuid4()
            self.runtime_instance_id = kwargs["runtime_instance_id"]
            self.publish = kwargs["publish"]
            self.on_ready = kwargs["on_ready"]
            self.on_complete = kwargs["on_complete"]
            self.on_error = kwargs["on_error"]
            workers.append(self)

        def start(self) -> None:
            self.on_ready()

        def stop(self, timeout: float = 5.0) -> bool:
            return True

    manager = MonitoringRuntimeManager(worker_factory=ControlledWorker)  # type: ignore[arg-type]
    session_id = uuid.uuid4()
    assert manager.start(session_id, Path("video.mp4"), profile(), 0)["state"] == "RUNNING"
    with pytest.raises(RuntimeError, match="đã hoạt động"):
        manager.start(session_id, Path("video.mp4"), profile(), 0)

    assert manager.stop(session_id)["state"] == "COMPLETED"
    first = workers[0]
    second_status = manager.start(session_id, Path("video.mp4"), profile(), 0)
    assert second_status["state"] == "RUNNING"
    assert second_status["runtime_instance_id"] != str(first.runtime_instance_id)

    first.publish(
        {
            "type": "tracking",
            "runtime_generation": 0,
            "tracking_seq": 999,
        }
    )
    first.on_error("late failure")
    current = manager.status(session_id)
    assert current["state"] == "RUNNING"
    assert current["tracking_seq"] == 0


def test_camera_source_loops_with_controlled_generation_reset() -> None:
    messages: list[dict[str, Any]] = []
    reached_second_generation = threading.Event()
    completed = threading.Event()
    errors: list[str] = []
    decoders: list[Any] = []
    trackers: list[FakeTracker] = []

    class OneFrameLoopDecoder:
        source_fps = 25.0
        source_width = 100
        source_height = 80

        def __init__(self, _: Path) -> None:
            self.emitted = False
            self.seek_count = 0
            self.released = False
            decoders.append(self)

        def seek(self, _: int) -> None:
            self.emitted = False
            self.seek_count += 1

        def read_for_timestamp(
            self,
            _: int,
            generation: int,
        ) -> tuple[FramePacket | None, int]:
            if self.emitted:
                return None, 0
            self.emitted = True
            return (
                FramePacket(
                    frame=np.zeros((80, 100, 3), dtype=np.uint8),
                    frame_id=0,
                    timestamp_ms=0,
                    source_width=100,
                    source_height=80,
                    generation=generation,
                ),
                0,
            )

        def release(self) -> None:
            self.released = True

    def publish(message: dict[str, Any]) -> None:
        messages.append(message)
        if message.get("type") == "tracking" and message.get("runtime_generation", 0) >= 1:
            reached_second_generation.set()

    def tracker_factory(config: object) -> FakeTracker:
        tracker = FakeTracker(config)
        trackers.append(tracker)
        return tracker

    worker = VideoAnalysisWorker(
        session_id=uuid.uuid4(),
        video_path=Path("camera-source.mp4"),
        profile=profile(),
        start_timestamp_ms=0,
        loop_source=True,
        publish=publish,
        on_ready=lambda: None,
        on_complete=completed.set,
        on_error=errors.append,
        detector_factory=FakeDetector,
        tracker_factory=tracker_factory,
        decoder_factory=OneFrameLoopDecoder,
    )
    started = time.monotonic()
    worker.start()
    assert reached_second_generation.wait(1)
    elapsed = time.monotonic() - started
    assert worker.stop() is True

    tracking = [item for item in messages if item["type"] == "tracking"]
    assert {item["runtime_generation"] for item in tracking} >= {0, 1}
    assert all(item["timestamp_ms"] == 0 for item in tracking[:2])
    assert elapsed >= 0.05
    assert completed.is_set() is False
    assert errors == []
    assert decoders[0].seek_count >= 2
    assert decoders[0].released is True
    assert trackers[0].reset_count >= 1
