"""Per-actor cheating classification on the live tracking stream (cheating_detection_v1 app).

  tracking frame (any analysis FPS >= 5)
    --first frame of every 200 ms bucket--> RGB frame + mapped box of every Stable Actor
    --every 1000 ms, 16 samples (3.2 s)--> one square crop per actor --> X3D-L (own thread)
    --> softmax per actor --> mean of the last `smooth_windows` windows
    --> cheat score = 1 - P(normal)
    --> hysteresis alert (start / keep threshold), type = most frequent cheating class of the alert
    --> alert lasting >= min_windows windows --> AggregatedEvent for the actor's SessionCandidate

The tracking loop never waits for the classifier: a new window replaces a pending one (capacity 1).
Everything uses source timestamps, so pause does not advance the rule and a seek resets it.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.ai.cheating_classifier.model import CheatClassifierModel, map_box, square_crop_box
from app.ai.domain import Track
from app.ai.event_aggregation.types import AggregatedEvent, EventBehavior
from app.monitoring.config import CheatingClassifierConfig

logger = logging.getLogger(__name__)

BEHAVIOR_BY_CLASS = {
    "looking": EventBehavior.SUSPICIOUS_LOOKING,
    "interaction": EventBehavior.COMMUNICATING,
    "phone_cheatsheet": EventBehavior.USING_PHONE_CHEAT_SHEET,
    "abnormal": EventBehavior.SUSPICIOUS_LOOKING,
}


@dataclass(slots=True)
class _Alert:
    start_ms: int
    end_ms: int
    peak_ms: int
    peak_score: float = 0.0
    score_sum: float = 0.0
    windows: int = 0
    labels: collections.Counter[str] = field(default_factory=collections.Counter)
    candidates: collections.Counter[uuid.UUID] = field(default_factory=collections.Counter)


@dataclass(slots=True)
class _Actor:
    actor_id: str
    track_id: int
    boxes: dict[int, np.ndarray] = field(default_factory=dict)
    last_sample: int = -1
    last_seen_ms: int = 0
    session_candidate_id: uuid.UUID | None = None
    seat_code: str | None = None
    probs: collections.deque[np.ndarray] = field(default_factory=collections.deque)
    alert: _Alert | None = None
    state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _Job:
    generation: int
    timestamp_ms: int
    actor_ids: tuple[str, ...]
    crops: tuple[tuple[tuple[int, int, int, int], tuple[int, ...]], ...]
    frames: dict[int, np.ndarray]


class CheatingClassifierRuntime:
    def __init__(
        self,
        *,
        session_id: uuid.UUID,
        runtime_instance_id: uuid.UUID,
        config: CheatingClassifierConfig,
        model: CheatClassifierModel,
        publish: Callable[[dict[str, Any]], None],
        emit_event: Callable[[AggregatedEvent], None] | None = None,
    ) -> None:
        model.ensure_loaded()
        self.session_id = session_id
        self.runtime_instance_id = runtime_instance_id
        self.config = config
        self._model = model
        self._publish = publish
        self._emit_event = emit_event
        self._classes = model.classes
        self._n_frames = model.num_frames
        self._infer_every = max(1, round(config.infer_interval_ms / config.sample_interval_ms))
        self._min_boxes = int(np.ceil(config.min_coverage * self._n_frames))
        self._lock = threading.Lock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=1)
        self._generation = 0
        self._frames: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self._actors: dict[str, _Actor] = {}
        self._sample = -1
        self._closed = False
        self._pending = 0
        self._counters = collections.Counter[str]()
        self._inference_ms: collections.deque[float] = collections.deque(maxlen=64)
        self._thread = threading.Thread(
            target=self._infer_loop, name=f"cheat-classifier-{session_id}", daemon=True
        )
        self._thread.start()

    # ---------- tracking thread ----------
    def update(
        self,
        frame_bgr: np.ndarray,
        tracks: tuple[Track, ...],
        timestamp_ms: int,
        generation: int,
    ) -> None:
        sample = timestamp_ms // self.config.sample_interval_ms
        if generation != self._generation:
            self.reset(generation)
        with self._lock:
            if generation != self._generation or sample == self._sample or self._closed:
                return
            self._sample = sample
            height, width = frame_bgr.shape[:2]
            for track in tracks:
                actor_id = track.actor_id or f"track-{track.track_id}"
                actor = self._actors.get(actor_id)
                if actor is None:
                    actor = self._actors[actor_id] = _Actor(actor_id, track.track_id)
                x1, y1, x2, y2 = track.bbox_norm
                box = (x1 * width, y1 * height, x2 * width, y2 * height)
                actor.boxes[sample] = map_box(box, width, height, self.config)
                actor.track_id = track.track_id
                actor.last_sample = sample
                actor.last_seen_ms = timestamp_ms
                if track.identity.session_candidate_id is not None:
                    actor.session_candidate_id = track.identity.session_candidate_id
                    actor.seat_code = track.identity.seat_code
            self._frames[sample] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            while len(self._frames) > self._n_frames:
                self._frames.popitem(last=False)
            window = list(self._frames)
            job = None
            if sample % self._infer_every == 0 and len(window) == self._n_frames:
                job = self._make_job(window, width, height, timestamp_ms)
            ended = self._expire(sample, timestamp_ms)
        for event in ended:
            self._emit(event)
        if job is not None:
            with self._lock:
                self._pending += 1
            try:
                self._jobs.put_nowait(job)
            except queue.Full:  # classifier still busy: the newest window replaces the pending one
                try:
                    self._jobs.get_nowait()
                    with self._lock:
                        self._pending -= 1
                        self._counters["windows_replaced"] += 1
                except queue.Empty:
                    pass
                self._jobs.put_nowait(job)

    def idle(self) -> bool:
        """No window waiting for or inside the classifier."""
        with self._lock:
            return self._pending == 0

    def _make_job(
        self, window: list[int], width: int, height: int, timestamp_ms: int
    ) -> _Job | None:
        actor_ids, crops = [], []
        for actor in self._actors.values():
            have = [s for s in window if s in actor.boxes]
            if len(have) < self._min_boxes:
                continue
            boxes = [actor.boxes[min(have, key=lambda k, s=s: abs(k - s))] for s in window]
            square = square_crop_box(boxes, width, height, self.config.crop_pad)
            crops.append((square, tuple(window)))
            actor_ids.append(actor.actor_id)
        if not actor_ids:
            return None
        return _Job(
            generation=self._generation,
            timestamp_ms=timestamp_ms,
            actor_ids=tuple(actor_ids),
            crops=tuple(crops),
            frames={s: self._frames[s] for s in window},
        )

    def _expire(self, sample: int, timestamp_ms: int) -> list[AggregatedEvent]:
        ended = []
        gap = self.config.max_actor_gap_ms // self.config.sample_interval_ms
        for actor_id in [k for k, a in self._actors.items() if sample - a.last_sample > gap]:
            event = self._close_alert(self._actors.pop(actor_id))
            if event is not None:
                ended.append(event)
        for actor in self._actors.values():  # keep only the boxes a window can still use
            for key in [k for k in actor.boxes if k < sample - self._n_frames]:
                if key != actor.last_sample:
                    del actor.boxes[key]
        return ended

    def reset(self, generation: int) -> None:
        """Seek / loop: open alerts are finalised, buffers and smoothing start again."""
        with self._lock:
            ended = [e for a in self._actors.values() if (e := self._close_alert(a)) is not None]
            self._generation = generation
            self._frames.clear()
            self._actors.clear()
            self._sample = -1
        for event in ended:
            self._emit(event)

    def close(self, timeout: float = 5.0) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            ended = [e for a in self._actors.values() if (e := self._close_alert(a)) is not None]
            self._actors.clear()
        for event in ended:
            self._emit(event)
        try:
            self._jobs.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # ---------- classifier thread ----------
    def _infer_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                started = time.perf_counter()
                size = self.config.clip_size
                clips = np.stack(
                    [
                        np.stack(
                            [
                                cv2.resize(
                                    job.frames[s][y1:y2, x1:x2],
                                    (size, size),
                                    interpolation=cv2.INTER_AREA,
                                )
                                for s in samples
                            ]
                        )
                        for (x1, y1, x2, y2), samples in job.crops
                    ]
                )
                probabilities = self._model.classify(clips)
                self._inference_ms.append((time.perf_counter() - started) * 1000)
                self._apply(job, probabilities)
            except Exception:  # the tracking stream keeps running without classification
                self._counters["inference_errors"] += 1
                logger.exception("X3D classification failed for session=%s", self.session_id)
            finally:
                with self._lock:
                    self._pending -= 1

    def _apply(self, job: _Job, probabilities: np.ndarray) -> None:
        rule = self.config.rule
        ended: list[AggregatedEvent] = []
        with self._lock:
            if job.generation != self._generation or self._closed:
                self._counters["stale_windows"] += 1
                return
            self._counters["windows"] += len(job.actor_ids)
            for actor_id, probs in zip(job.actor_ids, probabilities, strict=True):
                actor = self._actors.get(actor_id)
                if actor is None:
                    continue
                actor.probs.append(probs)
                while len(actor.probs) > rule.smooth_windows:
                    actor.probs.popleft()
                smoothed = np.mean(actor.probs, axis=0)
                score = float(1.0 - smoothed[0])
                label_index = int(np.argmax(smoothed[1:]) + 1)
                label = self._classes[label_index]
                t = job.timestamp_ms
                if actor.alert is None and score >= rule.start_threshold:
                    actor.alert = _Alert(start_ms=t, end_ms=t, peak_ms=t)
                elif actor.alert is not None and score < rule.keep_threshold:
                    event = self._close_alert(actor)
                    if event is not None:
                        ended.append(event)
                if actor.alert is not None:
                    alert = actor.alert
                    alert.end_ms = t
                    alert.windows += 1
                    alert.score_sum += score
                    alert.labels[label] += 1
                    if actor.session_candidate_id is not None:
                        alert.candidates[actor.session_candidate_id] += 1
                    if score > alert.peak_score:
                        alert.peak_score, alert.peak_ms = score, t
                confirmed = actor.alert is not None and actor.alert.windows >= rule.min_windows
                actor.state = {
                    "actor_id": actor.actor_id,
                    "track_id": actor.track_id,
                    "session_candidate_id": (
                        str(actor.session_candidate_id) if actor.session_candidate_id else None
                    ),
                    "seat_code": actor.seat_code,
                    "timestamp_ms": t,
                    "probabilities": [round(float(v), 4) for v in probs],
                    "smoothed": [round(float(v), 4) for v in smoothed],
                    "cheat_score": round(score, 4),
                    "alert": confirmed,
                    "label": label if confirmed else self._classes[0],
                }
            states = [a.state for a in self._actors.values() if a.state is not None]
            message = {
                "type": "cheat_prediction",
                "session_id": str(self.session_id),
                "runtime_instance_id": str(self.runtime_instance_id),
                "runtime_generation": job.generation,
                "timestamp_ms": job.timestamp_ms,
                "model_name": self._model.model_name,
                "classes": list(self._classes),
                "rule": rule.model_dump(),
                "actors": states,
                "diagnostics": self._diagnostics_locked(),
            }
        self._publish(message)
        for event in ended:
            self._emit(event)

    # ---------- alerts -> events ----------
    def _close_alert(self, actor: _Actor) -> AggregatedEvent | None:
        """Lock held. Returns the event of a confirmed alert whose candidate is known."""
        alert, actor.alert = actor.alert, None
        if alert is None or alert.windows < self.config.rule.min_windows:
            return None
        self._counters["alerts"] += 1
        if not alert.candidates:
            self._counters["alerts_without_candidate"] += 1
            return None
        label = alert.labels.most_common(1)[0][0]
        candidate = alert.candidates.most_common(1)[0][0]
        return AggregatedEvent(
            session_id=self.session_id,
            behavior=BEHAVIOR_BY_CLASS.get(label, EventBehavior.SUSPICIOUS_LOOKING),
            session_candidate_ids=(candidate,),
            proposal_ids=(f"actor:{actor.actor_id}",),
            start_ms=alert.start_ms,
            end_ms=alert.end_ms,
            peak_ms=alert.peak_ms,
            peak_probability=alert.peak_score,
            active_probability_sum=alert.score_sum,
            active_prediction_count=alert.windows,
        )

    def _emit(self, event: AggregatedEvent) -> None:
        if self._emit_event is None:
            return
        try:
            self._emit_event(event)
            self._counters["events"] += 1
            self._publish(
                {
                    "type": "cheat_event",
                    "session_id": str(self.session_id),
                    "runtime_instance_id": str(self.runtime_instance_id),
                    "behavior": event.behavior.value,
                    "session_candidate_ids": [str(v) for v in event.session_candidate_ids],
                    "start_ms": event.start_ms,
                    "end_ms": event.end_ms,
                    "events_total": self._counters["events"],
                }
            )
        except Exception:
            self._counters["event_errors"] += 1
            logger.exception("Persisting X3D event failed for session=%s", self.session_id)

    def _diagnostics_locked(self) -> dict[str, Any]:
        samples = list(self._inference_ms)
        return {
            "device": self._model.device,
            "windows_total": self._counters["windows"],
            "windows_replaced": self._counters["windows_replaced"],
            "stale_windows": self._counters["stale_windows"],
            "inference_ms_mean": round(float(np.mean(samples)), 1) if samples else None,
            "inference_ms_p95": round(float(np.percentile(samples, 95)), 1) if samples else None,
            "alerts_total": self._counters["alerts"],
            "alerts_without_candidate": self._counters["alerts_without_candidate"],
            "events_total": self._counters["events"],
            "active_alerts": sum(
                1
                for a in self._actors.values()
                if a.alert is not None and a.alert.windows >= self.config.rule.min_windows
            ),
            "inference_errors": self._counters["inference_errors"],
        }

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return self._diagnostics_locked()
