from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from app.ai.domain import Detection, TrackedObject
from app.monitoring.config import ByteTrackConfig


@dataclass(frozen=True, slots=True)
class TrackLifecycleEvent:
    event: str
    track_id: int


class ByteTrackAdapter:
    def __init__(self, config: ByteTrackConfig) -> None:
        self.config = config
        self.instance_id = uuid.uuid4()
        self._tracker = BYTETracker(SimpleNamespace(**config.model_dump()))
        self._active_ids: set[int] = set()
        self._seen_ids: set[int] = set()
        self._reported_removed_ids: set[int] = set()
        self._lifecycle_events: list[TrackLifecycleEvent] = []
        self._last_timestamp_ms: int | None = None

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
        timestamp_ms: int,
    ) -> list[TrackedObject]:
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                "ByteTrack input timestamp must increase within one runtime generation"
            )
        self._last_timestamp_ms = timestamp_ms
        rows = np.asarray(
            [
                [*detection.bbox_xyxy, detection.confidence, detection.class_id]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 6))
        results = self._tracker.update(Boxes(rows, orig_shape=frame_shape))
        tracked = [
            TrackedObject(
                track_id=int(row[4]),
                bbox_xyxy=tuple(float(value) for value in row[:4]),  # type: ignore[arg-type]
                confidence=float(row[5]),
            )
            for row in results
        ]
        current_ids = {item.track_id for item in tracked}
        removed_ids = {
            int(item.track_id)
            for item in getattr(self._tracker, "removed_stracks", ())
            if getattr(item, "track_id", None) is not None
        }
        for track_id in sorted(current_ids - self._seen_ids):
            self._lifecycle_events.append(TrackLifecycleEvent("TRACK_CREATED", track_id))
        for track_id in sorted(current_ids & self._seen_ids):
            self._lifecycle_events.append(TrackLifecycleEvent("TRACK_UPDATED", track_id))
        for track_id in sorted((self._active_ids - current_ids) - removed_ids):
            self._lifecycle_events.append(TrackLifecycleEvent("TRACK_LOST", track_id))
        for track_id in sorted(removed_ids - self._reported_removed_ids):
            self._lifecycle_events.append(TrackLifecycleEvent("TRACK_REMOVED", track_id))
        self._active_ids = current_ids
        self._seen_ids.update(current_ids)
        self._reported_removed_ids.update(removed_ids)
        return tracked

    def drain_lifecycle_events(self) -> tuple[TrackLifecycleEvent, ...]:
        events = tuple(self._lifecycle_events)
        self._lifecycle_events.clear()
        return events

    def reset(self) -> None:
        self._tracker.reset()
        self._active_ids.clear()
        self._seen_ids.clear()
        self._reported_removed_ids.clear()
        self._lifecycle_events.clear()
        self._last_timestamp_ms = None
