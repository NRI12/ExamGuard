from __future__ import annotations

import threading
import time


class AnalysisClock:
    def __init__(self, timestamp_ms: int = 0) -> None:
        self._lock = threading.Lock()
        self._anchor_timestamp_ms = float(timestamp_ms)
        self._anchor_wall = time.monotonic()
        self._paused = False

    def timestamp_ms(self, now: float | None = None) -> int:
        with self._lock:
            if self._paused:
                return round(self._anchor_timestamp_ms)
            elapsed_ms = ((now if now is not None else time.monotonic()) - self._anchor_wall) * 1000
            return max(0, round(self._anchor_timestamp_ms + elapsed_ms))

    def pause(self, timestamp_ms: int | None = None) -> None:
        with self._lock:
            if timestamp_ms is None:
                timestamp_ms = round(
                    self._anchor_timestamp_ms + (time.monotonic() - self._anchor_wall) * 1000
                )
            self._anchor_timestamp_ms = float(max(0, timestamp_ms))
            self._paused = True

    def resume(self, timestamp_ms: int | None = None) -> None:
        with self._lock:
            if timestamp_ms is not None:
                self._anchor_timestamp_ms = float(max(0, timestamp_ms))
            self._anchor_wall = time.monotonic()
            self._paused = False

    def seek(self, timestamp_ms: int) -> None:
        with self._lock:
            self._anchor_timestamp_ms = float(max(0, timestamp_ms))
            self._anchor_wall = time.monotonic()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused
