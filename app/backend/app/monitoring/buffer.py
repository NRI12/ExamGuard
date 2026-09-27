from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestValueBuffer(Generic[T]):
    """A one-slot buffer that replaces stale work instead of growing a backlog."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._closed = False
        self.dropped = 0

    def put(self, value: T) -> None:
        with self._condition:
            if self._closed:
                return
            if self._value is not None:
                self.dropped += 1
            self._value = value
            self._condition.notify()

    def get(self, timeout: float | None = None) -> T | None:
        with self._condition:
            if self._value is None and not self._closed:
                self._condition.wait(timeout)
            value = self._value
            self._value = None
            return value

    def clear(self) -> None:
        with self._condition:
            self._value = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._value = None
            self._condition.notify_all()

    @property
    def size(self) -> int:
        with self._condition:
            return int(self._value is not None)
