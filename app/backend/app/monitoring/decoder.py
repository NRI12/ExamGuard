from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np


class VideoDecoderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int
    timestamp_ms: int
    source_width: int
    source_height: int
    generation: int


class VideoDecoder:
    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise VideoDecoderError(f"Không thể mở video: {path}")
        self.source_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.source_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.source_fps <= 0 or self.source_width <= 0 or self.source_height <= 0:
            self.release()
            raise VideoDecoderError("Metadata video không hợp lệ cho phân tích")

    def seek(self, timestamp_ms: int) -> None:
        if not self._capture.set(cv2.CAP_PROP_POS_MSEC, float(max(0, timestamp_ms))):
            raise VideoDecoderError("Video decoder không hỗ trợ seek")

    def read_for_timestamp(
        self,
        target_ms: int,
        generation: int,
    ) -> tuple[FramePacket | None, int]:
        discarded = 0
        tolerance_ms = 500.0 / self.source_fps
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return None, discarded
            frame_id = max(0, int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            reported_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback_ms = frame_id / self.source_fps * 1000
            timestamp_ms = round(reported_ms if reported_ms > 0 else fallback_ms)
            if timestamp_ms + tolerance_ms >= target_ms:
                return (
                    FramePacket(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        source_width=self.source_width,
                        source_height=self.source_height,
                        generation=generation,
                    ),
                    discarded,
                )
            discarded += 1

    def release(self) -> None:
        self._capture.release()
