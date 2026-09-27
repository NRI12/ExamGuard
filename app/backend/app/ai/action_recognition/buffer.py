from __future__ import annotations

from collections import deque

import numpy as np

from app.ai.action_recognition.types import ActionClip, ActionProposal, BufferedRoiFrame


class TimestampedRoiBuffer:
    def __init__(
        self,
        *,
        clip_span_ms: int,
        num_segments: int,
        capture_interval_ms: int,
        sample_tolerance_ms: int,
        max_gap_ms: int,
    ) -> None:
        self.clip_span_ms = clip_span_ms
        self.num_segments = num_segments
        self.capture_interval_ms = capture_interval_ms
        self.sample_tolerance_ms = sample_tolerance_ms
        self.max_gap_ms = max_gap_ms
        self.max_frames = clip_span_ms // capture_interval_ms + 3
        self._frames: dict[str, deque[BufferedRoiFrame]] = {}

    @property
    def proposal_count(self) -> int:
        return len(self._frames)

    @property
    def stored_frame_count(self) -> int:
        return sum(len(frames) for frames in self._frames.values())

    def clear(self) -> None:
        self._frames.clear()

    def prune(self, active_ids: set[str], timestamp_ms: int) -> None:
        oldest_allowed = timestamp_ms - self.clip_span_ms - self.max_gap_ms
        for proposal_id in tuple(self._frames):
            frames = self._frames[proposal_id]
            expired = not frames or frames[-1].timestamp_ms < oldest_allowed
            if proposal_id not in active_ids and expired:
                del self._frames[proposal_id]

    def should_capture(self, proposal_id: str, timestamp_ms: int) -> bool:
        frames = self._frames.get(proposal_id)
        if not frames:
            return True
        if timestamp_ms <= frames[-1].timestamp_ms:
            return timestamp_ms < frames[-1].timestamp_ms
        return timestamp_ms - frames[-1].timestamp_ms >= self.capture_interval_ms

    def append(self, proposal: ActionProposal, timestamp_ms: int, rgb: np.ndarray) -> bool:
        frames = self._frames.setdefault(proposal.proposal_id, deque(maxlen=self.max_frames))
        if frames and timestamp_ms <= frames[-1].timestamp_ms:
            if timestamp_ms < frames[-1].timestamp_ms:
                frames.clear()
            else:
                return False
        if frames and timestamp_ms - frames[-1].timestamp_ms > self.max_gap_ms:
            frames.clear()
        if frames and timestamp_ms - frames[-1].timestamp_ms < self.capture_interval_ms:
            return False
        frames.append(BufferedRoiFrame(timestamp_ms=timestamp_ms, rgb=rgb))
        return True

    def clip_if_ready(
        self,
        proposal: ActionProposal,
        timestamp_ms: int,
    ) -> ActionClip | None:
        frames = self._frames.get(proposal.proposal_id)
        if not frames:
            return None
        start_ms = timestamp_ms - self.clip_span_ms
        segment_ms = self.clip_span_ms / self.num_segments
        targets = [start_ms + (index + 0.5) * segment_ms for index in range(self.num_segments)]
        selected: list[np.ndarray] = []
        for target in targets:
            nearest = min(frames, key=lambda item: abs(item.timestamp_ms - target))
            if abs(nearest.timestamp_ms - target) > self.sample_tolerance_ms:
                return None
            selected.append(nearest.rgb)
        return ActionClip(
            proposal=proposal,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=timestamp_ms,
            frames=tuple(selected),
        )
