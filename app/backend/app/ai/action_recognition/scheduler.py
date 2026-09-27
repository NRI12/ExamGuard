from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from app.ai.action_recognition.types import ActionClip, ActionPrediction, ProposalType


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    ready_proposals: int
    in_flight_proposals: int
    stale_drops: int
    replaced_ready_requests: int
    single_predictions: int
    pair_predictions: int
    single_interval_ms_mean: float | None
    single_interval_ms_p95: float | None
    single_interval_ms_max: float | None
    pair_interval_ms_mean: float | None
    pair_interval_ms_p95: float | None
    pair_interval_ms_max: float | None


class ActionScheduler:
    """Deterministic oldest-prediction-first compute budget for stable proposals."""

    def __init__(
        self,
        *,
        prediction_stride_ms: int,
        max_batch_size: int,
        max_prediction_age_ms: int,
        min_inference_interval_ms: int,
    ) -> None:
        self.prediction_stride_ms = prediction_stride_ms
        self.max_batch_size = max_batch_size
        self.max_prediction_age_ms = max_prediction_age_ms
        self.min_inference_interval_ms = min_inference_interval_ms
        self._ready: dict[str, ActionClip] = {}
        self._in_flight: set[str] = set()
        self._last_prediction_ms: dict[str, int] = {}
        self._proposal_types: dict[str, ProposalType] = {}
        self._intervals: dict[ProposalType, deque[float]] = defaultdict(
            lambda: deque(maxlen=1024)
        )
        self._prediction_counts: dict[ProposalType, int] = defaultdict(int)
        self._last_batch_timestamp_ms: int | None = None
        self._stale_drops = 0
        self._replaced = 0

    def observe(self, clips: tuple[ActionClip, ...], timestamp_ms: int) -> None:
        self._drop_stale(timestamp_ms)
        for clip in clips:
            proposal_id = clip.proposal.proposal_id
            self._proposal_types[proposal_id] = clip.proposal.proposal_type
            if timestamp_ms - clip.end_timestamp_ms > self.max_prediction_age_ms:
                self._stale_drops += 1
                continue
            previous = self._ready.get(proposal_id)
            if previous is not None and clip.end_timestamp_ms > previous.end_timestamp_ms:
                self._replaced += 1
            if previous is None or clip.end_timestamp_ms >= previous.end_timestamp_ms:
                self._ready[proposal_id] = clip

    def next_batch(self, timestamp_ms: int) -> tuple[ActionClip, ...]:
        self._drop_stale(timestamp_ms)
        if (
            self._last_batch_timestamp_ms is not None
            and timestamp_ms - self._last_batch_timestamp_ms < self.min_inference_interval_ms
        ):
            return ()
        eligible = [
            clip
            for proposal_id, clip in self._ready.items()
            if proposal_id not in self._in_flight and self._eligible(proposal_id, timestamp_ms)
        ]
        eligible.sort(key=self._priority)
        selected = tuple(eligible[: self.max_batch_size])
        if not selected:
            return ()
        for clip in selected:
            proposal_id = clip.proposal.proposal_id
            self._ready.pop(proposal_id, None)
            self._in_flight.add(proposal_id)
        self._last_batch_timestamp_ms = timestamp_ms
        return selected

    def complete(self, predictions: tuple[ActionPrediction, ...]) -> None:
        for prediction in predictions:
            proposal_id = prediction.proposal_id
            previous = self._last_prediction_ms.get(proposal_id)
            if previous is not None and prediction.timestamp_ms > previous:
                self._intervals[prediction.proposal_type].append(
                    float(prediction.timestamp_ms - previous)
                )
            self._last_prediction_ms[proposal_id] = prediction.timestamp_ms
            self._proposal_types[proposal_id] = prediction.proposal_type
            self._prediction_counts[prediction.proposal_type] += 1
            self._in_flight.discard(proposal_id)

    def fail(self, clips: tuple[ActionClip, ...]) -> None:
        for clip in clips:
            self._in_flight.discard(clip.proposal.proposal_id)

    def reset(self) -> None:
        self._ready.clear()
        self._in_flight.clear()
        self._last_prediction_ms.clear()
        self._proposal_types.clear()
        self._intervals.clear()
        self._prediction_counts.clear()
        self._last_batch_timestamp_ms = None
        self._stale_drops = 0
        self._replaced = 0

    def snapshot(self) -> SchedulerSnapshot:
        single = list(self._intervals[ProposalType.SINGLE])
        pair = list(self._intervals[ProposalType.PAIR])
        return SchedulerSnapshot(
            ready_proposals=len(self._ready),
            in_flight_proposals=len(self._in_flight),
            stale_drops=self._stale_drops,
            replaced_ready_requests=self._replaced,
            single_predictions=self._prediction_counts[ProposalType.SINGLE],
            pair_predictions=self._prediction_counts[ProposalType.PAIR],
            single_interval_ms_mean=self._mean(single),
            single_interval_ms_p95=self._p95(single),
            single_interval_ms_max=max(single, default=None),
            pair_interval_ms_mean=self._mean(pair),
            pair_interval_ms_p95=self._p95(pair),
            pair_interval_ms_max=max(pair, default=None),
        )

    def _eligible(self, proposal_id: str, timestamp_ms: int) -> bool:
        previous = self._last_prediction_ms.get(proposal_id)
        return previous is None or timestamp_ms - previous >= self.prediction_stride_ms

    def _drop_stale(self, timestamp_ms: int) -> None:
        for proposal_id, clip in tuple(self._ready.items()):
            if timestamp_ms - clip.end_timestamp_ms > self.max_prediction_age_ms:
                del self._ready[proposal_id]
                self._stale_drops += 1

    def _priority(self, clip: ActionClip) -> tuple[int, int, int, str]:
        proposal_id = clip.proposal.proposal_id
        previous = self._last_prediction_ms.get(proposal_id)
        return (
            0 if previous is None else 1,
            previous if previous is not None else -1,
            clip.end_timestamp_ms,
            proposal_id,
        )

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return float(np.mean(np.asarray(values))) if values else None

    @staticmethod
    def _p95(values: list[float]) -> float | None:
        return float(np.percentile(np.asarray(values), 95)) if values else None
