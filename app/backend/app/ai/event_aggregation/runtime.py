from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from app.ai.action_recognition.adapter import R3_CLASS_NAMES
from app.ai.action_recognition.types import ActionPrediction, ProposalType
from app.ai.event_aggregation.types import (
    AggregatedEvent,
    EventAggregationDiagnostics,
    EventBehavior,
    EventFsmState,
)
from app.monitoring.config import EventBehaviorConfig, EventDetectionConfig

_ROUTING: dict[ProposalType, tuple[EventBehavior, ...]] = {
    ProposalType.SINGLE: (
        EventBehavior.SUSPICIOUS_LOOKING,
        EventBehavior.USING_PHONE_CHEAT_SHEET,
    ),
    ProposalType.PAIR: (
        EventBehavior.COMMUNICATING,
        EventBehavior.EXCHANGE_OBJECT,
    ),
}
_PROBABILITY_INDEX = {
    EventBehavior.SUSPICIOUS_LOOKING: 1,
    EventBehavior.COMMUNICATING: 2,
    EventBehavior.EXCHANGE_OBJECT: 3,
    EventBehavior.USING_PHONE_CHEAT_SHEET: 4,
}


@dataclass(slots=True)
class _Fsm:
    proposal_id: str
    behavior: EventBehavior
    actors: tuple[uuid.UUID, ...]
    config: EventBehaviorConfig
    state: EventFsmState = EventFsmState.IDLE
    ema: float | None = None
    last_timestamp_ms: int | None = None
    candidate_start_ms: int | None = None
    last_active_ms: int | None = None
    cooldown_start_ms: int | None = None
    peak_ms: int | None = None
    peak_probability: float = 0.0
    active_probability_sum: float = 0.0
    active_prediction_count: int = 0

    def clear_event(self) -> None:
        self.state = EventFsmState.IDLE
        self.candidate_start_ms = None
        self.last_active_ms = None
        self.cooldown_start_ms = None
        self.peak_ms = None
        self.peak_probability = 0.0
        self.active_probability_sum = 0.0
        self.active_prediction_count = 0

    def clear_all(self) -> None:
        self.clear_event()
        self.ema = None
        self.last_timestamp_ms = None

    def add_evidence(self, timestamp_ms: int, score: float) -> None:
        self.last_active_ms = timestamp_ms
        self.active_probability_sum += score
        self.active_prediction_count += 1
        if self.peak_ms is None or score > self.peak_probability:
            self.peak_probability = score
            self.peak_ms = timestamp_ms

    def event(self, session_id: uuid.UUID, end_ms: int) -> AggregatedEvent:
        if (
            self.candidate_start_ms is None
            or self.last_active_ms is None
            or self.peak_ms is None
            or self.active_prediction_count < 2
        ):
            raise RuntimeError("cannot finalize an incomplete event FSM")
        return AggregatedEvent(
            session_id=session_id,
            behavior=self.behavior,
            session_candidate_ids=self.actors,
            proposal_ids=(self.proposal_id,),
            start_ms=self.candidate_start_ms,
            end_ms=max(self.candidate_start_ms, end_ms),
            peak_ms=self.peak_ms,
            peak_probability=self.peak_probability,
            active_probability_sum=self.active_probability_sum,
            active_prediction_count=self.active_prediction_count,
        )


@dataclass(slots=True)
class _PendingEvent:
    event: AggregatedEvent
    merge_gap_ms: int


@dataclass(slots=True)
class _Counters:
    created: int = 0
    suppressed: int = 0
    deduplicated: int = 0
    per_behavior: dict[str, int] = field(
        default_factory=lambda: {behavior.value: 0 for behavior in EventBehavior}
    )


def temporal_iou(left: AggregatedEvent, right: AggregatedEvent) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    if union == 0:
        return 1.0 if left.start_ms == right.start_ms else 0.0
    return intersection / union


def _gap_ms(left: AggregatedEvent, right: AggregatedEvent) -> int:
    if left.end_ms < right.start_ms:
        return right.start_ms - left.end_ms
    if right.end_ms < left.start_ms:
        return left.start_ms - right.end_ms
    return 0


class EventAggregationRuntime:
    """Causal, source-timestamp event aggregation for raw R3 predictions."""

    def __init__(
        self,
        *,
        session_id: uuid.UUID,
        config: EventDetectionConfig,
        emit: Callable[[AggregatedEvent], None],
    ) -> None:
        if not config.enabled:
            raise ValueError("event aggregation runtime requires event_detection.enabled=true")
        self.session_id = session_id
        self.config = config
        self._emit = emit
        self._generation = 0
        self._fsms: dict[tuple[str, EventBehavior], _Fsm] = {}
        self._pending: list[_PendingEvent] = []
        self._recently_emitted: list[_PendingEvent] = []
        self._counters = _Counters()
        self._lock = threading.RLock()

    def consume(
        self,
        predictions: tuple[ActionPrediction, ...],
        runtime_generation: int,
    ) -> None:
        emitted: list[AggregatedEvent] = []
        with self._lock:
            if runtime_generation != self._generation:
                return
            watermark_ms: int | None = None
            for prediction in sorted(
                predictions, key=lambda item: (item.timestamp_ms, item.proposal_id)
            ):
                self._expire_stale(prediction.timestamp_ms)
                self._consume_prediction(prediction)
                watermark_ms = prediction.timestamp_ms
            if watermark_ms is not None:
                emitted.extend(self._flush_ready(watermark_ms))
        for event in emitted:
            self._emit(event)

    def advance(self, timestamp_ms: int, runtime_generation: int) -> None:
        """Advance source-video time without using wall clock or fabricating evidence."""
        emitted: list[AggregatedEvent] = []
        with self._lock:
            if runtime_generation != self._generation:
                return
            self._expire_stale(timestamp_ms)
            emitted.extend(self._flush_ready(timestamp_ms))
        for event in emitted:
            self._emit(event)

    def reset(self, runtime_generation: int) -> None:
        with self._lock:
            self._generation = runtime_generation
            self._counters.suppressed += sum(
                fsm.state is not EventFsmState.IDLE for fsm in self._fsms.values()
            )
            self._fsms.clear()
            self._pending.clear()
            self._recently_emitted.clear()

    def stop(self) -> None:
        emitted: list[AggregatedEvent] = []
        with self._lock:
            for fsm in self._fsms.values():
                if fsm.state in {EventFsmState.ACTIVE, EventFsmState.COOLDOWN}:
                    assert fsm.last_active_ms is not None
                    self._offer(fsm.event(self.session_id, fsm.last_active_ms), fsm.config)
                elif fsm.state is EventFsmState.CANDIDATE:
                    self._counters.suppressed += 1
            self._fsms.clear()
            emitted = [item.event for item in self._pending]
            self._pending.clear()
            for event in emitted:
                self._record_created(event)
        for event in sorted(emitted, key=lambda item: (item.start_ms, item.fingerprint)):
            self._emit(event)

    def diagnostics(self) -> EventAggregationDiagnostics:
        with self._lock:
            return EventAggregationDiagnostics(
                candidate_fsms=sum(
                    fsm.state is EventFsmState.CANDIDATE for fsm in self._fsms.values()
                ),
                active_fsms=sum(
                    fsm.state is EventFsmState.ACTIVE for fsm in self._fsms.values()
                ),
                cooldown_fsms=sum(
                    fsm.state is EventFsmState.COOLDOWN for fsm in self._fsms.values()
                ),
                events_created_total=self._counters.created,
                events_suppressed_total=self._counters.suppressed,
                events_deduplicated_total=self._counters.deduplicated,
                per_behavior_event_count=dict(self._counters.per_behavior),
            )

    def _consume_prediction(self, prediction: ActionPrediction) -> None:
        actors = tuple(sorted(prediction.session_candidate_ids, key=str))
        expected_actor_count = 1 if prediction.proposal_type is ProposalType.SINGLE else 2
        if len(actors) != expected_actor_count:
            self._counters.suppressed += 1
            return
        for behavior in _ROUTING[prediction.proposal_type]:
            behavior_config = self.config.behavior(behavior.value)
            if behavior_config is None:
                continue
            key = (prediction.proposal_id, behavior)
            fsm = self._fsms.get(key)
            if fsm is None:
                fsm = _Fsm(
                    proposal_id=prediction.proposal_id,
                    behavior=behavior,
                    actors=actors,
                    config=behavior_config,
                )
                self._fsms[key] = fsm
            elif fsm.actors != actors:
                self._counters.suppressed += 1
                fsm.clear_all()
                fsm.actors = actors
            self._advance(fsm, prediction.timestamp_ms, prediction.probabilities)

    def _advance(self, fsm: _Fsm, timestamp_ms: int, probabilities: tuple[float, ...]) -> None:
        if len(probabilities) != len(R3_CLASS_NAMES):
            self._counters.suppressed += 1
            return
        if fsm.last_timestamp_ms is not None and timestamp_ms <= fsm.last_timestamp_ms:
            return
        if (
            fsm.last_timestamp_ms is not None
            and timestamp_ms - fsm.last_timestamp_ms > self.config.max_discontinuity_ms
        ):
            self._finalize_discontinuity(fsm)
            fsm.ema = None
        raw_score = probabilities[_PROBABILITY_INDEX[fsm.behavior]]
        alpha = fsm.config.smoothing.alpha
        score = raw_score if fsm.ema is None else alpha * raw_score + (1 - alpha) * fsm.ema
        fsm.ema = score
        fsm.last_timestamp_ms = timestamp_ms

        if fsm.state is EventFsmState.IDLE:
            if score >= fsm.config.start_threshold:
                fsm.state = EventFsmState.CANDIDATE
                fsm.candidate_start_ms = timestamp_ms
                fsm.add_evidence(timestamp_ms, score)
            return

        if fsm.state is EventFsmState.CANDIDATE:
            if score < fsm.config.keep_threshold:
                self._counters.suppressed += 1
                fsm.clear_event()
                return
            fsm.add_evidence(timestamp_ms, score)
            assert fsm.candidate_start_ms is not None
            if (
                timestamp_ms - fsm.candidate_start_ms >= fsm.config.min_active_ms
                and fsm.active_prediction_count >= 2
            ):
                fsm.state = EventFsmState.ACTIVE
            return

        if fsm.state is EventFsmState.ACTIVE:
            if score >= fsm.config.keep_threshold:
                fsm.add_evidence(timestamp_ms, score)
            else:
                fsm.state = EventFsmState.COOLDOWN
                fsm.cooldown_start_ms = timestamp_ms
            return

        assert fsm.state is EventFsmState.COOLDOWN
        assert fsm.cooldown_start_ms is not None
        elapsed = timestamp_ms - fsm.cooldown_start_ms
        if score >= fsm.config.keep_threshold and elapsed <= fsm.config.end_grace_ms:
            fsm.state = EventFsmState.ACTIVE
            fsm.cooldown_start_ms = None
            fsm.add_evidence(timestamp_ms, score)
            return
        if elapsed >= fsm.config.end_grace_ms:
            assert fsm.last_active_ms is not None
            event = fsm.event(
                self.session_id,
                fsm.last_active_ms + fsm.config.end_grace_ms,
            )
            self._offer(event, fsm.config)
            fsm.clear_event()
            if score >= fsm.config.start_threshold:
                fsm.state = EventFsmState.CANDIDATE
                fsm.candidate_start_ms = timestamp_ms
                fsm.add_evidence(timestamp_ms, score)

    def _expire_stale(self, timestamp_ms: int) -> None:
        stale_keys = [
            key
            for key, fsm in self._fsms.items()
            if fsm.last_timestamp_ms is not None
            and timestamp_ms - fsm.last_timestamp_ms > self.config.max_discontinuity_ms
        ]
        for key in stale_keys:
            fsm = self._fsms.pop(key)
            self._finalize_discontinuity(fsm)

    def _finalize_discontinuity(self, fsm: _Fsm) -> None:
        if fsm.state in {EventFsmState.ACTIVE, EventFsmState.COOLDOWN}:
            assert fsm.last_active_ms is not None
            self._offer(fsm.event(self.session_id, fsm.last_active_ms), fsm.config)
        elif fsm.state is EventFsmState.CANDIDATE:
            self._counters.suppressed += 1
        fsm.clear_event()

    def _offer(self, event: AggregatedEvent, behavior_config: EventBehaviorConfig) -> None:
        for pending in self._pending:
            current = pending.event
            if self._is_duplicate(current, event, behavior_config):
                peak = current
                if event.peak_probability > current.peak_probability:
                    peak = event
                pending.event = replace(
                    current,
                    proposal_ids=tuple(sorted(set(current.proposal_ids + event.proposal_ids))),
                    start_ms=min(current.start_ms, event.start_ms),
                    end_ms=max(current.end_ms, event.end_ms),
                    peak_ms=peak.peak_ms,
                    peak_probability=peak.peak_probability,
                    active_probability_sum=(
                        current.active_probability_sum + event.active_probability_sum
                    ),
                    active_prediction_count=(
                        current.active_prediction_count + event.active_prediction_count
                    ),
                )
                pending.merge_gap_ms = max(pending.merge_gap_ms, behavior_config.merge_gap_ms)
                self._counters.deduplicated += 1
                return
        if any(
            self._is_duplicate(item.event, event, behavior_config)
            for item in self._recently_emitted
        ):
            self._counters.deduplicated += 1
            return
        self._pending.append(_PendingEvent(event, behavior_config.merge_gap_ms))

    def _is_duplicate(
        self,
        current: AggregatedEvent,
        event: AggregatedEvent,
        behavior_config: EventBehaviorConfig,
    ) -> bool:
        return (
            current.behavior is event.behavior
            and current.session_candidate_ids == event.session_candidate_ids
            and (
                temporal_iou(current, event) >= self.config.dedup_temporal_iou
                or _gap_ms(current, event)
                <= min(self.config.dedup_max_gap_ms, behavior_config.merge_gap_ms)
            )
        )

    def _flush_ready(self, watermark_ms: int) -> list[AggregatedEvent]:
        self._recently_emitted = [
            item
            for item in self._recently_emitted
            if watermark_ms <= item.event.end_ms + self.config.max_discontinuity_ms
        ]
        ready = [
            item
            for item in self._pending
            if watermark_ms > item.event.end_ms + item.merge_gap_ms
        ]
        self._pending = [item for item in self._pending if item not in ready]
        events = sorted((item.event for item in ready), key=lambda item: item.fingerprint)
        for event in events:
            self._record_created(event)
            self._recently_emitted.append(
                _PendingEvent(event, self.config.dedup_max_gap_ms)
            )
        return events

    def _record_created(self, event: AggregatedEvent) -> None:
        self._counters.created += 1
        self._counters.per_behavior[event.behavior.value] += 1
