from __future__ import annotations

import itertools
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.ai.action_recognition.adapter import R3_CLASS_NAMES
from app.ai.action_recognition.types import ActionPrediction
from app.ai.event_aggregation.runtime import EventAggregationRuntime, temporal_iou
from app.ai.event_aggregation.types import AggregatedEvent, EventBehavior
from app.monitoring.config import (
    EventBehaviorConfig,
    EventDetectionConfig,
    EventSmoothingConfig,
)

CALIBRATION_SPLITS = frozenset({"development", "validation"})
FINAL_TEST_SPLITS = frozenset({"final", "final_test", "test"})


def _raw_behavior(value: str) -> str:
    return "using_phone/cheat_sheet" if value == "using_phone_cheat_sheet" else value


def _event_behavior(value: str) -> str:
    return value.replace("/", "_")


@dataclass(frozen=True, slots=True)
class GroundTruthEvent:
    event_id: str
    session_id: uuid.UUID
    video_id: str
    split: str
    behavior: str
    start_ms: int
    end_ms: int
    session_candidate_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class CalibrationPrediction:
    session_id: uuid.UUID
    video_id: str
    split: str
    prediction: ActionPrediction


def assert_calibration_only(splits: Iterable[str]) -> None:
    normalized = {value.strip().lower() for value in splits}
    forbidden = normalized & FINAL_TEST_SPLITS
    unknown = normalized - CALIBRATION_SPLITS - FINAL_TEST_SPLITS
    if forbidden:
        raise ValueError(f"final-test partitions are excluded from search: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"unsupported calibration partitions: {sorted(unknown)}")


def actors_compatible(
    predicted: tuple[uuid.UUID, ...],
    expected: tuple[uuid.UUID, ...],
) -> bool:
    return not expected or frozenset(predicted) == frozenset(expected)


def align_predictions(
    predictions: list[CalibrationPrediction],
    ground_truth: list[GroundTruthEvent],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Align by session/video/time and actor set; unmatched predictions are normal."""
    invalid: list[str] = []
    by_video: dict[tuple[uuid.UUID, str], list[GroundTruthEvent]] = {}
    for event in ground_truth:
        if event.end_ms < event.start_ms or event.start_ms < 0:
            invalid.append(f"{event.event_id}: invalid time range")
            continue
        if _raw_behavior(event.behavior) not in R3_CLASS_NAMES:
            invalid.append(f"{event.event_id}: unknown behavior {event.behavior}")
            continue
        by_video.setdefault((event.session_id, event.video_id), []).append(event)

    aligned: list[dict[str, Any]] = []
    for item in predictions:
        prediction = item.prediction
        matches = [
            event
            for event in by_video.get((item.session_id, item.video_id), [])
            if event.start_ms <= prediction.timestamp_ms <= event.end_ms
            and actors_compatible(prediction.session_candidate_ids, event.session_candidate_ids)
        ]
        matches.sort(key=lambda event: (event.end_ms - event.start_ms, event.event_id))
        matched = matches[0] if matches else None
        aligned.append(
            {
                "session_id": str(item.session_id),
                "video_id": item.video_id,
                "split": item.split,
                "timestamp_ms": prediction.timestamp_ms,
                "proposal_id": prediction.proposal_id,
                "proposal_type": prediction.proposal_type.value,
                "session_candidate_ids": [
                    str(value) for value in prediction.session_candidate_ids
                ],
                "seat_codes": list(prediction.seat_codes),
                **{
                    f"prob_{name.replace('/', '_')}": probability
                    for name, probability in zip(
                        R3_CLASS_NAMES, prediction.probabilities, strict=True
                    )
                },
                "top_class": prediction.predicted_class,
                "top_confidence": prediction.confidence,
                "matched_gt_event_id": matched.event_id if matched else None,
                "matched_gt_behavior": _raw_behavior(matched.behavior) if matched else "normal",
            }
        )
    return aligned, invalid


def classification_metrics(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in R3_CLASS_NAMES} for actual in R3_CLASS_NAMES}
    for row in aligned:
        actual = str(row["matched_gt_behavior"])
        predicted = str(row["top_class"])
        if actual not in matrix or predicted not in matrix[actual]:
            raise ValueError(f"unknown class in aligned prediction: {actual}/{predicted}")
        matrix[actual][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    total = len(aligned)
    weighted = 0.0
    for name in R3_CLASS_NAMES:
        tp = matrix[name][name]
        fp = sum(matrix[actual][name] for actual in R3_CLASS_NAMES if actual != name)
        fn = sum(matrix[name][predicted] for predicted in R3_CLASS_NAMES if predicted != name)
        support = sum(matrix[name].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted += f1 * support
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "prediction_count": total,
        "per_class": per_class,
        "macro_f1": sum(float(row["f1"]) for row in per_class.values()) / len(per_class),
        "weighted_f1": weighted / total if total else 0.0,
        "confusion_matrix": {
            "labels": list(R3_CLASS_NAMES),
            "rows": [
                [matrix[actual][predicted] for predicted in R3_CLASS_NAMES]
                for actual in R3_CLASS_NAMES
            ],
        },
    }


def probability_distributions(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for behavior in EventBehavior:
        field = f"prob_{behavior.value.replace('/', '_')}"
        raw_behavior = _raw_behavior(behavior.value)
        positive = sorted(
            float(row[field])
            for row in aligned
            if row["matched_gt_behavior"] == raw_behavior
        )
        negative = sorted(
            float(row[field])
            for row in aligned
            if row["matched_gt_behavior"] != raw_behavior
        )
        output[behavior.value] = {
            "positive": _distribution_summary(positive),
            "negative": _distribution_summary(negative),
        }
    return output


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p50": values[round((len(values) - 1) * 0.50)],
        "p95": values[round((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def event_metrics(
    predicted: list[AggregatedEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    duration_minutes: float,
    temporal_iou_threshold: float = 0.30,
) -> dict[str, Any]:
    if not 0 <= temporal_iou_threshold <= 1:
        raise ValueError("temporal_iou_threshold must be in [0,1]")
    event_behaviors = {behavior.value for behavior in EventBehavior}
    event_truth = [
        truth
        for truth in ground_truth
        if _event_behavior(truth.behavior) in event_behaviors
    ]
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, event in enumerate(predicted):
        for truth_index, truth in enumerate(event_truth):
            if (
                event.session_id != truth.session_id
                or event.behavior.value != _event_behavior(truth.behavior)
                or not actors_compatible(event.session_candidate_ids, truth.session_candidate_ids)
            ):
                continue
            truth_event = AggregatedEvent(
                session_id=truth.session_id,
                behavior=EventBehavior(_event_behavior(truth.behavior)),
                session_candidate_ids=truth.session_candidate_ids,
                proposal_ids=(truth.event_id,),
                start_ms=truth.start_ms,
                end_ms=truth.end_ms,
                peak_ms=truth.start_ms,
                peak_probability=1.0,
                active_probability_sum=1.0,
                active_prediction_count=1,
            )
            overlap = temporal_iou(event, truth_event)
            if overlap >= temporal_iou_threshold:
                candidates.append((overlap, predicted_index, truth_index))
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for _, predicted_index, truth_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if predicted_index not in matched_predictions and truth_index not in matched_truth:
            matched_predictions.add(predicted_index)
            matched_truth.add(truth_index)

    per_class: dict[str, dict[str, float | int]] = {}
    for behavior in EventBehavior:
        predicted_indexes = {
            index for index, event in enumerate(predicted) if event.behavior is behavior
        }
        truth_indexes = {
            index
            for index, event in enumerate(event_truth)
            if _event_behavior(event.behavior) == behavior.value
        }
        tp = len(predicted_indexes & matched_predictions)
        fp = len(predicted_indexes - matched_predictions)
        fn = len(truth_indexes - matched_truth)
        per_class[behavior.value] = _prf(tp, fp, fn)
    tp = len(matched_predictions)
    fp = len(predicted) - tp
    fn = len(event_truth) - len(matched_truth)
    duplicate_count = _duplicate_count(predicted)
    return {
        **_prf(tp, fp, fn),
        "per_class": per_class,
        "macro_event_f1": sum(float(value["f1"]) for value in per_class.values())
        / len(per_class),
        "false_events_per_minute": fp / duration_minutes if duration_minutes > 0 else 0.0,
        "duplicate_event_rate": duplicate_count / len(predicted) if predicted else 0.0,
        "event_counts": dict(Counter(event.behavior.value for event in predicted)),
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _duplicate_count(events: list[AggregatedEvent]) -> int:
    duplicates = 0
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            if (
                left.behavior is right.behavior
                and left.session_candidate_ids == right.session_candidate_ids
                and temporal_iou(left, right) > 0
            ):
                duplicates += 1
    return duplicates


def aggregate_predictions(
    predictions: list[CalibrationPrediction],
    config: EventDetectionConfig,
) -> list[AggregatedEvent]:
    output: list[AggregatedEvent] = []
    by_session: dict[uuid.UUID, list[ActionPrediction]] = {}
    for item in predictions:
        by_session.setdefault(item.session_id, []).append(item.prediction)
    for session_id in sorted(by_session, key=str):
        runtime = EventAggregationRuntime(session_id=session_id, config=config, emit=output.append)
        for prediction in sorted(
            by_session[session_id], key=lambda item: (item.timestamp_ms, item.proposal_id)
        ):
            runtime.consume((prediction,), 0)
        runtime.stop()
    return output


def bounded_parameter_search(
    predictions: list[CalibrationPrediction],
    ground_truth: list[GroundTruthEvent],
    *,
    duration_minutes: float,
    search_space: dict[str, list[float | int]],
) -> tuple[EventDetectionConfig, dict[str, Any]]:
    assert_calibration_only(item.split for item in predictions)
    assert_calibration_only(item.split for item in ground_truth)
    keys = (
        "alpha",
        "start_threshold",
        "keep_threshold",
        "min_active_ms",
        "end_grace_ms",
        "merge_gap_ms",
    )
    missing = [key for key in keys if not search_space.get(key)]
    if missing:
        raise ValueError(f"search space is missing values for: {missing}")
    combinations = list(itertools.product(*(search_space[key] for key in keys)))
    if len(combinations) > 5000:
        raise ValueError("bounded search is limited to 5000 combinations per behavior")

    selected: dict[EventBehavior, EventBehaviorConfig] = {}
    selected_metrics: dict[str, Any] = {}
    for behavior in EventBehavior:
        best: tuple[tuple[float, float, float], EventBehaviorConfig, dict[str, Any]] | None = None
        for values in combinations:
            alpha, start, keep, minimum, grace, merge = values
            if float(start) <= float(keep):
                continue
            candidate = EventBehaviorConfig(
                proposal_types=[
                    "SINGLE"
                    if behavior
                    in {
                        EventBehavior.SUSPICIOUS_LOOKING,
                        EventBehavior.USING_PHONE_CHEAT_SHEET,
                    }
                    else "PAIR"
                ],
                smoothing=EventSmoothingConfig(alpha=float(alpha)),
                start_threshold=float(start),
                keep_threshold=float(keep),
                min_active_ms=int(minimum),
                end_grace_ms=int(grace),
                merge_gap_ms=int(merge),
            )
            trial_configs = {
                item: selected.get(
                    item,
                    candidate.model_copy(
                        update={
                            "proposal_types": [
                                "SINGLE"
                                if item
                                in {
                                    EventBehavior.SUSPICIOUS_LOOKING,
                                    EventBehavior.USING_PHONE_CHEAT_SHEET,
                                }
                                else "PAIR"
                            ]
                        }
                    ),
                )
                for item in EventBehavior
            }
            trial_configs[behavior] = candidate
            config = _event_config(trial_configs)
            events = [
                event
                for event in aggregate_predictions(predictions, config)
                if event.behavior is behavior
            ]
            truth = [
                event
                for event in ground_truth
                if _event_behavior(event.behavior) == behavior.value
            ]
            metrics = event_metrics(
                events,
                truth,
                duration_minutes=duration_minutes,
            )
            score = (
                float(metrics["f1"]),
                float(metrics["recall"]),
                -float(metrics["false_events_per_minute"]),
            )
            if best is None or score > best[0]:
                best = score, candidate, metrics
        if best is None:
            raise ValueError(f"no valid hysteresis combination for {behavior.value}")
        selected[behavior] = best[1]
        selected_metrics[behavior.value] = best[2]
    return _event_config(selected), selected_metrics


def _event_config(
    behaviors: dict[EventBehavior, EventBehaviorConfig],
) -> EventDetectionConfig:
    return EventDetectionConfig(
        enabled=True,
        suspicious_looking=behaviors[EventBehavior.SUSPICIOUS_LOOKING],
        communicating=behaviors[EventBehavior.COMMUNICATING],
        exchange_object=behaviors[EventBehavior.EXCHANGE_OBJECT],
        using_phone_cheat_sheet=behaviors[EventBehavior.USING_PHONE_CHEAT_SHEET],
    )
