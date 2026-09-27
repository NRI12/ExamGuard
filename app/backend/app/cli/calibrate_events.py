"""Collect production-path ActionPredictions and calibrate Phase 7 Event FSMs."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.ai.action_recognition.adapter import R3_CLASS_NAMES, ActionModelAdapter
from app.ai.action_recognition.runtime import ActionRecognitionRuntime
from app.ai.action_recognition.types import ActionPrediction, ProposalType
from app.ai.detector.yolo import PersonDetector
from app.ai.domain import Track, normalize_bbox
from app.ai.event_aggregation.calibration import (
    CALIBRATION_SPLITS,
    CalibrationPrediction,
    GroundTruthEvent,
    aggregate_predictions,
    align_predictions,
    assert_calibration_only,
    bounded_parameter_search,
    classification_metrics,
    event_metrics,
    probability_distributions,
)
from app.ai.seat_identity.assignment import SeatAssignmentEngine
from app.ai.seat_identity.types import AssignmentState, TrackIdentity
from app.ai.tracker.bytetrack import ByteTrackAdapter
from app.cli.validate_seat_identity import load_context
from app.monitoring.config import load_runtime_profile_from_paths
from app.monitoring.decoder import VideoDecoder


def _manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sessions"), list):
        raise ValueError("event calibration manifest must contain a sessions list")
    return payload


def _resolve_sessions(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for raw in payload["sessions"]:
        if not isinstance(raw, dict):
            raise ValueError("each calibration session must be a mapping")
        session = dict(raw)
        source = Path(str(session.get("source", "")))
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        session["source"] = str(source)
        session["split"] = str(session.get("split", "")).strip().lower()
        sessions.append(session)
    return sessions


def _ground_truth(sessions: list[dict[str, Any]]) -> list[GroundTruthEvent]:
    output: list[GroundTruthEvent] = []
    for session in sessions:
        clip_id = str(session["id"])
        context, _ = load_context(clip_id, {**session, "expected_actors": []})
        actors_by_seat = {
            seat.code: binding.session_candidate_id
            for seat in context.seats
            for binding in context.bindings
            if binding.seat_id == seat.id
        }
        for raw in session.get("events", []):
            seat_codes = tuple(str(value) for value in raw.get("seat_codes", []))
            missing = [code for code in seat_codes if code not in actors_by_seat]
            if missing:
                raise ValueError(f"event {raw.get('id')} references unknown seats: {missing}")
            output.append(
                GroundTruthEvent(
                    event_id=str(raw["id"]),
                    session_id=context.session_id,
                    video_id=str(session.get("video_id", clip_id)),
                    split=str(session["split"]),
                    behavior=str(raw["behavior"]),
                    start_ms=int(raw["start_ms"]),
                    end_ms=int(raw["end_ms"]),
                    session_candidate_ids=tuple(
                        sorted((actors_by_seat[code] for code in seat_codes), key=str)
                    ),
                )
            )
    return output


def _prediction_from_message(row: dict[str, Any]) -> ActionPrediction:
    probabilities = tuple(float(row["class_probabilities"][name]) for name in R3_CLASS_NAMES)
    return ActionPrediction(
        proposal_id=str(row["proposal_id"]),
        proposal_type=ProposalType(str(row["proposal_type"])),
        session_candidate_ids=tuple(uuid.UUID(value) for value in row["session_candidate_ids"]),
        seat_codes=tuple(str(value) for value in row["seat_codes"]),
        timestamp_ms=int(row["timestamp_ms"]),
        probabilities=probabilities,
        predicted_class=str(row["predicted_class"]),
        confidence=float(row["confidence"]),
        model_name=str(row.get("model_name", "r3_tsm_r50_k400_diff_final")),
    )


def collect(
    args: argparse.Namespace,
    sessions: list[dict[str, Any]],
) -> list[CalibrationPrediction]:
    selected = [session for session in sessions if session["split"] in CALIBRATION_SPLITS]
    assert_calibration_only(session["split"] for session in selected)
    if not selected:
        raise ValueError("manifest has no development/validation sessions")
    profile = load_runtime_profile_from_paths(
        config_root=args.config_root,
        model_root=args.model_root,
        profile_name=args.profile,
    )
    action_config = profile.action_recognition.model_copy(update={"precision": "fp32"})
    profile = profile.model_copy(
        update={
            "device": args.device,
            "detector": profile.detector.model_copy(
                update={"device": args.device, "half": args.device.startswith("cuda")}
            ),
            "action_recognition": action_config,
        }
    )
    model = ActionModelAdapter(action_config, args.device)
    model.ensure_loaded()
    output: list[CalibrationPrediction] = []
    for session in selected:
        clip_id = str(session["id"])
        context, _ = load_context(clip_id, {**session, "expected_actors": []})
        detector = PersonDetector(profile.detector)
        tracker = ByteTrackAdapter(profile.tracker)
        assignment = SeatAssignmentEngine(context, profile.seat_assignment)
        decoder = VideoDecoder(Path(str(session["source"])))
        messages: list[dict[str, Any]] = []
        runtime = ActionRecognitionRuntime(
            session_id=context.session_id,
            runtime_instance_id=uuid.uuid4(),
            config=action_config,
            identity_context=context,
            model=model,
            publish=messages.append,
        )
        start_ms = int(session.get("start_ms", 0))
        end_ms = start_ms + int(session["duration_ms"])
        target_ms = float(start_ms)
        interval_seconds = 1.0 / args.sample_fps
        next_sample = time.monotonic()
        next_progress_ms = start_ms
        decoder.seek(start_ms)
        print(
            json.dumps(
                {
                    "status": "session_started",
                    "session_id": clip_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            ),
            flush=True,
        )
        try:
            while target_ms <= end_ms:
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                packet, _ = decoder.read_for_timestamp(round(target_ms), 0)
                if packet is None:
                    break
                detections = detector.detect(packet.frame)
                tracked = tracker.update(detections, packet.frame.shape[:2], packet.timestamp_ms)
                tracks = tuple(
                    Track(
                        track_id=item.track_id,
                        bbox_norm=normalize_bbox(
                            item.bbox_xyxy, packet.source_width, packet.source_height
                        ),
                        confidence=item.confidence,
                        identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
                    )
                    for item in tracked
                )
                snapshot = assignment.update(tracks, packet.timestamp_ms)
                resolved = tuple(
                    Track(
                        track_id=track.track_id,
                        bbox_norm=track.bbox_norm,
                        confidence=track.confidence,
                        identity=snapshot.identities[track.track_id],
                    )
                    for track in tracks
                )
                runtime.update(packet.frame, resolved, packet.timestamp_ms, 0)
                if packet.timestamp_ms >= next_progress_ms:
                    print(
                        json.dumps(
                            {
                                "status": "session_progress",
                                "session_id": clip_id,
                                "timestamp_ms": packet.timestamp_ms,
                                "end_ms": end_ms,
                            }
                        ),
                        flush=True,
                    )
                    next_progress_ms = packet.timestamp_ms + args.progress_interval_ms
                target_ms += 1000.0 / args.sample_fps
                next_sample = max(next_sample + interval_seconds, time.monotonic())
            if not runtime.wait_idle(timeout=args.idle_timeout):
                raise RuntimeError(f"action runtime did not become idle for {clip_id}")
        finally:
            decoder.release()
            runtime.close(timeout=args.idle_timeout)
        print(
            json.dumps(
                {
                    "status": "session_completed",
                    "session_id": clip_id,
                    "prediction_messages": len(messages),
                }
            ),
            flush=True,
        )
        for message in messages:
            if message.get("type") != "action_prediction":
                continue
            for row in message["predictions"]:
                output.append(
                    CalibrationPrediction(
                        session_id=context.session_id,
                        video_id=str(session.get("video_id", clip_id)),
                        split=str(session["split"]),
                        prediction=_prediction_from_message(row),
                    )
                )
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> list[CalibrationPrediction]:
    output: list[CalibrationPrediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probabilities = tuple(
            float(row[f"prob_{name.replace('/', '_')}"]) for name in R3_CLASS_NAMES
        )
        output.append(
            CalibrationPrediction(
                session_id=uuid.UUID(row["session_id"]),
                video_id=str(row["video_id"]),
                split=str(row["split"]),
                prediction=ActionPrediction(
                    proposal_id=str(row["proposal_id"]),
                    proposal_type=ProposalType(str(row["proposal_type"])),
                    session_candidate_ids=tuple(
                        uuid.UUID(value) for value in row["session_candidate_ids"]
                    ),
                    seat_codes=tuple(str(value) for value in row["seat_codes"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    probabilities=probabilities,
                    predicted_class=str(row["top_class"]),
                    confidence=float(row["top_confidence"]),
                ),
            )
        )
    if not output:
        raise ValueError(f"no predictions found in {path}")
    return output


def _predicted_event_rows(
    events: list[Any],
    truth: list[GroundTruthEvent],
    predictions: list[CalibrationPrediction],
) -> list[dict[str, Any]]:
    video_by_session = {item.session_id: item.video_id for item in predictions}
    seats_by_actors = {
        (item.session_id, tuple(sorted(item.prediction.session_candidate_ids, key=str))): list(
            item.prediction.seat_codes
        )
        for item in predictions
    }
    rows: list[dict[str, Any]] = []
    for event in events:
        candidates: list[tuple[float, GroundTruthEvent]] = []
        for expected in truth:
            if (
                expected.session_id != event.session_id
                or expected.behavior.replace("/", "_") != event.behavior.value
                or frozenset(expected.session_candidate_ids)
                != frozenset(event.session_candidate_ids)
            ):
                continue
            intersection = max(
                0, min(event.end_ms, expected.end_ms) - max(event.start_ms, expected.start_ms)
            )
            union = max(event.end_ms, expected.end_ms) - min(
                event.start_ms, expected.start_ms
            )
            overlap = intersection / union if union > 0 else 0.0
            candidates.append((overlap, expected))
        candidates.sort(key=lambda item: (-item[0], item[1].event_id))
        best = candidates[0] if candidates else None
        actors = tuple(sorted(event.session_candidate_ids, key=str))
        rows.append(
            {
                "session_id": str(event.session_id),
                "video_id": video_by_session[event.session_id],
                "behavior": event.behavior.value,
                "session_candidate_ids": [str(value) for value in actors],
                "seat_codes": seats_by_actors.get((event.session_id, actors), []),
                "proposal_ids": list(event.proposal_ids),
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
                "duration_ms": event.end_ms - event.start_ms,
                "peak_ms": event.peak_ms,
                "peak_probability": event.peak_probability,
                "mean_active_probability": event.ai_confidence,
                "active_prediction_count": event.active_prediction_count,
                "best_gt_event_id": best[1].event_id if best else None,
                "best_gt_temporal_iou": best[0] if best else 0.0,
                "evaluation_match": best is not None and best[0] >= 0.30,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["video_id"], row["start_ms"], row["behavior"], row["proposal_ids"]),
    )


def run(args: argparse.Namespace) -> None:
    payload = _manifest(args.manifest)
    sessions = _resolve_sessions(args.manifest, payload)
    calibration_sessions = [item for item in sessions if item["split"] in CALIBRATION_SPLITS]
    final_sessions = [item for item in sessions if item["split"] not in CALIBRATION_SPLITS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = _ground_truth(calibration_sessions)

    if args.mode in {"collect", "all"}:
        predictions = collect(args, sessions)
        aligned, invalid = align_predictions(predictions, truth)
        _write_jsonl(args.output_dir / "predictions.jsonl", aligned)
        pair_behaviors = {"communicating", "exchange_object"}
        pair_truth_total = 0
        pair_truth_reachable = 0
        for session in calibration_sessions:
            graph = {
                frozenset(str(value) for value in pair)
                for pair in session.get("adjacent_seat_pairs", [])
            }
            for event in session.get("events", []):
                if event["behavior"] not in pair_behaviors:
                    continue
                pair_truth_total += 1
                if frozenset(str(value) for value in event["seat_codes"]) in graph:
                    pair_truth_reachable += 1
        _write_json(
            args.output_dir / "dataset_summary.json",
            {
                "dataset_id": payload.get("dataset_id"),
                "included_splits": sorted(
                    {str(item["split"]) for item in calibration_sessions}
                ),
                "included_sessions": [item["id"] for item in calibration_sessions],
                "included_videos": [item["video_id"] for item in calibration_sessions],
                "excluded_final_test_sessions": [item["id"] for item in final_sessions],
                "excluded_final_test_videos": [item["video_id"] for item in final_sessions],
                "ground_truth_events": len(truth),
                "ground_truth_events_by_behavior": dict(
                    sorted(Counter(item.behavior for item in truth).items())
                ),
                "ground_truth_event_ids": [item.event_id for item in truth],
                "excluded_final_test_event_ids": [
                    str(event["id"])
                    for session in final_sessions
                    for event in session.get("events", [])
                ],
                "pair_ground_truth_events": pair_truth_total,
                "pair_ground_truth_reachable_by_layout_graph": pair_truth_reachable,
                "raw_action_predictions": len(predictions),
                "matched_action_predictions": sum(
                    row["matched_gt_event_id"] is not None for row in aligned
                ),
                "unmatched_action_predictions": sum(
                    row["matched_gt_event_id"] is None for row in aligned
                ),
                "invalid_annotations": invalid,
            },
        )
    else:
        predictions = _load_predictions(args.output_dir / "predictions.jsonl")
        aligned, invalid = align_predictions(predictions, truth)

    if args.mode in {"search", "all"}:
        assert_calibration_only(item.split for item in predictions)
        raw_metrics = classification_metrics(aligned)
        raw_metrics["probability_distributions"] = probability_distributions(aligned)
        raw_metrics["invalid_annotations"] = invalid
        _write_json(args.output_dir / "prediction_metrics.json", raw_metrics)
        search_space = payload.get("search_space") or {
            "alpha": [0.4, 0.6],
            "start_threshold": [0.5, 0.65, 0.8],
            "keep_threshold": [0.35, 0.5],
            "min_active_ms": [1500, 3000],
            "end_grace_ms": [1500, 3000],
            "merge_gap_ms": [1500],
        }
        duration_minutes = sum(int(item["duration_ms"]) for item in calibration_sessions) / 60000
        selected, validation_metrics = bounded_parameter_search(
            predictions,
            truth,
            duration_minutes=duration_minutes,
            search_space=search_space,
        )
        events = aggregate_predictions(predictions, selected)
        _write_jsonl(
            args.output_dir / "predicted_events.jsonl",
            _predicted_event_rows(events, truth, predictions),
        )
        final_metrics = event_metrics(events, truth, duration_minutes=duration_minutes)
        final_metrics["per_behavior_search_metrics"] = validation_metrics
        final_metrics["temporal_match_criterion"] = (
            "same behavior, compatible exact actor set, temporal IoU >= 0.30"
        )
        _write_json(args.output_dir / "event_metrics.json", final_metrics)
        selected_payload = {
            "calibration_dataset_id": payload.get("dataset_id"),
            "tuned_splits": sorted({str(item["split"]) for item in calibration_sessions}),
            "final_test_excluded": True,
            "deployment_approved": False,
            "deployment_note": (
                "Search artifacts never auto-enable production; review event metrics first."
            ),
            "search_space": search_space,
            "event_detection": selected.model_copy(update={"enabled": False}).model_dump(
                mode="json", exclude_none=True
            ),
            "validation_metrics": validation_metrics,
        }
        (args.output_dir / "selected_event_thresholds.yaml").write_text(
            yaml.safe_dump(selected_payload, sort_keys=False), encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("collect", "search", "all"), default="all")
    parser.add_argument("--config-root", type=Path, default=Path("/app/configs"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--profile", choices=("gtx1650", "rtx3060"), default="gtx1650")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-fps", type=float, default=12.5)
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument("--progress-interval-ms", type=int, default=30000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_fps <= 0 or args.idle_timeout <= 0 or args.progress_interval_ms <= 0:
        raise SystemExit("sample-fps, idle-timeout and progress-interval-ms must be positive")
    run(args)


if __name__ == "__main__":
    main()
