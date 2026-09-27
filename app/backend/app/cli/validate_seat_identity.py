from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.ai.detector.yolo import PersonDetector
from app.ai.domain import Track, normalize_bbox
from app.ai.seat_identity.assignment import SeatAssignmentEngine
from app.ai.seat_identity.geometry import intersection_area, rect_area
from app.ai.seat_identity.types import (
    AssignmentState,
    NormalizedRect,
    SeatCandidateBinding,
    SeatDefinition,
    SeatIdentityContext,
    TrackIdentity,
)
from app.ai.tracker.bytetrack import ByteTrackAdapter
from app.monitoring.config import load_runtime_profile_from_paths
from app.monitoring.decoder import VideoDecoder


@dataclass(frozen=True, slots=True)
class ExpectedActor:
    seat_code: str
    bbox_norm: NormalizedRect


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def bbox_iou(left: NormalizedRect, right: NormalizedRect) -> float:
    intersection = intersection_area(left, right)
    union = rect_area(left) + rect_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _uuid(namespace: uuid.UUID, value: str) -> uuid.UUID:
    return uuid.uuid5(namespace, value)


def load_context(
    clip_id: str,
    clip: dict[str, Any],
) -> tuple[SeatIdentityContext, list[ExpectedActor]]:
    seats = tuple(
        SeatDefinition(
            id=_uuid(uuid.NAMESPACE_URL, f"{clip_id}/seat/{item['code']}"),
            code=str(item["code"]),
            bbox_norm=tuple(float(value) for value in item["bbox_norm"]),  # type: ignore[arg-type]
        )
        for item in clip["seats"]
    )
    bindings = tuple(
        SeatCandidateBinding(
            seat_id=seat.id,
            session_candidate_id=_uuid(uuid.NAMESPACE_URL, f"{clip_id}/assignment/{seat.code}"),
            candidate_id=_uuid(uuid.NAMESPACE_URL, f"{clip_id}/candidate/{seat.code}"),
            candidate_code=f"VALIDATION-{seat.code}",
            candidate_name=f"Validation candidate {seat.code}",
        )
        for seat in seats
    )
    seats_by_code = {seat.code: seat for seat in seats}
    adjacent_pairs: list[tuple[uuid.UUID, uuid.UUID]] = []
    seen_pairs: set[frozenset[uuid.UUID]] = set()
    for raw_pair in clip.get("adjacent_seat_pairs", []):
        if not isinstance(raw_pair, list | tuple) or len(raw_pair) != 2:
            raise ValueError("adjacent_seat_pairs entries must contain exactly two seat codes")
        left_code, right_code = (str(value) for value in raw_pair)
        try:
            left = seats_by_code[left_code]
            right = seats_by_code[right_code]
        except KeyError as exc:
            raise ValueError(f"adjacent pair references unknown seat: {exc.args[0]}") from exc
        key = frozenset((left.id, right.id))
        if left.id == right.id or key in seen_pairs:
            raise ValueError(f"invalid or duplicate adjacent pair: {left_code}, {right_code}")
        seen_pairs.add(key)
        adjacent_pairs.append((left.id, right.id))
    actors = [
        ExpectedActor(
            seat_code=str(item["seat_code"]),
            bbox_norm=tuple(float(value) for value in item["bbox_norm"]),  # type: ignore[arg-type]
        )
        for item in clip["expected_actors"]
    ]
    return (
        SeatIdentityContext(
            session_id=_uuid(uuid.NAMESPACE_URL, f"{clip_id}/session"),
            seats=seats,
            bindings=bindings,
            adjacent_seat_pairs=tuple(adjacent_pairs),
        ),
        actors,
    )


def match_expected_actors(
    actors: list[ExpectedActor],
    tracks: tuple[Track, ...],
) -> dict[str, Track]:
    candidates = sorted(
        (
            (bbox_iou(actor.bbox_norm, track.bbox_norm), actor.seat_code, track)
            for actor in actors
            for track in tracks
        ),
        key=lambda item: (-item[0], item[1], item[2].track_id),
    )
    matched: dict[str, Track] = {}
    used_tracks: set[int] = set()
    for overlap, seat_code, track in candidates:
        if overlap < 0.10 or seat_code in matched or track.track_id in used_tracks:
            continue
        matched[seat_code] = track
        used_tracks.add(track.track_id)
    return matched


def validate_clip(
    clip: dict[str, Any],
    *,
    profile: Any,
    sample_fps: float,
) -> dict[str, Any]:
    clip_id = str(clip["id"])
    video = Path(str(clip["source"]))
    context, actors = load_context(clip_id, clip)
    detector = PersonDetector(profile.detector)
    tracker = ByteTrackAdapter(profile.tracker)
    assignment = SeatAssignmentEngine(context, profile.seat_assignment)
    decoder = VideoDecoder(video)
    start_ms = int(clip.get("start_ms", 0))
    end_ms = start_ms + int(clip["duration_ms"])
    interval_ms = 1000 / sample_fps
    decoder.seek(start_ms)

    expected_samples = 0
    correct_samples = 0
    wrong_samples = 0
    unassigned_samples = 0
    extra_track_samples = 0
    false_extra_assignments = 0
    assignment_latencies: list[float] = []
    first_seen: dict[str, int] = {}
    stabilized: set[str] = set()
    previous_tracks: dict[str, int] = {}
    pending_recovery: dict[str, int] = {}
    recovery_delays: list[float] = []
    stabilization_delays: list[float] = []
    recovery_attempts = 0
    recovery_successes = 0
    frames = 0
    target_ms = float(start_ms)
    last_snapshot = None

    try:
        while target_ms <= end_ms:
            packet, _ = decoder.read_for_timestamp(round(target_ms), 0)
            if packet is None:
                break
            detections = detector.detect(packet.frame)
            tracked = tracker.update(detections, packet.frame.shape[:2], packet.timestamp_ms)
            tracks = tuple(
                Track(
                    track_id=item.track_id,
                    bbox_norm=normalize_bbox(
                        item.bbox_xyxy,
                        packet.source_width,
                        packet.source_height,
                    ),
                    confidence=item.confidence,
                    identity=TrackIdentity(state=AssignmentState.UNASSIGNED),
                )
                for item in tracked
            )
            started = time.perf_counter()
            snapshot = assignment.update(tracks, packet.timestamp_ms)
            assignment_latencies.append((time.perf_counter() - started) * 1000)
            resolved = tuple(
                Track(
                    track_id=track.track_id,
                    bbox_norm=track.bbox_norm,
                    confidence=track.confidence,
                    identity=snapshot.identities[track.track_id],
                )
                for track in tracks
            )
            matched = match_expected_actors(actors, resolved)
            used_track_ids = {track.track_id for track in matched.values()}
            active_track_ids = {track.track_id for track in resolved}
            for actor in actors:
                expected_samples += 1
                track = matched.get(actor.seat_code)
                if track is None:
                    unassigned_samples += 1
                    continue
                first_seen.setdefault(actor.seat_code, packet.timestamp_ms)
                previous = previous_tracks.get(actor.seat_code)
                if (
                    previous is not None
                    and previous != track.track_id
                    and previous not in active_track_ids
                ):
                    recovery_attempts += 1
                    pending_recovery[actor.seat_code] = packet.timestamp_ms
                previous_tracks[actor.seat_code] = track.track_id
                actual = (
                    track.identity.seat_code
                    if track.identity.state == AssignmentState.ASSIGNED
                    else None
                )
                if actual == actor.seat_code:
                    correct_samples += 1
                    if actor.seat_code not in stabilized:
                        stabilization_delays.append(
                            float(packet.timestamp_ms - first_seen[actor.seat_code])
                        )
                        stabilized.add(actor.seat_code)
                    if actor.seat_code in pending_recovery:
                        recovery_successes += 1
                        recovery_delays.append(
                            float(packet.timestamp_ms - pending_recovery.pop(actor.seat_code))
                        )
                elif actual is None:
                    unassigned_samples += 1
                else:
                    wrong_samples += 1
            for track in resolved:
                if track.track_id in used_track_ids:
                    continue
                extra_track_samples += 1
                if track.identity.state == AssignmentState.ASSIGNED:
                    false_extra_assignments += 1
            last_snapshot = snapshot
            frames += 1
            target_ms += interval_ms
    finally:
        decoder.release()

    evaluated_samples = expected_samples + extra_track_samples
    false_assignments = wrong_samples + false_extra_assignments
    return {
        "id": clip_id,
        "source": str(video),
        "scenarios": clip.get("scenarios", []),
        "frames_analyzed": frames,
        "expected_actor_samples": expected_samples,
        "correct_actor_samples": correct_samples,
        "wrong_actor_samples": wrong_samples,
        "unassigned_actor_samples": unassigned_samples,
        "extra_track_samples": extra_track_samples,
        "false_extra_assignments": false_extra_assignments,
        "correct_seat_assignment_rate": (
            correct_samples / expected_samples if expected_samples else None
        ),
        "wrong_seat_rate": false_assignments / evaluated_samples if evaluated_samples else None,
        "unassigned_rate": unassigned_samples / expected_samples if expected_samples else None,
        "false_seat_switch_count": last_snapshot.seat_switches if last_snapshot else 0,
        "engine_identity_recoveries": last_snapshot.identity_recoveries if last_snapshot else 0,
        "mean_stabilization_delay_ms": (
            statistics.fmean(stabilization_delays) if stabilization_delays else None
        ),
        "p95_stabilization_delay_ms": percentile(stabilization_delays, 0.95),
        "fragmentation_recovery_attempts": recovery_attempts,
        "fragmentation_recovery_successes": recovery_successes,
        "fragmentation_recovery_success_rate": (
            recovery_successes / recovery_attempts if recovery_attempts else None
        ),
        "mean_recovery_delay_ms": statistics.fmean(recovery_delays) if recovery_delays else None,
        "seat_assignment_ms": {
            "mean": statistics.fmean(assignment_latencies) if assignment_latencies else None,
            "p95": percentile(assignment_latencies, 0.95),
        },
        "_stabilization_delays": stabilization_delays,
        "_recovery_delays": recovery_delays,
        "_seat_assignment_latencies": assignment_latencies,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("clips"), list):
        raise ValueError("Validation manifest must contain a clips list")
    profile = load_runtime_profile_from_paths(
        config_root=args.config_root,
        model_root=args.model_root,
        profile_name=args.profile,
    )
    if args.device != profile.detector.device:
        profile = profile.model_copy(
            update={
                "device": args.device,
                "detector": profile.detector.model_copy(
                    update={"device": args.device, "half": args.device.startswith("cuda")}
                ),
            }
        )
    clips = []
    for raw_clip in manifest["clips"]:
        clip = dict(raw_clip)
        source = Path(str(clip["source"]))
        if not source.is_absolute():
            clip["source"] = str((args.manifest.parent / source).resolve())
        clips.append(validate_clip(clip, profile=profile, sample_fps=args.sample_fps))
    stabilization_delays = [value for clip in clips for value in clip.pop("_stabilization_delays")]
    recovery_delays = [value for clip in clips for value in clip.pop("_recovery_delays")]
    assignment_latencies = [
        value for clip in clips for value in clip.pop("_seat_assignment_latencies")
    ]
    totals = {
        key: sum(int(clip[key]) for clip in clips)
        for key in (
            "expected_actor_samples",
            "correct_actor_samples",
            "wrong_actor_samples",
            "unassigned_actor_samples",
            "extra_track_samples",
            "false_extra_assignments",
            "false_seat_switch_count",
            "engine_identity_recoveries",
            "fragmentation_recovery_attempts",
            "fragmentation_recovery_successes",
        )
    }
    evaluated = totals["expected_actor_samples"] + totals["extra_track_samples"]
    report = {
        "validation_set": manifest.get("validation_set"),
        "profile": profile.profile,
        "device": profile.detector.device,
        "sample_fps": args.sample_fps,
        "clips": clips,
        "aggregate": {
            **totals,
            "correct_seat_assignment_rate": (
                totals["correct_actor_samples"] / totals["expected_actor_samples"]
                if totals["expected_actor_samples"]
                else None
            ),
            "wrong_seat_rate": (
                (totals["wrong_actor_samples"] + totals["false_extra_assignments"]) / evaluated
                if evaluated
                else None
            ),
            "unassigned_rate": (
                totals["unassigned_actor_samples"] / totals["expected_actor_samples"]
                if totals["expected_actor_samples"]
                else None
            ),
            "mean_stabilization_delay_ms": (
                statistics.fmean(stabilization_delays) if stabilization_delays else None
            ),
            "p95_stabilization_delay_ms": percentile(stabilization_delays, 0.95),
            "fragmentation_recovery_success_rate": (
                totals["fragmentation_recovery_successes"]
                / totals["fragmentation_recovery_attempts"]
                if totals["fragmentation_recovery_attempts"]
                else None
            ),
            "mean_recovery_delay_ms": (
                statistics.fmean(recovery_delays) if recovery_delays else None
            ),
            "p95_recovery_delay_ms": percentile(recovery_delays, 0.95),
            "seat_assignment_ms": {
                "mean": (statistics.fmean(assignment_latencies) if assignment_latencies else None),
                "p95": percentile(assignment_latencies, 0.95),
            },
        },
        "metric_definition": (
            "Expected actor regions are human-calibrated evaluation annotations. "
            "The production YOLO, ByteTrack and SeatAssignmentEngine are used unchanged."
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate production Seat-Stable Identity")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("gtx1650", "rtx3060"), default="gtx1650")
    parser.add_argument("--config-root", type=Path, default=Path("/app/configs"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
