from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import cv2
import yaml  # type: ignore[import-untyped]

NormalizedRect = tuple[float, float, float, float]


def _iou(left: NormalizedRect, right: NormalizedRect) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _source_dimensions(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot read source dimensions: {path}")
    return width, height


def _clip_key(clip_id: str) -> str:
    parts = clip_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Clip id must start with clip_<letter>: {clip_id}")
    return "_".join(parts[:2])


def evaluate_csv(
    csv_path: Path,
    regions: dict[str, NormalizedRect],
    width: int,
    height: int,
    mode: str,
) -> dict[str, int | float | None]:
    previous: dict[str, tuple[int, str]] = {}
    actor_owner: dict[str, str] = {}
    matched_samples = 0
    raw_transitions = 0
    logical_switches = 0
    recovered_transitions = 0
    wrong_actor_region_links: set[tuple[str, str, str]] = set()

    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_boxes = json.loads(row["track_boxes"])
            actor_ids = row.get("actor_ids", "").split()
            tracks: list[tuple[int, str, NormalizedRect]] = []
            for index, raw in enumerate(raw_boxes):
                track_id = int(raw["track_id"])
                actor_id = (
                    actor_ids[index]
                    if mode != "raw" and index < len(actor_ids)
                    else f"T{track_id}"
                )
                box = raw["bbox"]
                normalized = (
                    float(box[0]) / width,
                    float(box[1]) / height,
                    float(box[2]) / width,
                    float(box[3]) / height,
                )
                tracks.append((track_id, actor_id, normalized))

            candidates = sorted(
                (
                    (_iou(region, box), region_id, track_id, actor_id)
                    for region_id, region in regions.items()
                    for track_id, actor_id, box in tracks
                ),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            matched: dict[str, tuple[int, str]] = {}
            used_tracks: set[int] = set()
            for overlap, region_id, track_id, actor_id in candidates:
                if overlap < 0.10 or region_id in matched or track_id in used_tracks:
                    continue
                matched[region_id] = (track_id, actor_id)
                used_tracks.add(track_id)

            active_raw_ids = {track_id for track_id, _, _ in tracks}
            for region_id, (track_id, actor_id) in matched.items():
                matched_samples += 1
                prior = previous.get(region_id)
                if prior is not None:
                    prior_track_id, prior_actor_id = prior
                    fragmented = track_id != prior_track_id and prior_track_id not in active_raw_ids
                    if fragmented:
                        raw_transitions += 1
                        if actor_id == prior_actor_id:
                            recovered_transitions += 1
                        else:
                            logical_switches += 1
                owner = actor_owner.get(actor_id)
                if owner is not None and owner != region_id:
                    wrong_actor_region_links.add((actor_id, owner, region_id))
                actor_owner.setdefault(actor_id, region_id)
                previous[region_id] = (track_id, actor_id)

    return {
        "matched_samples": matched_samples,
        "raw_fragmentation_transitions": raw_transitions,
        "logical_id_switches": logical_switches,
        "recovered_transitions": recovered_transitions,
        "fragmentation_recovery_rate": (
            recovered_transitions / raw_transitions if raw_transitions else None
        ),
        "wrong_actor_region_links": len(wrong_actor_region_links),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    clips = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(clips, list):
        raise ValueError("Manifest must contain a clips list")
    results: dict[str, Any] = {}
    for mode in args.modes:
        mode_rows: list[dict[str, Any]] = []
        for raw_clip in clips:
            clip = dict(raw_clip)
            clip_id = str(clip["id"])
            source = Path(str(clip["source"]))
            if not source.is_absolute():
                source = (args.manifest.parent / source).resolve()
            width, height = _source_dimensions(source)
            regions = {
                str(item["seat_code"]): cast(
                    NormalizedRect, tuple(float(value) for value in item["bbox_norm"])
                )
                for item in clip["expected_actors"]
            }
            csv_path = args.benchmark_root / f"{_clip_key(clip_id)}_{mode}" / "frames_or_tracks.csv"
            metrics = evaluate_csv(csv_path, regions, width, height, mode)
            mode_rows.append({"clip_id": clip_id, **metrics})
        totals: dict[str, Any] = {
            key: sum(int(row[key]) for row in mode_rows)
            for key in (
                "matched_samples",
                "raw_fragmentation_transitions",
                "logical_id_switches",
                "recovered_transitions",
                "wrong_actor_region_links",
            )
        }
        totals["fragmentation_recovery_rate"] = (
            totals["recovered_transitions"] / totals["raw_fragmentation_transitions"]
            if totals["raw_fragmentation_transitions"]
            else None
        )
        results[mode] = {"clips": mode_rows, "aggregate": totals}
    return {
        "annotation_method": "Static human-calibrated actor regions; not dense MOT ground truth",
        "minimum_match_iou": 0.10,
        "modes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw/logical identities by actor regions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("raw", "motion", "sparse_reid"),
        default=("raw", "motion", "sparse_reid"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
