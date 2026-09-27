"""Build a Phase 7 manifest from the versioned exam_dataset_pilot metadata."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

SUPPORTED_BEHAVIORS = {
    "normal",
    "suspicious_looking",
    "communicating",
    "exchange_object",
    "using_phone/cheat_sheet",
}
PAIR_BEHAVIORS = {"communicating", "exchange_object"}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required dataset file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _session_splits(rows: list[dict[str, str]]) -> dict[str, str]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[row["session_id"]].add(row["partition"].strip().lower())
    invalid = {session: sorted(splits) for session, splits in values.items() if len(splits) != 1}
    if invalid:
        raise ValueError(f"sessions cross development/final-test partitions: {invalid}")
    return {session: next(iter(splits)) for session, splits in values.items()}


def _seat_layouts(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]], dict[str, list[list[str]]]]:
    seats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actors: dict[str, dict[str, str]] = defaultdict(dict)
    neighbor_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        session = row["session_id"]
        code = row["seat_id"]
        if code in actors[session].values():
            raise ValueError(f"duplicate seat {code} in session {session}")
        actors[session][row["participant_id"]] = code
        seats[session].append(
            {
                "code": code,
                "bbox_norm": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            }
        )
        for neighbor in filter(None, row.get("neighbors", "").split(";")):
            edge = (code, neighbor) if code < neighbor else (neighbor, code)
            neighbor_edges[session].add(edge)

    pairs: dict[str, list[list[str]]] = {}
    for session, edges in neighbor_edges.items():
        known = {seat["code"] for seat in seats[session]}
        missing = sorted({code for edge in edges for code in edge if code not in known})
        if missing:
            raise ValueError(
                f"session {session} neighbor graph references unknown seats: {missing}"
            )
        pairs[session] = [list(edge) for edge in sorted(edges)]
    return dict(seats), dict(actors), pairs


def build_manifest(dataset_root: Path) -> dict[str, Any]:
    manifests = dataset_root / "manifests"
    inventory = _rows(manifests / "video_inventory.csv")
    events = _rows(manifests / "events.csv")
    split_rows = _rows(manifests / "checkpoint5_master_split.csv")
    seat_rows = _rows(dataset_root / "research_roi/inputs/seat_map.csv")

    splits = _session_splits(split_rows)
    seats, actor_seats, adjacent_pairs = _seat_layouts(seat_rows)
    events_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        session = row["session_id"]
        behavior = row["training_label"].strip()
        if behavior not in SUPPORTED_BEHAVIORS:
            raise ValueError(f"unsupported behavior {behavior!r} in event {row['event_id']}")
        try:
            seat_codes = [actor_seats[session][row["participant_id"]]]
        except KeyError as exc:
            raise ValueError(f"event {row['event_id']} has unmapped primary actor") from exc
        related = row.get("related_actor", "").strip()
        if behavior in PAIR_BEHAVIORS:
            if not related:
                raise ValueError(f"pair event {row['event_id']} has no related actor")
            try:
                seat_codes.append(actor_seats[session][related])
            except KeyError as exc:
                raise ValueError(f"event {row['event_id']} has unmapped related actor") from exc
        elif related:
            raise ValueError(f"single event {row['event_id']} unexpectedly has a related actor")
        events_by_session[session].append(
            {
                "id": row["event_id"],
                "behavior": behavior,
                "start_ms": round(float(row["start_sec"]) * 1000),
                "end_ms": round(float(row["end_sec"]) * 1000),
                "seat_codes": seat_codes,
            }
        )

    sessions: list[dict[str, Any]] = []
    for row in sorted(inventory, key=lambda item: item["session_id"]):
        session = row["session_id"]
        if session not in splits or session not in seats:
            raise ValueError(f"session {session} is missing split or seat metadata")
        source = (dataset_root / "normalized" / row["video_name"]).resolve()
        if not source.is_file():
            raise ValueError(f"rebased video path does not exist: {source}")
        sessions.append(
            {
                "id": session,
                "video_id": row["video_name"],
                "source": str(source),
                "split": splits[session],
                "start_ms": 0,
                "duration_ms": round(float(row["duration_sec"]) * 1000),
                "seats": seats[session],
                "adjacent_seat_pairs": adjacent_pairs.get(session, []),
                "events": sorted(
                    events_by_session[session], key=lambda item: (item["start_ms"], item["id"])
                ),
            }
        )

    inventory_sessions = {row["session_id"] for row in inventory}
    orphan_events = sorted(set(events_by_session) - inventory_sessions)
    if orphan_events:
        raise ValueError(f"events reference sessions missing from inventory: {orphan_events}")
    return {
        "dataset_id": "exam_dataset_pilot-checkpoint5-runtime-events",
        "source_dataset_root": str(dataset_root.resolve()),
        "split_policy": "checkpoint5_master_split.partition; final_test is excluded from tuning",
        "search_space": {
            "alpha": [0.4, 0.6],
            "start_threshold": [0.5, 0.65, 0.8, 0.9, 0.95, 0.98],
            "keep_threshold": [0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
            "min_active_ms": [1500, 3000, 4500],
            "end_grace_ms": [0, 1500, 3000],
            "merge_gap_ms": [0, 1500],
        },
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
