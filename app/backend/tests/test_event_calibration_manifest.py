from __future__ import annotations

import csv
from pathlib import Path

from app.cli.build_event_calibration_manifest import build_manifest


def _csv(path: Path, fieldnames: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_build_manifest_rebases_paths_preserves_split_and_2d_neighbors(tmp_path: Path) -> None:
    (tmp_path / "normalized").mkdir()
    (tmp_path / "normalized/S00.mp4").touch()
    (tmp_path / "normalized/S05.mp4").touch()
    _csv(
        tmp_path / "manifests/video_inventory.csv",
        ["session_id", "video_name", "duration_sec"],
        [["S00", "S00.mp4", 12.5], ["S05", "S05.mp4", 20]],
    )
    _csv(
        tmp_path / "manifests/checkpoint5_master_split.csv",
        ["session_id", "partition"],
        [["S00", "development"], ["S05", "final_test"]],
    )
    _csv(
        tmp_path / "research_roi/inputs/seat_map.csv",
        [
            "session_id",
            "seat_id",
            "participant_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "neighbors",
        ],
        [
            ["S00", "A1", "P1", 0.1, 0.1, 0.3, 0.8, "A2"],
            ["S00", "A2", "P2", 0.4, 0.1, 0.6, 0.8, "A1"],
            ["S05", "A1", "P7", 0.1, 0.1, 0.3, 0.8, ""],
        ],
    )
    _csv(
        tmp_path / "manifests/events.csv",
        [
            "event_id",
            "session_id",
            "training_label",
            "participant_id",
            "related_actor",
            "start_sec",
            "end_sec",
        ],
        [
            ["event-1", "S00", "communicating", "P1", "P2", 1.2, 4.8],
            ["event-2", "S05", "normal", "P7", "", 5, 9],
        ],
    )

    manifest = build_manifest(tmp_path)

    development, final_test = manifest["sessions"]
    assert development["split"] == "development"
    assert development["source"] == str((tmp_path / "normalized/S00.mp4").resolve())
    assert development["duration_ms"] == 12500
    assert development["adjacent_seat_pairs"] == [["A1", "A2"]]
    assert development["events"][0]["seat_codes"] == ["A1", "A2"]
    assert final_test["split"] == "final_test"
