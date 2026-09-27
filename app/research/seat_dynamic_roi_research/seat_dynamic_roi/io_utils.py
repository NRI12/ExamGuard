from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from .geometry import Box


CANONICAL_CLASSES = {
    "normal",
    "suspicious_looking",
    "communicating",
    "exchange_object",
    "using_phone/cheat_sheet",
}


@dataclass(frozen=True)
class EventRow:
    source_event_id: str
    session_id: str
    video_path: str
    participant_id: str
    global_subject_id: str
    behavior: str
    start_sec: float
    end_sec: float
    split: str
    fold: str
    extras: dict[str, str]


@dataclass(frozen=True)
class SeatRow:
    session_id: str
    camera_id: str
    seat_id: str
    participant_id: str
    global_subject_id: str
    anchor_norm: Box
    neighbors: tuple[str, ...]


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _required(row: dict[str, str], keys: list[str], kind: str, line_no: int) -> None:
    missing = [k for k in keys if not str(row.get(k, "")).strip()]
    if missing:
        raise ValueError(f"{kind} line {line_no}: missing {missing}")


def load_events(path: str | Path) -> list[EventRow]:
    required = [
        "source_event_id", "session_id", "video_path", "participant_id",
        "global_subject_id", "behavior", "start_sec", "end_sec", "split",
    ]
    rows: list[EventRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            _required(row, required, "events", i)
            behavior = row["behavior"].strip()
            if behavior not in CANONICAL_CLASSES:
                raise ValueError(f"events line {i}: invalid behavior={behavior!r}")
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            if start < 0 or end <= start:
                raise ValueError(f"events line {i}: invalid interval {start}..{end}")
            known = set(required) | {"fold"}
            extras = {k: (v or "") for k, v in row.items() if k not in known}
            rows.append(EventRow(
                source_event_id=row["source_event_id"].strip(),
                session_id=row["session_id"].strip(),
                video_path=row["video_path"].strip(),
                participant_id=row["participant_id"].strip(),
                global_subject_id=row["global_subject_id"].strip(),
                behavior=behavior,
                start_sec=start,
                end_sec=end,
                split=row["split"].strip(),
                fold=(row.get("fold") or "").strip(),
                extras=extras,
            ))
    return rows


def load_seats(path: str | Path) -> list[SeatRow]:
    required = [
        "session_id", "camera_id", "seat_id", "participant_id",
        "global_subject_id", "x1", "y1", "x2", "y2",
    ]
    rows: list[SeatRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            _required(row, required, "seat_map", i)
            vals = [float(row[k]) for k in ["x1", "y1", "x2", "y2"]]
            if not all(0.0 <= v <= 1.0 for v in vals):
                raise ValueError(f"seat_map line {i}: normalized coords must be [0,1]")
            x1, y1, x2, y2 = vals
            if not x1 < x2 or not y1 < y2:
                raise ValueError(f"seat_map line {i}: invalid xyxy")
            neighbors = tuple(
                s.strip() for s in (row.get("neighbors") or "").split(";") if s.strip()
            )
            rows.append(SeatRow(
                session_id=row["session_id"].strip(),
                camera_id=row["camera_id"].strip(),
                seat_id=row["seat_id"].strip(),
                participant_id=row["participant_id"].strip(),
                global_subject_id=row["global_subject_id"].strip(),
                anchor_norm=Box(x1, y1, x2, y2),
                neighbors=neighbors,
            ))
    return rows


def resolve_video_path(video_path: str, project_root: str | Path | None) -> Path:
    p = Path(video_path)
    if p.is_absolute():
        return p
    if project_root is None:
        return p.resolve()
    return (Path(project_root) / p).resolve()


def read_video_meta(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return VideoMeta(fps, count, width, height, count / fps)


def deterministic_id(*parts: object, length: int = 20) -> str:
    data = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:length]


def jsonl_write_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def class_slug(name: str) -> str:
    return name.replace("/", "_")
