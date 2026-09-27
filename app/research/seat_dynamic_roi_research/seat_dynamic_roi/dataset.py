from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import Detection, DetectionCache, YoloPersonDetector
from .geometry import (
    Box,
    association_score,
    expand_box,
    expand_uniform,
    normalized_to_pixels,
    pixels_to_normalized,
    robust_stable_box,
    union_boxes,
)
from .io_utils import (
    EventRow,
    SeatRow,
    class_slug,
    deterministic_id,
    jsonl_write_line,
    read_video_meta,
    resolve_video_path,
)


def generate_window_starts(
    event_start: float,
    event_end: float,
    video_duration: float,
    window_sec: float,
    stride_sec: float,
    max_windows: int,
) -> list[float]:
    if window_sec <= 0 or stride_sec <= 0 or max_windows <= 0:
        raise ValueError("window/stride/max_windows must be positive")

    if video_duration <= window_sec:
        return [0.0]

    event_duration = event_end - event_start
    max_start = max(0.0, video_duration - window_sec)

    if event_duration <= window_sec:
        center = (event_start + event_end) / 2.0
        return [min(max(center - window_sec / 2.0, 0.0), max_start)]

    starts: list[float] = []
    s = event_start
    last_full = event_end - window_sec
    while s <= last_full + 1e-9:
        starts.append(min(max(s, 0.0), max_start))
        s += stride_sec

    if not starts:
        center = (event_start + event_end) / 2.0
        starts = [min(max(center - window_sec / 2.0, 0.0), max_start)]

    # Deduplicate after clamping.
    starts = sorted(set(round(x, 6) for x in starts))
    if len(starts) <= max_windows:
        return starts

    # Deterministically keep evenly distributed windows, controlling redundancy/event.
    indices = np.linspace(0, len(starts) - 1, num=max_windows)
    chosen = sorted(set(int(round(i)) for i in indices))
    return [starts[i] for i in chosen]


def center_segment_frame_indices(
    start_sec: float,
    end_sec: float,
    fps: float,
    frame_count: int,
    t: int,
) -> list[int]:
    start_frame = int(math.floor(start_sec * fps))
    end_frame = min(frame_count, int(math.ceil(end_sec * fps)))
    end_frame = max(end_frame, start_frame + 1)
    n = end_frame - start_frame
    indices: list[int] = []
    for seg in range(t):
        a = start_frame + (seg * n) / t
        b = start_frame + ((seg + 1) * n) / t
        idx = int(math.floor((a + b) / 2.0))
        idx = max(start_frame, min(idx, end_frame - 1, frame_count - 1))
        indices.append(idx)
    return indices


def read_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames: list[np.ndarray] = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode {video_path} frame={idx}")
            frames.append(frame)
    finally:
        cap.release()
    return frames



def _global_assign_frame(
    detections: list[Detection],
    anchors: dict[str, Box],
    association_cfg: dict,
) -> dict[str, Detection | None]:
    """
    One-to-one greedy assignment for one frame:
    - each seat gets at most one person detection;
    - each person detection is used by at most one seat.

    Greedy is sufficient here because Seat Map is already a strong spatial prior.
    It avoids the important failure mode where one person box is assigned to two
    adjacent seats. If later experiments show association errors, this is the
    place to replace with Hungarian matching without changing the manifest contract.
    """
    assigned: dict[str, Detection | None] = {seat_id: None for seat_id in anchors}
    candidates: list[tuple[float, float, str, int]] = []

    for seat_id, anchor in anchors.items():
        for det_idx, det in enumerate(detections):
            score = association_score(
                det.box,
                anchor,
                float(association_cfg["iou_weight"]),
                float(association_cfg["center_weight"]),
                float(association_cfg["inside_bonus_weight"]),
            )
            if score >= float(association_cfg["min_score"]):
                candidates.append((score, det.confidence, seat_id, det_idx))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    used_seats: set[str] = set()
    used_detections: set[int] = set()

    for _, _, seat_id, det_idx in candidates:
        if seat_id in used_seats or det_idx in used_detections:
            continue
        assigned[seat_id] = detections[det_idx]
        used_seats.add(seat_id)
        used_detections.add(det_idx)

    return assigned


def associate_seats_over_frames(
    per_frame_detections: list[list[Detection]],
    anchors: dict[str, Box],
    association_cfg: dict,
) -> dict[str, list[Detection | None]]:
    out = {seat_id: [] for seat_id in anchors}
    for detections in per_frame_detections:
        frame_assignment = _global_assign_frame(detections, anchors, association_cfg)
        for seat_id in anchors:
            out[seat_id].append(frame_assignment[seat_id])
    return out


def stabilized_activity_roi_from_matches(
    matches: list[Detection | None],
    anchor_px: Box,
    width: int,
    height: int,
    association_cfg: dict,
    local_cfg: dict,
) -> tuple[Box, str, int, list[float]]:
    matched_boxes = [m.box for m in matches if m is not None]
    match_confs = [m.confidence for m in matches if m is not None]

    min_frames = int(association_cfg["min_detection_frames"])
    if len(matched_boxes) < min_frames:
        return anchor_px, "SEAT_FALLBACK", len(matched_boxes), match_confs

    person = robust_stable_box(matched_boxes).clip(width, height)
    activity = expand_box(
        person,
        width,
        height,
        left=float(local_cfg["expand_left"]),
        right=float(local_cfg["expand_right"]),
        top=float(local_cfg["expand_top"]),
        bottom=float(local_cfg["expand_bottom"]),
    )
    return activity, "DETECTED", len(matched_boxes), match_confs

def write_cropped_views(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    views: dict[str, tuple[Box, Path]],
    fps: float,
    width: int,
    height: int,
    codec: str,
) -> dict[str, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writers: dict[str, cv2.VideoWriter] = {}
    written = {name: 0 for name in views}
    fourcc = cv2.VideoWriter_fourcc(*codec)

    try:
        for name, (box, path) in views.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = box.to_int(width, height)
            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                fps,
                (x2 - x1, y2 - y1),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output clip: {path}")
            writers[name] = writer

        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for name, (box, _) in views.items():
                x1, y1, x2, y2 = box.to_int(width, height)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Empty crop for {name} at frame {frame_idx}")
                writers[name].write(crop)
                written[name] += 1
            frame_idx += 1
    finally:
        cap.release()
        for w in writers.values():
            w.release()

    return written


def save_preview(
    frame: np.ndarray,
    target_anchor: Box,
    local_roi: Box,
    interactions: list[tuple[str, Box]],
    seat_id: str,
    path: Path,
) -> None:
    canvas = frame.copy()

    def rect(box: Box, color, thickness=2):
        x1, y1, x2, y2 = box.to_int(canvas.shape[1], canvas.shape[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    rect(target_anchor, (255, 0, 0), 2)   # anchor
    rect(local_roi, (0, 255, 0), 3)       # local dynamic
    for neighbor, box in interactions:
        rect(box, (0, 165, 255), 2)
        x1, y1, _, _ = box.to_int(canvas.shape[1], canvas.shape[0])
        cv2.putText(
            canvas, f"pair:{seat_id}-{neighbor}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )

    x1, y1, _, _ = local_roi.to_int(canvas.shape[1], canvas.shape[0])
    cv2.putText(
        canvas, f"target:{seat_id}", (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


class DatasetBuilder:
    def __init__(
        self,
        events: list[EventRow],
        seats: list[SeatRow],
        cfg: dict[str, Any],
        out_dir: Path,
        project_root: Path | None,
    ):
        self.events = events
        self.seats = seats
        self.cfg = cfg
        self.out_dir = out_dir
        self.project_root = project_root

        self.seat_by_session_participant: dict[tuple[str, str], SeatRow] = {}
        self.seat_by_session_seat: dict[tuple[str, str], SeatRow] = {}
        for s in seats:
            self.seat_by_session_participant[(s.session_id, s.participant_id)] = s
            self.seat_by_session_seat[(s.session_id, s.seat_id)] = s

        self.cache = DetectionCache(out_dir / "detection_cache.sqlite3")
        self.detector = YoloPersonDetector(cfg["detector"], self.cache)
        self.preview_counts = Counter()
        self.summary = Counter()
        self.per_class = Counter()
        self.fallback_per_class = Counter()

    def close(self):
        self.cache.close()

    def _seat(self, session_id: str, participant_id: str) -> SeatRow:
        key = (session_id, participant_id)
        if key not in self.seat_by_session_participant:
            raise KeyError(f"No seat mapping for session={session_id} participant={participant_id}")
        return self.seat_by_session_participant[key]

    def build(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.out_dir / "deployment_roi_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for event in self.events:
                self._process_event(event, mf)

        summary = {
            "total_samples": int(self.summary["samples"]),
            "total_source_events": len({e.source_event_id for e in self.events}),
            "class_samples": dict(sorted(self.per_class.items())),
            "fallback_samples": int(self.summary["fallback_samples"]),
            "fallback_by_class": dict(sorted(self.fallback_per_class.items())),
            "config": self.cfg,
        }
        with open(self.out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _process_event(self, event: EventRow, mf) -> None:
        video_path = resolve_video_path(event.video_path, self.project_root)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        meta = read_video_meta(video_path)
        if event.end_sec > meta.duration_sec + 0.25:
            raise ValueError(
                f"Event {event.source_event_id} end={event.end_sec:.3f}s exceeds "
                f"video duration={meta.duration_sec:.3f}s"
            )

        target_seat = self._seat(event.session_id, event.participant_id)
        if (
            target_seat.global_subject_id
            and event.global_subject_id
            and target_seat.global_subject_id != event.global_subject_id
        ):
            raise ValueError(
                f"Global subject mismatch event={event.source_event_id}: "
                f"{event.global_subject_id} vs seat_map={target_seat.global_subject_id}"
            )

        wc = self.cfg["window"]
        starts = generate_window_starts(
            event.start_sec, event.end_sec, meta.duration_sec,
            float(wc["seconds"]), float(wc["stride_seconds"]),
            int(wc["max_windows_per_event"]),
        )
        for window_idx, start in enumerate(starts):
            end = min(start + float(wc["seconds"]), meta.duration_sec)
            self._process_window(
                event, target_seat, video_path, meta,
                window_idx, start, end, mf,
            )

    def _process_window(
        self,
        event: EventRow,
        target_seat: SeatRow,
        video_path: Path,
        meta,
        window_idx: int,
        start_sec: float,
        end_sec: float,
        mf,
    ) -> None:
        t = int(self.cfg["window"]["detector_sample_frames"])
        frame_indices = center_segment_frame_indices(
            start_sec, end_sec, meta.fps, meta.frame_count, t
        )
        frames = read_frames(video_path, frame_indices)

        stat = video_path.stat()
        video_key = f"{video_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
        detections = self.detector.detect_batch(video_key, frame_indices, frames)

        target_anchor = normalized_to_pixels(
            target_seat.anchor_norm, meta.width, meta.height
        )

        # Build all relevant anchors first, then perform one-to-one person↔seat
        # assignment per frame. This prevents the same detection from being reused
        # by target and an adjacent neighbor.
        max_neighbors = int(self.cfg["interaction_roi"].get("max_neighbors", 4))
        relevant_neighbors: list[SeatRow] = []
        relevant_anchors: dict[str, Box] = {
            target_seat.seat_id: target_anchor,
        }
        for neighbor_seat_id in target_seat.neighbors[:max_neighbors]:
            key = (event.session_id, neighbor_seat_id)
            if key not in self.seat_by_session_seat:
                raise KeyError(
                    f"Seat {target_seat.seat_id} references unknown neighbor "
                    f"{neighbor_seat_id} in session {event.session_id}"
                )
            neighbor = self.seat_by_session_seat[key]
            relevant_neighbors.append(neighbor)
            relevant_anchors[neighbor.seat_id] = normalized_to_pixels(
                neighbor.anchor_norm, meta.width, meta.height
            )

        seat_matches = associate_seats_over_frames(
            detections, relevant_anchors, self.cfg["association"]
        )

        local_roi, local_source, n_match, match_confs = (
            stabilized_activity_roi_from_matches(
                seat_matches[target_seat.seat_id],
                target_anchor,
                meta.width,
                meta.height,
                self.cfg["association"],
                self.cfg["local_roi"],
            )
        )

        neighbor_records = []
        interaction_boxes: list[tuple[str, Box]] = []
        for neighbor in relevant_neighbors:
            neighbor_anchor = relevant_anchors[neighbor.seat_id]
            neighbor_local, neighbor_source, neighbor_n_match, neighbor_confs = (
                stabilized_activity_roi_from_matches(
                    seat_matches[neighbor.seat_id],
                    neighbor_anchor,
                    meta.width,
                    meta.height,
                    self.cfg["association"],
                    self.cfg["local_roi"],
                )
            )
            pair = expand_uniform(
                union_boxes([local_roi, neighbor_local]),
                meta.width, meta.height,
                float(self.cfg["interaction_roi"]["margin_ratio"]),
            )
            interaction_boxes.append((neighbor.seat_id, pair))
            neighbor_records.append({
                "neighbor_seat_id": neighbor.seat_id,
                "neighbor_participant_id": neighbor.participant_id,
                "neighbor_global_subject_id": neighbor.global_subject_id,
                "neighbor_roi_source": neighbor_source,
                "neighbor_detection_frames": neighbor_n_match,
                "neighbor_detection_conf_mean": (
                    float(np.mean(neighbor_confs)) if neighbor_confs else None
                ),
                "interaction_roi_xyxy_px": pair.as_list(),
                "interaction_roi_xyxy_norm": pixels_to_normalized(
                    pair, meta.width, meta.height
                ).as_list(),
            })

        sample_id = deterministic_id(
            event.source_event_id, event.participant_id,
            f"{start_sec:.6f}", f"{end_sec:.6f}",
        )
        slug = class_slug(event.behavior)
        local_rel = Path("clips") / "local" / slug / f"{sample_id}.mp4"

        views: dict[str, tuple[Box, Path]] = {
            "local": (local_roi, self.out_dir / local_rel)
        }
        for neighbor_seat_id, pair in interaction_boxes:
            rel = (
                Path("clips") / "interaction" / slug
                / f"{sample_id}__{neighbor_seat_id}.mp4"
            )
            views[f"interaction:{neighbor_seat_id}"] = (pair, self.out_dir / rel)

        written_frames = {}
        if bool(self.cfg["output"]["write_clips"]):
            written_frames = write_cropped_views(
                video_path, start_sec, end_sec, views,
                meta.fps, meta.width, meta.height,
                str(self.cfg["output"].get("codec", "mp4v")),
            )

        interaction_manifest = []
        for rec in neighbor_records:
            neighbor_seat_id = rec["neighbor_seat_id"]
            rel = (
                Path("clips") / "interaction" / slug
                / f"{sample_id}__{neighbor_seat_id}.mp4"
            )
            rec = dict(rec)
            rec["clip_path"] = str(rel) if bool(self.cfg["output"]["write_clips"]) else None
            rec["written_frames"] = written_frames.get(
                f"interaction:{neighbor_seat_id}"
            )
            interaction_manifest.append(rec)

        row = {
            "sample_id": sample_id,
            "source_event_id": event.source_event_id,
            "window_index_within_event": window_idx,
            "session_id": event.session_id,
            "video_path": str(video_path),
            "participant_id": event.participant_id,
            "global_subject_id": event.global_subject_id,
            "seat_id": target_seat.seat_id,
            "camera_id": target_seat.camera_id,
            "behavior": event.behavior,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "split": event.split,
            "fold": event.fold,
            "source_event_start_sec": event.start_sec,
            "source_event_end_sec": event.end_sec,
            "sampled_frame_indices_for_roi": frame_indices,
            "video_metadata": {
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "width": meta.width,
                "height": meta.height,
                "duration_sec": meta.duration_sec,
            },
            "seat_anchor_xyxy_norm": target_seat.anchor_norm.as_list(),
            "local_roi_source": local_source,
            "local_detection_frames": n_match,
            "local_detection_conf_mean": float(np.mean(match_confs)) if match_confs else None,
            "local_roi_xyxy_px": local_roi.as_list(),
            "local_roi_xyxy_norm": pixels_to_normalized(
                local_roi, meta.width, meta.height
            ).as_list(),
            "local_clip_path": str(local_rel) if bool(self.cfg["output"]["write_clips"]) else None,
            "local_written_frames": written_frames.get("local"),
            "interactions": interaction_manifest,
            "annotation_extras": event.extras,
        }
        jsonl_write_line(mf, row)

        self.summary["samples"] += 1
        self.per_class[event.behavior] += 1
        if local_source == "SEAT_FALLBACK":
            self.summary["fallback_samples"] += 1
            self.fallback_per_class[event.behavior] += 1

        out_cfg = self.cfg["output"]
        if bool(out_cfg.get("save_previews", True)):
            max_prev = int(out_cfg.get("max_previews_per_class", 12))
            if self.preview_counts[event.behavior] < max_prev:
                preview_frame = frames[len(frames) // 2]
                preview_path = (
                    self.out_dir / "previews" / slug
                    / f"{sample_id}.jpg"
                )
                save_preview(
                    preview_frame, target_anchor, local_roi,
                    interaction_boxes, target_seat.seat_id, preview_path
                )
                self.preview_counts[event.behavior] += 1
