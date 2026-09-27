from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import Box


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


class DetectionCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            video_key TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            detector_key TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(video_key, frame_idx, detector_key)
        )
        """)
        self.conn.commit()

    def get(self, video_key: str, frame_idx: int, detector_key: str) -> list[Detection] | None:
        row = self.conn.execute(
            "SELECT payload FROM detections WHERE video_key=? AND frame_idx=? AND detector_key=?",
            (video_key, frame_idx, detector_key),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return [
            Detection(Box(*d["box"]), float(d["confidence"]))
            for d in payload
        ]

    def put(
        self,
        video_key: str,
        frame_idx: int,
        detector_key: str,
        detections: list[Detection],
    ) -> None:
        payload = json.dumps([
            {"box": d.box.as_list(), "confidence": d.confidence}
            for d in detections
        ])
        self.conn.execute(
            "INSERT OR REPLACE INTO detections(video_key, frame_idx, detector_key, payload) VALUES(?,?,?,?)",
            (video_key, frame_idx, detector_key, payload),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class YoloPersonDetector:
    def __init__(self, cfg: dict, cache: DetectionCache):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Missing ultralytics. Install exact research pin, e.g. ultralytics==8.4.147"
            ) from e

        self.model_ref = str(cfg["model"])
        self.device = cfg.get("device", 0)
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.70))
        self.person_class_id = int(cfg.get("person_class_id", 0))
        self.cache = cache
        self.model = YOLO(self.model_ref)
        self.detector_key = (
            f"{self.model_ref}|device={self.device}|imgsz={self.imgsz}|"
            f"conf={self.conf}|iou={self.iou}|class={self.person_class_id}"
        )

    def detect_batch(
        self,
        video_key: str,
        frame_indices: Sequence[int],
        frames: Sequence[np.ndarray],
    ) -> list[list[Detection]]:
        outputs: list[list[Detection] | None] = [None] * len(frames)
        missing_positions: list[int] = []
        missing_frames: list[np.ndarray] = []

        for pos, frame_idx in enumerate(frame_indices):
            cached = self.cache.get(video_key, int(frame_idx), self.detector_key)
            if cached is None:
                missing_positions.append(pos)
                missing_frames.append(frames[pos])
            else:
                outputs[pos] = cached

        if missing_frames:
            results = self.model.predict(
                source=missing_frames,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=[self.person_class_id],
                verbose=False,
            )
            if len(results) != len(missing_frames):
                raise RuntimeError("YOLO returned unexpected batch length")

            for pos, result in zip(missing_positions, results):
                dets: list[Detection] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    for b, c in zip(xyxy, confs):
                        dets.append(Detection(
                            box=Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                            confidence=float(c),
                        ))
                outputs[pos] = dets
                self.cache.put(video_key, int(frame_indices[pos]), self.detector_key, dets)

        return [x or [] for x in outputs]
