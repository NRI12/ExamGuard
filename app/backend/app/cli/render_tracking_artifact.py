from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2  # type: ignore[import-untyped]

from app.monitoring.decoder import VideoDecoder


def nearest_row(
    rows: list[dict[str, str]],
    timestamp_ms: int,
    tolerance_ms: int,
) -> dict[str, str]:
    if not rows:
        raise ValueError("Tracking CSV contains no frames")
    row = min(rows, key=lambda item: abs(int(item["timestamp_ms"]) - timestamp_ms))
    if abs(int(row["timestamp_ms"]) - timestamp_ms) > tolerance_ms:
        raise ValueError("Tracking CSV has no frame within the requested tolerance")
    return row


def render_artifact(args: argparse.Namespace) -> None:
    with args.csv.open(encoding="utf-8", newline="") as stream:
        row = nearest_row(list(csv.DictReader(stream)), args.timestamp_ms, args.tolerance_ms)

    decoder = VideoDecoder(args.video)
    try:
        timestamp_ms = int(row["timestamp_ms"])
        decoder.seek(timestamp_ms)
        packet, _ = decoder.read_for_timestamp(timestamp_ms, 0)
    finally:
        decoder.release()
    if packet is None:
        raise RuntimeError("Video ended before the requested tracking frame")

    frame = packet.frame.copy()
    detections = json.loads(row["detection_boxes"])
    tracks = json.loads(row["track_boxes"])
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (round(value) for value in detection["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 32, 230), 2)
        cv2.putText(
            frame,
            f"D{index} {detection['confidence']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (32, 32, 230),
            2,
        )
    for track in tracks:
        x1, y1, x2, y2 = (round(value) for value in track["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 210, 20), 3)
        cv2.putText(
            frame,
            f"T{track['track_id']} {track['confidence']:.2f}",
            (x1, min(frame.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 210, 20),
            2,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"Could not write tracking artifact: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render YOLO detections and ByteTrack tracks from benchmark CSV",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--timestamp-ms", type=int, required=True)
    parser.add_argument("--tolerance-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.timestamp_ms < 0 or args.tolerance_ms < 0:
        raise SystemExit("--timestamp-ms and --tolerance-ms must be non-negative")
    render_artifact(args)


if __name__ == "__main__":
    main()
