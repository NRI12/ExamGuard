from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from .geometry import normalized_to_pixels
from .io_utils import load_events, load_seats, read_video_meta, resolve_video_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--seats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    events = load_events(args.events)
    seats = load_seats(args.seats)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root) if args.project_root else None

    by_session_events = defaultdict(list)
    by_session_seats = defaultdict(list)
    for e in events:
        by_session_events[e.session_id].append(e)
    for s in seats:
        by_session_seats[s.session_id].append(s)

    for session_id in sorted(by_session_seats):
        if session_id not in by_session_events:
            print(f"[WARN] no source event to preview session {session_id}")
            continue
        e = by_session_events[session_id][0]
        video_path = resolve_video_path(e.video_path, project_root)
        meta = read_video_meta(video_path)
        t = (e.start_sec + e.end_sec) / 2.0
        frame_idx = min(meta.frame_count - 1, max(0, int(round(t * meta.fps))))

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode preview frame {video_path}:{frame_idx}")

        for s in by_session_seats[session_id]:
            box = normalized_to_pixels(s.anchor_norm, meta.width, meta.height)
            x1, y1, x2, y2 = box.to_int(meta.width, meta.height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{s.seat_id}/{s.participant_id}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        path = out / f"{session_id}.jpg"
        cv2.imwrite(str(path), frame)
        print(path)


if __name__ == "__main__":
    main()
