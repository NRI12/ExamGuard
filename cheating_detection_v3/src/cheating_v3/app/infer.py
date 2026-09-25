"""Run the detector on a video file, webcam index or RTSP/HTTP stream.

  python -m cheating_v3 infer --source exam.mp4 --out annotated.mp4 --events events.csv
  python -m cheating_v3 infer --source 0 --show
  python -m cheating_v3 infer --source rtsp://cam/stream --show --set app.det_imgsz=1280
"""
import csv
import json
import os
import time

import cv2
import torch
from tqdm import tqdm

from ..config import config_parser, load_config
from .detector import CheatDetector


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="video file, webcam index or stream URL")
    p.add_argument("--checkpoint", help="default: paths.weights")
    p.add_argument("--out", help="annotated output video (.mp4)")
    p.add_argument("--events", help="events CSV")
    p.add_argument("--log", help="per-window JSONL log")
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)

    det = CheatDetector(cfg, args.checkpoint, args.device)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if n and args.max_seconds:
        n = min(n, int(args.max_seconds * fps))
    live = not os.path.isfile(args.source)
    writer, f, t_start = None, 0, time.time()
    bar = tqdm(total=n, unit="fr", desc="frames", mininterval=1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.time() - t_start) if live else f / fps
        if args.max_seconds and t > args.max_seconds:
            break
        det.process(frame, t)
        if args.out or args.show:
            det.draw(frame)
        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
            writer.write(frame)
        if args.show:
            cv2.imshow(os.path.basename(cfg["root"]), frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        f += 1
        bar.update(1)
        bar.set_postfix(tracks=len(det.tracks), alerts=sum(tr.alert for tr in det.tracks.values()),
                        events=len(det.events))
    bar.close()
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    events = det.finish()
    elapsed = time.time() - t_start
    print(f"{f} frames in {elapsed:.1f}s ({f / max(elapsed, 1e-6):.1f} fps), {len(events)} events")
    for e in events:
        print(f"  track #{e['track']:<3} {e['start']:7.1f}s - {e['end']:7.1f}s  {e['type']:17s} peak {e['peak']:.2f}")
    if args.events:
        with open(args.events, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["track", "start", "end", "type", "peak", "windows"])
            wr.writeheader()
            wr.writerows(events)
    if args.log:
        with open(args.log, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in det.log)


if __name__ == "__main__":
    main()
