"""Fit app.box_map: how YOLO full-body boxes relate to the annotated 'upper body + desk' boxes.

Matches YOLO person boxes to CVAT person boxes on training sessions and prints the median offsets to paste
into configs/default.yaml (app.box_map). Re-run for a camera angle very different from DOAN1.

  python -m cheating_v2 calibrate [--sessions S00 S01 S03 S05]
"""
import os

import cv2
import numpy as np

from ..config import config_parser, load_config
from ..data.cvat import load_tracks
from .eval_videos import iou, session_file


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--sessions", nargs="+", default=["S00", "S01", "S03", "S05"])
    p.add_argument("--frames-per-session", type=int, default=12)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    from ultralytics import YOLO
    det = YOLO(cfg["app"]["detector"])
    rows = []
    for short in args.sessions:
        sess = session_file(cfg, short)
        n, tracks = load_tracks(cfg["paths"]["raw_data"], sess)
        persons = [t for t in tracks if t["label"] == "person"]
        cap = cv2.VideoCapture(os.path.join(cfg["paths"]["videos"], f"{sess}.mp4"))
        for f in np.linspace(100, n - 100, args.frames_per_session).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            yb = det(img, classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.cpu().numpy()
            for pt in persons:
                g = pt["boxes"].get(f)
                if g is None or not len(yb):
                    continue
                # horizontal overlap only: YOLO boxes include the legs, annotated boxes do not
                s, y = max(((iou([y[0], g[1], y[2], g[3]], g), y) for y in yb), key=lambda c: c[0])
                if s >= 0.5:
                    w = y[2] - y[0]
                    rows.append(((g[0] - y[0]) / w, (g[2] - y[2]) / w, (g[1] - y[1]) / w, (g[3] - g[1]) / w))
        cap.release()
    a = np.array(rows)
    med = np.median(a, 0)
    print(f"matched {len(a)} boxes; paste into app.box_map:")
    print(f"  dx1: {med[0]:.3f}\n  dx2: {med[1]:.3f}\n  dy1: {med[2]:.3f}\n  h: {med[3]:.3f}")


if __name__ == "__main__":
    main()
