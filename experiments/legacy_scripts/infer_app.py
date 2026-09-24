"""Real-time exam-cheating detection app.

Pipeline (per camera):
  frame --(every 0.2 s)--> YOLO person detection + ByteTrack --> box mapped to the training crop geometry
        --> per-track history of the last 16 samples (3.2 s)
        --(every --infer-every s)--> one batched X3D forward for all tracks --> softmax
        --> temporal smoothing (mean of last K windows) --> cheat_score = 1 - P(normal)
        --> hysteresis alert + behaviour type (argmax of cheat classes) --> overlay / events CSV / JSONL log

Cost per camera: YOLO at 5 fps + X3D-L once per second per person.

  python infer_app.py --source exam.mp4 --ckpt best.pt --out annotated.mp4 --events events.csv
  python infer_app.py --source 0 --ckpt best.pt --show                 # webcam
  python infer_app.py --source rtsp://user:pass@cam/stream --ckpt best.pt --show
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_data import gpu_eval_transform  # noqa: E402

# YOLO full-body box -> CVAT-style "upper body + desk" box, as fractions of the YOLO box width.
# Fitted on 288 matched boxes from train sessions S00/S01/S03/S05 (median values).
BOX_MAP = dict(dx1=-0.101, dx2=0.094, dy1=-0.131, h=1.021)
VARIANTS = {"x3d_xs": (4, 160), "x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}
CROP_PAD = 0.15      # same padding as build_clips.py
CLIP_SIZE = 356      # training clips (clips_hr) were stored at 356x356
SAMPLE_DT = 0.2      # training stride: 5 frames @ 25 fps
COLORS = {0: (80, 200, 80), 1: (0, 165, 255), 2: (255, 120, 0), 3: (60, 60, 230)}  # BGR per class


def map_box(b, w_img, h_img):
    x1, y1, x2, _ = b
    w = x2 - x1
    nx1, nx2 = x1 + BOX_MAP["dx1"] * w, x2 + BOX_MAP["dx2"] * w
    ny1 = y1 + BOX_MAP["dy1"] * w
    ny2 = ny1 + BOX_MAP["h"] * w
    return np.clip([nx1, ny1, nx2, ny2], 0, [w_img, h_img, w_img, h_img])


def square_crop_box(boxes, w, h):
    b = np.stack(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x2 - x1, y2 - y1) * (1 + CROP_PAD), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return x1, y1, x1 + int(side), y1 + int(side)


class X3DClassifier:
    """Our X3D checkpoint (train_x3d.py): 16 frames 0.2 s apart, 356 px crop -> 312 centre crop."""

    def __init__(self, ckpt_path, device):
        from pytorchvideo.models import hub
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        name = ck["args"]["model"]
        m = getattr(hub, name)(pretrained=False)
        m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, len(ck["classes"]))
        m.load_state_dict(ck["model"])
        self.model, self.classes = m.to(device).eval(), ck["classes"]
        self.n_frames, self.size = VARIANTS[name]
        self.clip_size = CLIP_SIZE
        assert abs(ck["args"]["stride"] / 25 - SAMPLE_DT) < 1e-6, "model trained with a different frame stride"

    def prepare(self, x):
        return gpu_eval_transform(x, self.size)


def load_classifier(spec, device):
    """best.pt from train_x3d.py (ver 1: 4 classes, ver 2: normal / abnormal)."""
    return X3DClassifier(spec, device)


class Track:
    def __init__(self, tid):
        self.id = tid
        self.boxes = {}                     # sample index -> mapped box
        self.last_seen = -1
        self.probs = collections.deque()    # recent window probabilities
        self.smoothed = None
        self.alert = False
        self.event = None


class CheatDetector:
    def __init__(self, ckpt, det_weights="yolo11m.pt", device="cuda", det_conf=0.3, det_imgsz=640, infer_every=1.0,
                 smooth=3, thr_on=0.85, thr_off=0.7, min_windows=2, min_coverage=0.75):
        from ultralytics import YOLO
        self.device = device
        self.clf = load_classifier(ckpt, device)
        self.classes, self.n_frames = self.clf.classes, self.clf.n_frames
        self.det = YOLO(det_weights)
        self.det_conf, self.det_imgsz = det_conf, det_imgsz
        self.infer_every_samples = max(1, round(infer_every / SAMPLE_DT))
        self.smooth, self.thr_on, self.thr_off, self.min_windows = smooth, thr_on, thr_off, min_windows
        self.min_boxes = int(np.ceil(min_coverage * self.n_frames))
        self.reset()

    def reset(self):
        self.tracks = {}
        self.frames = collections.OrderedDict()   # sample index -> RGB frame
        self.events, self.log = [], []
        self.det.predictor = None                  # fresh tracker state
        self.sample = -1

    # ------------------------------------------------------------------ per frame
    def process(self, frame_bgr, t):
        """Feed every frame; detection/inference only run on the sampled ones. Returns True if sampled."""
        s = int(t / SAMPLE_DT + 1e-6)
        if s == self.sample:
            return False
        self.sample = s
        h, w = frame_bgr.shape[:2]
        r = self.det.track(frame_bgr, persist=True, classes=[0], conf=self.det_conf, imgsz=self.det_imgsz,
                           tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                tr = self.tracks.setdefault(tid, Track(tid))
                tr.boxes[s] = map_box(box, w, h)
                tr.last_seen = s
        self.frames[s] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        while len(self.frames) > self.n_frames:
            self.frames.popitem(last=False)
        if s % self.infer_every_samples == 0 and len(self.frames) == self.n_frames:
            self._infer(s, t, w, h)
        self._expire(s, t)
        return True

    @torch.no_grad()
    def _infer(self, s, t, w, h):
        window = list(self.frames)
        batch, ready = [], []
        for tr in self.tracks.values():
            have = [i for i in window if i in tr.boxes]
            if len(have) < self.min_boxes:
                continue
            boxes = [tr.boxes[min(have, key=lambda k: abs(k - i))] for i in window]
            x1, y1, x2, y2 = square_crop_box(boxes, w, h)
            cs = self.clf.clip_size
            clip = np.stack([cv2.resize(self.frames[i][y1:y2, x1:x2], (cs, cs),
                                        interpolation=cv2.INTER_AREA) for i in window])
            batch.append(clip)
            ready.append(tr)
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).to(self.device, non_blocking=True)
        x = self.clf.prepare(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            p = self.clf.model(x).float().softmax(-1).cpu().numpy()
        for tr, pi in zip(ready, p):
            tr.probs.append(pi)
            while len(tr.probs) > self.smooth:
                tr.probs.popleft()
            tr.smoothed = np.mean(tr.probs, 0)
            score = 1 - tr.smoothed[0]
            label = int(np.argmax(tr.smoothed[1:]) + 1)
            if not tr.alert and score >= self.thr_on:
                tr.alert = True
                tr.event = dict(track=tr.id, start=t, end=t, types=collections.Counter(), peak=0.0, windows=0)
            elif tr.alert and score < self.thr_off:
                self._close(tr)
            if tr.alert:
                tr.event["end"] = t
                tr.event["windows"] += 1
                tr.event["types"][label] += 1
                tr.event["peak"] = max(tr.event["peak"], float(score))
            self.log.append(dict(t=round(t, 2), s=s, track=tr.id, box=[round(float(v), 1) for v in tr.boxes[max(tr.boxes)]],
                                 probs=[round(float(v), 4) for v in pi], cheat_score=round(float(score), 4),
                                 alert=tr.alert, label=self.classes[label] if tr.alert else self.classes[0]))

    def _close(self, tr):
        e = tr.event
        e["type"] = self.classes[e.pop("types").most_common(1)[0][0]]
        if e["windows"] >= self.min_windows:  # drop one-off blips
            self.events.append(e)
        tr.alert, tr.event = False, None

    def _expire(self, s, t, max_gap=25):
        for tid in [k for k, tr in self.tracks.items() if s - tr.last_seen > max_gap]:
            if self.tracks[tid].alert:
                self._close(self.tracks[tid])
            del self.tracks[tid]
        oldest = s - self.n_frames
        for tr in self.tracks.values():  # keep only the boxes a window can still use (long streams)
            for k in [k for k in tr.boxes if k < oldest and k != tr.last_seen]:
                del tr.boxes[k]

    def finish(self):
        for tr in self.tracks.values():
            if tr.alert:
                self._close(tr)
        return sorted(self.events, key=lambda e: e["start"])

    # ------------------------------------------------------------------ drawing
    def draw(self, frame):
        k = max(1.0, frame.shape[1] / 1920)  # scale text / lines with resolution (4K sources)
        fs, th_txt, th_box = 0.7 * k, max(2, round(2 * k)), max(2, round(2 * k))
        for tr in self.tracks.values():
            if self.sample - tr.last_seen > 2:
                continue
            x1, y1, x2, y2 = map(int, tr.boxes[tr.last_seen])
            if tr.smoothed is None:
                color, text = (160, 160, 160), f"#{tr.id} warming up"
            else:
                score = 1 - tr.smoothed[0]
                label = int(np.argmax(tr.smoothed[1:]) + 1)
                color = COLORS[label] if tr.alert else COLORS[0]
                text = f"#{tr.id} {self.classes[label] if tr.alert else 'normal'} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th_box + (1 if tr.alert else 0))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th_txt)
            ty = max(th + int(10 * k), y1)  # keep the label inside the frame
            cv2.rectangle(frame, (x1, ty - th - int(10 * k)), (x1 + tw + int(8 * k), ty), color, -1)
            cv2.putText(frame, text, (x1 + int(4 * k), ty - int(6 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th_txt)
        return frame


def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return cap, fps, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, webcam index or RTSP/HTTP URL")
    ap.add_argument("--ckpt", required=True,
                    help="X3D best.pt from train_x3d.py (ver 1 or ver 2)")
    ap.add_argument("--det", default="yolo11m.pt", help="ultralytics person detector weights")
    ap.add_argument("--det-imgsz", type=int, default=640, help="detector input size; 1280 for crowded rooms")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--out", help="annotated output video (.mp4)")
    ap.add_argument("--events", help="events CSV")
    ap.add_argument("--log", help="per-window JSONL log")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds of video (0 = all)")
    ap.add_argument("--infer-every", type=float, default=1.0, help="seconds between classifier runs")
    ap.add_argument("--smooth", type=int, default=3, help="windows averaged per track")
    # defaults from tune_thresholds.py on S07+S08: 83% of cheating events caught, ~16 false alarms / person-hour
    ap.add_argument("--thr-on", type=float, default=0.85, help="cheat_score to raise an alert")
    ap.add_argument("--thr-off", type=float, default=0.7, help="cheat_score to clear it")
    ap.add_argument("--min-windows", type=int, default=2, help="alert must last this many windows to count")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    det = CheatDetector(args.ckpt, args.det, args.device, det_conf=args.det_conf, det_imgsz=args.det_imgsz,
                        infer_every=args.infer_every, smooth=args.smooth,
                        thr_on=args.thr_on, thr_off=args.thr_off, min_windows=args.min_windows)
    cap, fps, n = open_source(args.source)
    writer = None
    live = not os.path.isfile(args.source)
    f, t_start = 0, time.time()
    if n and args.max_seconds:
        n = min(n, int(args.max_seconds * fps))
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
            cv2.imshow("exam cheating detection", frame)
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
    print(f"{f} frames in {elapsed:.1f}s ({f / max(elapsed, 1e-6):.1f} fps, source {fps:.1f} fps), "
          f"{len(events)} events")
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
