"""Real-time pipeline: YOLO + ByteTrack persons -> per-person windows -> RGB / flow / fusion model -> alerts.

  frame --(every 0.2 s)--> YOLO person detection + ByteTrack
        --> (flow models) RAFT-small flow previous sample -> this sample on the half-resolution frame
        --> box mapped to the training crop geometry --> per-track history of the last `frames` samples
        --(every app.infer_every s)--> one batched forward for all tracks --> softmax
        --> mean of the last app.smooth windows --> cheat_score = 1 - P(normal)
        --> hysteresis alert (thr_on / thr_off) + type = argmax of the cheating classes
"""
import collections

import cv2
import numpy as np
import torch

import torch.nn.functional as F

from ..data.dataset import gpu_eval_transform
from ..data.prepare import square_crop_box
from ..models.raft_flow import FlowEstimator, flow_to_input, resize_flow
from ..models.registry import forward, load_model, uses_flow, uses_rgb

COLORS = [(80, 200, 80), (0, 165, 255), (255, 120, 0), (60, 60, 230), (200, 60, 200)]  # BGR per class index


def map_box(b, w_img, h_img, bm):
    """YOLO full-body box -> CVAT-style 'upper body + desk' box (see calibrate)."""
    x1, y1, x2, _ = b
    w = x2 - x1
    ny1 = y1 + bm["dy1"] * w
    return np.clip([x1 + bm["dx1"] * w, ny1, x2 + bm["dx2"] * w, ny1 + bm["h"] * w], 0, [w_img, h_img, w_img, h_img])


class Track:
    def __init__(self, tid):
        self.id, self.boxes, self.last_seen = tid, {}, -1
        self.probs, self.smoothed, self.alert, self.event = collections.deque(), None, False, None


class CheatDetector:
    def __init__(self, cfg, checkpoint=None, device="cuda"):
        from ultralytics import YOLO
        a, d = cfg["app"], cfg["data"]
        self.cfg, self.device = a, device
        self.model, ck, self.info = load_model(checkpoint or cfg["paths"]["weights"], d, device)
        self.classes = ck["classes"]
        self.n_frames, self.sample_dt = self.info["frames"], self.info["sample_dt"]
        self.flow_est = FlowEstimator(device) if uses_flow(self.info) else None
        self.clip_size, self.crop_pad = d["clip_size"], d["crop_pad"]
        self.det = YOLO(a["detector"])
        self.infer_every = max(1, round(a["infer_every"] / self.sample_dt))
        self.min_boxes = int(np.ceil(a["min_coverage"] * self.n_frames))
        self.reset()

    def reset(self):
        self.tracks, self.frames = {}, collections.OrderedDict()
        self.flows, self.prev_half = collections.OrderedDict(), None  # sample -> [2, H/2, W/2] flow (px)
        self.events, self.log, self.sample = [], [], -1
        self.det.predictor = None  # fresh tracker state

    def process(self, frame_bgr, t):
        """Feed every frame with its timestamp (s); detection / classification run on sampled frames only."""
        s = int(t / self.sample_dt + 1e-6)
        if s == self.sample:
            return
        self.sample = s
        h, w = frame_bgr.shape[:2]
        r = self.det.track(frame_bgr, persist=True, classes=[0], conf=self.cfg["det_conf"],
                           imgsz=self.cfg["det_imgsz"], tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                tr = self.tracks.setdefault(tid, Track(tid))
                tr.boxes[s], tr.last_seen = map_box(box, w, h, self.cfg["box_map"]), s
        self.frames[s] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.flow_est is not None:
            self._flow(s)
        while len(self.frames) > self.n_frames:
            self.frames.popitem(last=False)
        while len(self.flows) > self.n_frames:
            self.flows.popitem(last=False)
        if s % self.infer_every == 0 and len(self.frames) == self.n_frames:
            self._infer(s, t, w, h)
        self._expire(s)

    @torch.no_grad()
    def _flow(self, s):
        """Flow from the previous sample (0.2 s earlier) to this one, at half resolution (zero at a gap)."""
        img = torch.from_numpy(self.frames[s]).to(self.device).permute(2, 0, 1)[None].float()
        half = F.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
        if self.prev_half is not None and self.prev_half[0] == s - 1:
            self.flows[s] = self.flow_est(self.prev_half[1], half)[0].half()
        else:
            self.flows[s] = torch.zeros(2, *half.shape[-2:], device=self.device, dtype=torch.half)
        self.prev_half = (s, half)

    @torch.no_grad()
    def _infer(self, s, t, w, h):
        window, batch, flows, ready = list(self.frames), [], [], []
        for tr in self.tracks.values():
            have = [i for i in window if i in tr.boxes]
            if len(have) < self.min_boxes:
                continue
            boxes = [tr.boxes[min(have, key=lambda k: abs(k - i))] for i in window]
            x1, y1, x2, y2 = square_crop_box(boxes, w, h, self.crop_pad)
            if uses_rgb(self.info):
                batch.append(np.stack([cv2.resize(self.frames[i][y1:y2, x1:x2], (self.clip_size, self.clip_size),
                                                  interpolation=cv2.INTER_AREA) for i in window]))
            if uses_flow(self.info):  # same crop on the half-resolution flow, vectors rescaled to flow_size
                crop = torch.stack([self.flows[i][:, y1 // 2:y2 // 2, x1 // 2:x2 // 2].float() for i in window])
                flows.append(flow_to_input(resize_flow(crop, self.info["flow_size"])).transpose(0, 1))
            ready.append(tr)
        if not ready:
            return
        x_rgb = (gpu_eval_transform(torch.from_numpy(np.stack(batch)).to(self.device), self.info["rgb_size"])
                 if batch else None)
        x_flow = torch.stack(flows) if flows else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            p = forward(self.model, self.info, x_rgb, x_flow).float().softmax(-1).cpu().numpy()
        for tr, pi in zip(ready, p):
            tr.probs.append(pi)
            while len(tr.probs) > self.cfg["smooth"]:
                tr.probs.popleft()
            tr.smoothed = np.mean(tr.probs, 0)
            score, label = 1 - tr.smoothed[0], int(np.argmax(tr.smoothed[1:]) + 1)
            if not tr.alert and score >= self.cfg["thr_on"]:
                tr.alert = True
                tr.event = dict(track=tr.id, start=t, end=t, types=collections.Counter(), peak=0.0, windows=0)
            elif tr.alert and score < self.cfg["thr_off"]:
                self._close(tr)
            if tr.alert:
                e = tr.event
                e["end"], e["windows"], e["peak"] = t, e["windows"] + 1, max(e["peak"], float(score))
                e["types"][label] += 1
            self.log.append(dict(t=round(t, 2), s=s, track=tr.id, box=[round(float(v), 1) for v in tr.boxes[tr.last_seen]],
                                 probs=[round(float(v), 4) for v in pi], cheat_score=round(float(score), 4),
                                 alert=tr.alert, label=self.classes[label] if tr.alert else self.classes[0]))

    def _close(self, tr):
        e = tr.event
        e["type"] = self.classes[e.pop("types").most_common(1)[0][0]]
        if e["windows"] >= self.cfg["min_windows"]:
            self.events.append(e)
        tr.alert, tr.event = False, None

    def _expire(self, s, max_gap=25):
        for tid in [k for k, tr in self.tracks.items() if s - tr.last_seen > max_gap]:
            if self.tracks[tid].alert:
                self._close(self.tracks[tid])
            del self.tracks[tid]
        for tr in self.tracks.values():  # keep only boxes a window can still use (long streams)
            for k in [k for k in tr.boxes if k < s - self.n_frames and k != tr.last_seen]:
                del tr.boxes[k]

    def finish(self):
        for tr in self.tracks.values():
            if tr.alert:
                self._close(tr)
        return sorted(self.events, key=lambda e: e["start"])

    def draw(self, frame):
        k = max(1.0, frame.shape[1] / 1920)
        fs, th_txt = 0.7 * k, max(2, round(2 * k))
        for tr in self.tracks.values():
            if self.sample - tr.last_seen > 2:
                continue
            x1, y1, x2, y2 = map(int, tr.boxes[tr.last_seen])
            if tr.smoothed is None:
                color, text = (160, 160, 160), f"#{tr.id} warming up"
            else:
                label = int(np.argmax(tr.smoothed[1:]) + 1)
                color = COLORS[label % len(COLORS)] if tr.alert else COLORS[0]
                text = f"#{tr.id} {self.classes[label] if tr.alert else self.classes[0]} {1 - tr.smoothed[0]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th_txt + (1 if tr.alert else 0))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th_txt)
            ty = max(th + int(10 * k), y1)
            cv2.rectangle(frame, (x1, ty - th - int(10 * k)), (x1 + tw + int(8 * k), ty), color, -1)
            cv2.putText(frame, text, (x1 + int(4 * k), ty - int(6 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th_txt)
        return frame
