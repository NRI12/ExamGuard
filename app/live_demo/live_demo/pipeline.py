"""Live inference: the cheating_v1 / v3 real-time pipeline, driven by the viewer's playback clock.

  video frame t (only once the viewer's player has reached t)
    --every 0.2 s--> YOLO11m person detection + ByteTrack --> box mapped to the training crop geometry
                      (+ RAFT-small flow previous sample -> this one, half resolution, when a fusion model is on)
    --every 1 s----> per track: 16 samples (3.2 s) square crop 356 px --> one batch --> every enabled model
                      --> softmax probabilities sent to the page; smoothing / alert rule run in the browser

One ModelBank (classifiers + RAFT, loaded once) is shared by all viewers; each viewer gets an Analyzer with its own
decoder, YOLO tracker and buffers, and chooses which models run (at least the one on screen). Decoding never runs
ahead of the viewer. When the analysis falls more than MAX_LAG behind (GPU too slow for the chosen models or the
speed), it jumps to the present like a seek instead of accumulating latency.

Speed: X3D is mostly depthwise 3D convolutions and elementwise ops (memory bound). torch.compile fuses them
(RTX 3060, 6 people: 560 -> 126 ms per model and window); batches are padded to a fixed size so each model
compiles once, at start-up (a few minutes the first time, cached on disk afterwards).
"""
import collections
import concurrent.futures
import logging
import math
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import APP, BATCH, COMPILE, DEVICE, ENABLED, MAX_LAG, MODELS, PREROLL, WEIGHTS
from .nets import FlowEstimator, flow_to_input, gpu_eval_transform, load_classifier, map_box, resize_flow, square_crop_box

log = logging.getLogger("live_demo")
# X3D graph-breaks once per ResStage and the five models share that code: the default cache (8 entries per code
# object) would silently send the later models back to slow eager mode
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024


class ModelBank:
    def __init__(self, device=DEVICE):
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, running on CPU (far from real time)")
            device = "cpu"
        self.device = device
        self.lock = threading.Lock()  # one GPU job at a time: predictable latency with several viewers
        self.models, self.specs = {}, []
        for spec in MODELS:
            path = WEIGHTS / spec["weights"]
            if spec["id"] not in ENABLED:
                continue
            if not path.exists():
                log.warning("model %s skipped: %s not found", spec["id"], path)
                continue
            m, classes, info = load_classifier(path, device)
            if COMPILE and device.startswith("cuda"):
                m = torch.compile(m)
            self.models[spec["id"]] = (m, info)
            self.specs.append({k: v for k, v in spec.items() if k != "weights"} | dict(classes=classes, arch=info["arch"]))
            log.info("loaded %s (%s, %s)", spec["id"], info["arch"], path.name)
        if not self.models:
            raise SystemExit(f"no classifier weights found in {WEIGHTS}")
        infos = [i for _, i in self.models.values()]
        self.n_frames, self.sample_dt = infos[0]["frames"], infos[0]["sample_dt"]
        self.rgb_size = infos[0]["rgb_size"]
        self.flow_size = next((i["flow_size"] for i in infos if i["arch"] == "fusion"), None)
        assert all(i["frames"] == self.n_frames and i["sample_dt"] == self.sample_dt and i["rgb_size"] == self.rgb_size
                   for i in infos), "all live models must share the sampling and the RGB input"
        self.flow = FlowEstimator(device) if self.flow_size else None
        if self.flow and COMPILE and device.startswith("cuda"):
            self.flow.model = torch.compile(self.flow.model)
        self.cuda = device.startswith("cuda")
        self.bf16 = self.cuda and torch.cuda.is_bf16_supported()
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda else "CPU"
        self.warmup()

    @torch.no_grad()
    def warmup(self):
        t0 = time.time()
        clips = np.zeros((2, self.n_frames, APP["clip_size"], APP["clip_size"], 3), np.uint8)
        flows = [torch.zeros(2, self.n_frames, self.flow_size, self.flow_size, device=self.device)] * 2 \
            if self.flow_size else None
        if self.flow:  # 1080p source -> half-resolution flow, the shape of the built-in sessions
            t1 = time.time()
            half = torch.zeros(1, 3, 540, 960, device=self.device)
            self.flow_between(half, half)
            log.info("warm-up RAFT %.1fs", time.time() - t1)
        for mid in self.models:  # compiles each model for the fixed batch
            t1 = time.time()
            self.classify(clips, flows, [mid])
            log.info("warm-up %s %.1fs", mid, time.time() - t1)
        log.info("models ready on %s in %.1fs (bf16=%s, compiled=%s)", self.gpu_name, time.time() - t0, self.bf16,
                 COMPILE and self.cuda)

    def needs_flow(self, ids):
        return any(self.models[m][1]["arch"] == "fusion" for m in ids if m in self.models)

    @torch.no_grad()
    def classify(self, clips, flows, ids=None):
        """clips uint8 [B, T, S, S, 3] RGB, flows list of [2, T, f, f] -> {model id: probs [B, C]}.
        Runs the models `ids` (default all) on chunks padded to BATCH clips (one compiled shape)."""
        ids = [m for m in (ids or self.models) if m in self.models]
        n = len(clips)
        with self.lock:
            # contiguous: one memory layout, so the compiled graphs' stride guards always hold
            x_rgb = gpu_eval_transform(torch.from_numpy(clips).to(self.device), self.rgb_size).contiguous()
            x_flow = torch.stack(flows).contiguous() if flows else None
            pad = (-n) % BATCH
            if pad:
                x_rgb = torch.cat([x_rgb, x_rgb[:1].expand(pad, *x_rgb.shape[1:])])
                if x_flow is not None:
                    x_flow = torch.cat([x_flow, x_flow[:1].expand(pad, *x_flow.shape[1:])])
            out = {}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                for mid in ids:
                    m, info = self.models[mid]
                    if info["arch"] == "fusion" and x_flow is None:
                        continue
                    logits = [m(x_rgb[i:i + BATCH].contiguous()) if info["arch"] == "rgb" else
                              m(x_rgb[i:i + BATCH].contiguous(), x_flow[i:i + BATCH].contiguous())
                              for i in range(0, n + pad, BATCH)]
                    out[mid] = torch.cat(logits)[:n].float().softmax(-1).cpu().numpy()
            return out

    @torch.no_grad()
    def half_frame(self, rgb):
        img = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float()
        return F.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)

    @torch.no_grad()
    def flow_between(self, a, b):
        with self.lock:
            return self.flow(a, b)[0].half()


class Track:
    def __init__(self, tid):
        self.id, self.boxes, self.last_seen = tid, {}, -1


class Analyzer:
    """One viewer's analysis of one video. Thread-safe controls: clock(), seek(), close()."""

    def __init__(self, bank, video, emit, models=None):
        from ultralytics import YOLO
        self.bank, self.video, self.emit = bank, video, emit
        self.det = YOLO(APP["detector"])
        self.cap = cv2.VideoCapture(video.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {video.source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or video.fps
        self.n_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or video.frames
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.scale = (video.width / self.src_w) if video.width else 1.0  # boxes are sent in video.width space
        self.n_frames, self.sample_dt = bank.n_frames, bank.sample_dt
        self.infer_every = max(1, round(APP["infer_every"] / self.sample_dt))
        self.min_boxes = int(math.ceil(APP["min_coverage"] * self.n_frames))
        self.yolo_device = 0 if bank.cuda else "cpu"
        self.cond = threading.Condition()
        self.clock_t, self.playing, self.rate, self.clock_at = 0.0, False, 1.0, time.monotonic()
        self.pending_seek, self.closed = (0.0, "start"), False
        self.max_lag = MAX_LAG
        self.run_models = [m for m in bank.models if m in (models or bank.models)] or list(bank.models)[:1]
        self.pool = concurrent.futures.ThreadPoolExecutor(8)  # crop resizes (cv2 releases the GIL)
        self.max_id, self.id_offset = 0, 0
        self.perf = collections.deque(maxlen=50)  # (wall time, seconds of GPU work) per processed sample
        self.last_stat = 0.0
        self.thread = threading.Thread(target=self._run, name=f"analyzer-{video.id}", daemon=True)
        self.thread.start()

    # ---------- controls (called from the websocket task) ----------
    def clock(self, t, playing, rate):
        with self.cond:
            self.clock_t, self.playing, self.rate, self.clock_at = float(t), bool(playing), float(rate), time.monotonic()
            self.cond.notify()

    def seek(self, t):
        with self.cond:
            self.clock_t, self.clock_at = float(t), time.monotonic()
            self.pending_seek = (float(t), "seek")
            self.cond.notify()

    def set_models(self, ids):
        """Models to run from now on; restarts the analysis at the current time (flow buffer, clean windows)."""
        ids = [m for m in self.bank.models if m in ids] or list(self.bank.models)[:1]
        with self.cond:
            if ids == self.run_models:
                return
            self.run_models = ids
            self.pending_seek = (self._target(), "models")
            self.cond.notify()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify()
        self.thread.join(timeout=5)
        self.cap.release()
        self.pool.shutdown(wait=False)

    def _target(self):
        t = self.clock_t + ((time.monotonic() - self.clock_at) * self.rate if self.playing else 0.0)
        return min(t, self.n_total / self.fps)

    # ---------- worker thread ----------
    def _run(self):
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001 - reported to the page, the viewer can reload
            log.exception("analyzer %s crashed", self.video.id)
            self.emit(dict(type="error", message=f"analysis stopped: {e}"))

    def _loop(self):
        while True:
            with self.cond:
                if self.closed:
                    return
                seek, self.pending_seek = self.pending_seek, None
            if seek is not None:
                self._restart(*seek)
                continue
            target, t_next = self._target(), self.f / self.fps
            if self.eof or t_next > target:
                self._stats(target)
                with self.cond:
                    if not self.closed and self.pending_seek is None:
                        self.cond.wait(0.02)
                continue
            # jump only after 2 s of decisions since the last restart: when the GPU cannot keep up at all (4x on a
            # small GPU) the viewer still gets bursts of decisions instead of an endless series of pre-rolls
            if t_next >= self.emit_from + 2.0 and target - t_next > self.max_lag:
                self._restart(target, "lag")
                continue
            s = int(t_next / self.sample_dt + 1e-6)
            if s == self.sample:  # not a sampled frame: skip without converting it
                self.eof = not self.cap.grab()
                self.f += 1
                continue
            ok, frame = self.cap.read()
            self.f += 1
            if not ok:
                self.eof = True
                continue
            t0 = time.perf_counter()
            self._process(frame, t_next, s)
            self.perf.append((time.monotonic(), time.perf_counter() - t0))
            self._stats(target)

    def _restart(self, t, reason):
        """Fresh tracker and buffers, decoding from t - PREROLL; decisions are sent from t on."""
        start = max(0.0, t - PREROLL)
        self.f = int(round(start * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
        self.eof = False
        self.det.predictor = None  # new ByteTrack state
        self.id_offset = self.max_id
        self.tracks, self.frames, self.flows, self.prev_half, self.sample = {}, collections.OrderedDict(), \
            collections.OrderedDict(), None, -1
        self.emit_from = t
        self.use_flow = self.bank.flow is not None and self.bank.needs_flow(self.run_models)
        self.emit(dict(type="reset", t=round(t, 3), reason=reason, models=self.run_models))

    def _stats(self, target):
        now = time.monotonic()
        if now - self.last_stat < 0.5:
            return
        self.last_stat = now
        recent = [p for p in self.perf if now - p[0] < 5]
        span = (recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0
        self.emit(dict(type="stat", proc=round(self.f / self.fps, 2), target=round(target, 2),
                       lag=round(max(0.0, target - self.f / self.fps), 2), tracks=len(self.tracks),
                       sps=round((len(recent) - 1) / span, 1) if span > 0 else 0,
                       ms=round(1000 * float(np.mean([p[1] for p in recent])), 1) if recent else 0,
                       load=round(sum(p[1] for p in recent[1:]) / span, 2) if span > 0 else 0))

    @torch.no_grad()
    def _process(self, frame_bgr, t, s):
        self.sample = s
        h, w = frame_bgr.shape[:2]
        with self.bank.lock:
            r = self.det.track(frame_bgr, persist=True, classes=[0], conf=APP["det_conf"], imgsz=APP["det_imgsz"],
                               tracker="bytetrack.yaml", verbose=False, device=self.yolo_device)[0]
        seen = []
        if r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                gid = self.id_offset + tid
                self.max_id = max(self.max_id, gid)
                tr = self.tracks.setdefault(gid, Track(gid))
                tr.boxes[s], tr.last_seen = map_box(box, w, h, APP["box_map"]), s
                seen.append(tr)
        self.frames[s] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.use_flow:
            half = self.bank.half_frame(self.frames[s])
            if self.prev_half is not None and self.prev_half[0] == s - 1:
                self.flows[s] = self.bank.flow_between(self.prev_half[1], half)
            else:
                self.flows[s] = torch.zeros(2, *half.shape[-2:], device=self.bank.device, dtype=torch.half)
            self.prev_half = (s, half)
        while len(self.frames) > self.n_frames:
            self.frames.popitem(last=False)
        while len(self.flows) > self.n_frames:
            self.flows.popitem(last=False)
        live = t >= self.emit_from - 1e-6
        if live:
            k = self.scale
            self.emit(dict(type="det", t=round(t, 2),
                           b=[[tr.id] + [round(float(v) * k, 1) for v in tr.boxes[s]] for tr in seen]))
        if s % self.infer_every == 0 and len(self.frames) == self.n_frames and live:
            self._infer(t, w, h)
        self._expire(s)

    def _infer(self, t, w, h):
        window, clips, flows, ready = list(self.frames), [], [], []
        for tr in self.tracks.values():
            have = [i for i in window if i in tr.boxes]
            if len(have) < self.min_boxes:
                continue
            boxes = [tr.boxes[min(have, key=lambda k: abs(k - i))] for i in window]
            x1, y1, x2, y2 = square_crop_box(boxes, w, h, APP["crop_pad"])
            cs = APP["clip_size"]
            clips.append(self.pool.map(lambda i, c=(y1, y2, x1, x2): cv2.resize(
                self.frames[i][c[0]:c[1], c[2]:c[3]], (cs, cs), interpolation=cv2.INTER_AREA), window))
            if self.use_flow:  # same crop on the half-resolution flow, vectors rescaled
                crop = torch.stack([self.flows[i][:, y1 // 2:y2 // 2, x1 // 2:x2 // 2].float() for i in window])
                flows.append(flow_to_input(resize_flow(crop, self.bank.flow_size)).transpose(0, 1))
            ready.append(tr)
        if not ready:
            return
        clips = np.stack([np.stack(list(c)) for c in clips])
        probs = self.bank.classify(clips, flows or None, self.run_models)
        k = self.scale
        rows = [dict(t=round(t, 2), track=tr.id, box=[round(float(v) * k, 1) for v in tr.boxes[tr.last_seen]],
                     p={mid: [round(float(v), 4) for v in pr[j]] for mid, pr in probs.items()})
                for j, tr in enumerate(ready)]
        self.emit(dict(type="win", rows=rows))

    def _expire(self, s, max_gap=25):
        for tid in [k for k, tr in self.tracks.items() if s - tr.last_seen > max_gap]:
            del self.tracks[tid]
        for tr in self.tracks.values():  # keep only boxes a window can still use
            for k in [k for k in tr.boxes if k < s - self.n_frames and k != tr.last_seen]:
                del tr.boxes[k]
