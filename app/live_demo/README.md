# live_demo: live GPU inference behind the demo_web interface

Same page and features as `demo_web` (video with overlay, decision-rule sliders, timeline, person cards, live
metrics, alert log / ground truth, CSV export, model comparison), but the decisions come from **real inference
running now on the server GPU**, not from saved logs:

```
browser plays the video ──clock (every 250 ms, play / pause / seek / speed)──> server
server decodes the same video, never ahead of the player:
  every 0.2 s  YOLO11m person detection + ByteTrack (+ RAFT-small optical flow for v3)   ──det──> boxes
  every 1 s    per person: 16 frames (3.2 s), square crop ──> v1, v2, v3, v3late, v1a     ──win──> probabilities
browser: smoothing + hysteresis alert rule (sliders), scoring against ground truth, drawing
```

- The model on screen always runs. "Run all models live" (comparison card) runs the five on the same tracks, so
  the comparison table scores every model up to now; switching the model on screen or this option restarts the
  analysis at the current time (a model only has decisions from when it was running).
- A seek restarts the tracker at the new position with a 3.2 s pre-roll, so the first decision is there at once.
- If the GPU falls more than `LIVE_MAX_LAG` s (default 3) behind the player (e.g. 4x speed with all models), the
  analysis jumps to the present instead of accumulating latency.
- Videos: S07 / S08 (validation sessions, with ground truth: metrics, caught / missed, GT boxes) and any
  uploaded video (no ground truth: person cards and timeline rows are the tracks).
- The pipeline is a copy of `cheating_detection_v1/v3` `app/detector.py` (same crops, resize, normalisation, flow
  convention). Checked against the eval-videos logs of S07: v1 53/53 windows with identical tracks, median
  probability difference 0.003; v3 fusion 21/21 windows, median 0.003 (CPU fp32 vs the GPU bf16 reference).

## Speed on an RTX 3060 12 GB (Vast.ai, S07 1080p, 6 people + passers-by)

X3D-L is mostly depthwise 3D convolutions and elementwise ops: eager PyTorch needs 560 ms per model and window
of 6 people on a 3060. `torch.compile` (fused kernels, batch padded to 8, compiled at start-up) brings it to
165 ms (fusion 187 ms, RAFT 51 ms per sample, YOLO11m + ByteTrack 24 ms per sample).

| models running | player speed | GPU time per 0.2 s sample | lag behind the player (mean / max) |
|---|---|---|---|
| v1 | 1x | 91 ms | 0.00 / 0.04 s |
| v1 | 2x | 100 ms | 0.09 / 0.41 s |
| v3 (RAFT + fusion) | 1x | 149 ms | 0.03 / 0.15 s |
| v1 + v2 + v1a | 1x | 166 ms | 0.00 / 0.00 s |
| all five | 1x | 226 ms | 2.9 / 4.2 s: too much for a 3060, the analysis jumps ahead |

Hence the default (`LIVE_DEFAULT_RUN=family`): the model on screen runs with the models of the same kind that fit
in real time (v1 / v2 / v1a together, v3 and v3 late fusion alone). "Run all models live" runs the five. 4x
speed is beyond a 3060 even for one model (decoding + detection of 100 frames/s of 1080p): the analysis follows in
bursts.

## Layout

```
live_demo/
├── live_demo/
│   ├── server.py        FastAPI: page, /api/manifest, /api/upload, /media (HTTP Range), WebSocket /ws
│   ├── pipeline.py      ModelBank (models loaded once) + Analyzer (one per viewer: decoder, tracker, buffers)
│   ├── nets.py          X3D / two-stream fusion / RAFT, preprocessing (copied from the v1 / v3 projects)
│   ├── videos.py        built-in sessions + uploads (ffprobe, browser copy)
│   ├── fetch_media.py   S07 / S08 1080p sources from the DOAN1 Google Drive, 720p browser copies
│   ├── bench.py         throughput / lag / comparison with a reference log, no browser needed
│   └── config.py        models, thresholds, paths, LIVE_* environment overrides
├── static/              index.html, style.css (demo_web), app.js (demo_web logic fed by the WebSocket)
├── data/                gt_S07.json, gt_S08.json (ground truth, from demo_web)
├── scripts/             deploy.sh (your machine), setup_vast.sh + start.sh (GPU instance)
├── weights/             not in git, see below
└── media/               not in git: <id>_src.mp4 (1080p, analysed), <id>.mp4 (720p, played), uploads/
```

Weights (copy into `weights/` before deploying, ~175 MB):

| file | from |
|---|---|
| yolo11m.pt | cheating_detection_v1/weights/ |
| x3d_l_v1_best.pt | cheating_detection_v1/weights/ |
| x3d_l_v2_best.pt | cheating_detection_v2/weights/ |
| x3d_l_v1a_best.pt | cheating_detection_v1/outputs/variant_a/weights/ |
| fusion_best.pt, late_fusion.pt | cheating_detection_v3/weights/ |

A missing file only disables that model.

## Run on a Vast.ai RTX 3060 (no Docker to build)

```bash
# 1. rent a machine: image vastai/pytorch (torch preinstalled), port 8000 opened
vastai search offers 'gpu_name=RTX_3060 num_gpus=1 inet_down>=3000 rentable=true disk_space>=30' -o dph
vastai create instance <offer_id> --image vastai/pytorch:2.11.0-cuda-12.6.3-24.04-2026-09-08 \
    --disk 40 --ssh --direct --env '-p 8000:8000' --label examguard-live
#    pick the image tag whose CUDA is <= the offer's cuda_max_good (cuda-12.8.1-auto needs a 12.8 driver)
vastai show instance <instance_id> --raw     # public_ipaddr, ports: "22/tcp" (ssh) and "8000/tcp" (web)

# 2. from your machine (Git Bash): send code + weights + 720p videos, install, fetch 1080p sources, start
LIVE_TOKEN=<secret> bash scripts/deploy.sh <ssh_port> root@<ip> /workspace

# 3. open  http://<ip>:<port mapped to 8000>/?token=<secret>
```

On the instance: `bash scripts/start.sh` restarts the server (log in `logs/server.log`), `LIVE_MODELS=v1,v2 bash
scripts/start.sh` limits the models, `python -m live_demo.bench --video S07 --seconds 60 --rate 1` measures the
lag of a viewer at 1x. Stop billing with `vastai destroy instance <id>` (uploads are lost with the instance).

## Run locally

```bash
pip install torch torchvision   # CUDA build for your GPU
pip install -r requirements.txt
python -m live_demo.fetch_media  # or copy media/ from demo_web + the 1080p videos
python -m live_demo.server --port 8000
```
Without a CUDA GPU it runs on CPU, far slower than real time (`LIVE_DEVICE=cpu LIVE_MODELS=v1` for a functional test).

## Settings (environment)

| variable | default | |
|---|---|---|
| LIVE_MODELS | v1,v2,v3,v3late,v1a | models loaded (offered in the page) |
| LIVE_DEFAULT_RUN | family | models run for a viewer: `family`, `selected` (only the model on screen) or `all` |
| LIVE_COMPILE | 1 | torch.compile the classifiers and RAFT (0: eager, 3 to 4x slower, instant start) |
| LIVE_BATCH | 8 | clips per classifier call (padded, one compiled shape) |
| LIVE_TOKEN | (none) | if set, the page must be opened once with `?token=...` (cookie afterwards) |
| LIVE_MAX_SESSIONS | 2 | viewers analysed at the same time (one GPU) |
| LIVE_MAX_LAG | 3.0 | seconds behind the player before jumping to the present |
| LIVE_MAX_UPLOAD_BYTES | 4 GiB | upload size limit |
| LIVE_DEVICE | cuda | `cpu` for a test without GPU |
