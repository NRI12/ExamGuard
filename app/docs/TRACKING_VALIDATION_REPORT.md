# Tracking Validation and Benchmark Report

Updated: 2026-09-22

## Gate Decision

```text
TRACKING STATUS: PASSED ON THE AVAILABLE MINI VALIDATION SET
```

Runtime, ordering, WebSocket, Canvas, cleanup, and bounded-latency requirements
pass. The earlier duplicate partial/full-body tracks and reviewed partial-occlusion
miss are resolved on the three available independent clips. This is a scoped gate:
supervisor-walk, complete stand/sit, GPU, and browser-to-CUDA validation are still
unavailable, and no formal MOT accuracy claim is made.

## Current Configuration

YOLO11n:

```yaml
imgsz: 640
classes: [0]
conf: 0.10
iou: 0.50
max_det: 64
device: cuda:0
half: true
duplicate_suppression:
  enabled: true
  containment_threshold: 0.90
  max_area_ratio: 0.70
  max_center_distance_ratio: 0.35
  preferred_detection_confidence: 0.25
  larger_box_min_confidence_ratio: 0.65
```

ByteTrack GTX1650:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.40
track_buffer: 30
match_thresh: 0.80
fuse_score: true
```

The product profile remains CUDA-only. CPU was an explicit benchmark override
because the host NVIDIA driver was unavailable.

## Runtime Verification

- The system instantiates `PersonDetector` once and a direct Ultralytics
  `BYTETracker` once per `VideoAnalysisWorker`.
- `model.track()` is not used, so `persist=True` is not applicable. Persistence
  comes from retaining the same `BYTETracker` object for the generation.
- Duplicate start is rejected. Runtime, worker, and tracker UUIDs remain stable
  during normal playback.
- Pause/resume preserves the tracker. Seek advances generation, clears pending
  work, resets ByteTrack, and restarts sequence numbering.
- Tracker input now requires strictly increasing timestamps. Non-increasing
  packets in one generation are dropped before detector/tracker invocation.
- The one-slot buffer replaces stale work. Realtime queue peak was 1 and lag
  stayed bounded.
- WebSocket messages carry runtime instance, generation, tracker instance, and
  monotonic per-generation sequence. The frontend rejects stale/duplicate state.
- Canvas renders only one nearest `TrackingFrame`, clears the full DPR backing
  store, and owns one cleaned-up render loop. Diagnostics are development-only.

## Validation Set

The manifest is `docs/tracking_validation/manifest.yaml`; manual references are
in `docs/tracking_validation/manual_annotations.csv`.

Three independent 60-second, 1920x1080, 25 FPS sources were used. Three other
uploads were excluded because their SHA-256 matched clip B exactly.

| Clip | People | Coverage | Final peak tracks |
|---|---:|---|---:|
| A | 7 | seated, posture, adjacent, partial occlusion | 7 |
| B | 7 | seated, posture, adjacent, partial occlusion | 7 |
| C | 6 | stable seated, posture, adjacent | 6 |

No independent source contained a supervisor walking through the room or a full
stand-move-sit sequence. Those scenarios remain validation blockers.

## Manual Quality Metrics

Nine timestamps produced 60 visible-person observations:

```text
person recall: 60 / 60 = 100% on the reviewed timestamps
isolated non-person false tracks: 0
extra duplicate tracks: 0
duplicate-person observation rate: 0 / 60 = 0%
mean active-track count error: 0 / 9 = 0
maximum active-track count error: 0 at reviewed timestamps
confirmed clear-person ID switches: 0 in the sparse reviewed timestamps
```

The zero ID-switch count is limited to the sparse manually reviewed timestamps;
it is not a universal claim. Across every analyzed frame, active track count never
exceeded the fixed per-clip person reference; mean absolute count error was 0.424.
The fixed reference does not annotate temporary visibility changes, so this is a
diagnostic count measure rather than a formal MOTA/MOTP/HOTA metric. Final
60-second lifetime summaries were:

| Clip | Created tracks | Frames at / below / above reference | Lifetime min / median / P95 / max (ms) |
|---|---:|---:|---|
| A | 13 | 319 / 432 / 0 | 0 / 34,880 / 60,000 / 60,000 |
| B | 13 | 419 / 332 / 0 | 2,640 / 23,840 / 60,000 / 60,000 |
| C | 8 | 652 / 99 / 0 | 1,680 / 60,000 / 60,000 / 60,000 |

The 34 created runtime IDs for 20 people expose remaining fragmentation under
long/partial occlusion. This is preferable to simultaneous duplicate boxes for the
operator overlay, but it reinforces the invariant that Track ID is not permanent
candidate identity.

## Root Causes

### Duplicate partial and full-body tracks

**Symptom:** Clip A has 8-10 active tracks for 7 people.

**Root cause:** Raw YOLO output contains 11-13 person boxes at the failure
timestamps. Several are partial/full-body boxes for the same person with mutual
IoU below 0.50, so standard IoU NMS does not remove them. Some cross the new-track
threshold and become simultaneous ByteTrack tracks.

**Evidence:** At 28.32s, 7 people produce 13 detections and 10 active tracks. The
annotated artifact shows three duplicated people. The WebSocket payload contains
the same 10 tracks and Canvas draws only that one state, excluding a frontend
duplication cause.

**Fix:** Add a conservative post-NMS containment pass. It only compares strongly
contained, center-aligned boxes with an area ratio at or below 0.70. It normally
keeps the full-person box, but retains a materially stronger partial detection when
the larger candidate is weak. Ordinary overlapping neighbors are not suppressed.

**Regression coverage:** Raw detection/tracked boxes and classes are exported;
person-only config and single-inference/runtime behavior are tested.

**Post-fix result:** Clip A peak falls from 10 to 7, matching the seven-person
reference with no over-count frame in the final 60-second run.

### Partial-occlusion miss

**Symptom:** Clip B has 6 tracks for 7 visible people at 44.96s.

**Root cause:** The rear person is visible only as a low-confidence partial box.
Raising `new_track_thresh` prevents duplicate starts but also prevents this person
from establishing/re-establishing a track.

**Evidence:** The artifact shows a `0.25` rear-person detection without an active
track. Raising `new_track_thresh` to 0.60 reduced duplicates but left only 4-6
tracks for long portions of clips A/B.

**Fix:** Reduce `new_track_thresh` from 0.50 to 0.40 after nested duplicates are
removed, while retaining `conf=0.10` and `track_low_thresh=0.10` for association.

**Regression coverage:** Lifecycle logs and CSV make loss/recreation measurable.

**Post-fix result:** The rear person is tracked at the reviewed 44.96-second point;
all nine reviewed timestamps match their visible-person reference.

### Pause/resume, stale runtime, and renderer risks

The earlier lifecycle defects remain fixed: resume no longer seeks, late worker
callbacks are runtime-UUID guarded, generation/sequence rejects stale messages,
and Canvas/render/WebSocket cleanup is deterministic. Regression tests pass.

## Ablation Record

Representative 60-second clip B unless noted:

| Experiment | Created / peak | Result |
|---|---:|---|
| Baseline IoU .70, new .25, buffer 30 | 23 / 11 | Failed |
| Buffer 20 only | 26 / 11 | More fragmentation; rejected |
| New .50 only | 12 / 8 | Improved duplicate creation |
| IoU .50 only | 17 / 9 | Removed IoU>=.50 overlaps, still duplicates |
| IoU .50 + new .50 | 11 / 7 | Balanced candidate |
| IoU .35 + new .50, clip A | 12 / 8, 66 LOST | Rejected |
| IoU .50 + new .55, clip A | 10 / 9 | Still duplicates |
| IoU .50 + new .60, clip A | 9 / 7 | Recall too low; rejected |
| IoU .50 + containment + new .50 | Lower fragmentation | Misses A at 15s and B at 45s; rejected |
| IoU .50 + containment + new .45 | A 11 / 7; B 12 / 7 | Still misses B at 45s; rejected |
| IoU .50 + containment + new .40 | A 13 / 7; B 13 / 7; C 8 / 6 | Final; 60/60 reviewed observations |

## Performance

Across the final 180-second set (2,253 frames; first lazy frame per source
excluded from steady-state latency):

```text
pipeline: mean 23.1227 ms, P95 26.1956 ms, max 46.3440 ms
```

Final 20-second realtime CPU probe:

```text
target analysis FPS: 12.5
actual analysis FPS: 12.50
analysis lag mean / P95 / max: 23.53 / 27 / 111 ms
dropped analysis frames: 251
maximum queue size: 1
clean stop: true
runtime errors: none
```

Drops include intentional 25-to-12.5 FPS sampling. GPU utilization and VRAM are
unavailable because `nvidia-smi` cannot communicate with the host driver.

## Artifacts

Final benchmark outputs are under `/tmp/examguard-tracking/final3-clip-{a,b,c}`
and `/tmp/examguard-tracking/final3-realtime`. Each benchmark
directory contains `summary.json`, `frames_or_tracks.csv`, and
`config_snapshot.yaml`. Important images include:

```text
candidate-6627/error-duplicate-28320.jpg
candidate-6627/recovered-count-60000.jpg
candidate-8eaf/error-missed-44960.jpg
candidate-a456/stable-30000.jpg
```

CLI usage:

```bash
python -m app.cli.benchmark_tracking --video /data/test.mp4 \
  --profile gtx1650 --output /data/tracking_eval/results

python -m app.cli.render_tracking_artifact --video /data/test.mp4 \
  --csv /data/tracking_eval/results/frames_or_tracks.csv \
  --timestamp-ms 28320 --output /data/tracking_eval/results/failure.jpg
```

The gate includes the earlier 900 seconds of offline ablations, a final 180-second
run over the three unique sources, and a final 20-second realtime probe.

## Verification and Scope

```text
targeted detector/tracker/runtime pytest: 20 passed
full backend pytest: not completed in this host environment (Docker daemon unavailable;
the temporary host CPython build lacks its native SQLite module)
Ruff check: passed
Ruff format check for changed backend files: passed
mypy: passed, 75 source files
frontend Vitest: 4 files, 13 tests passed
frontend typecheck and production build: passed
```

No database migration or tracking table was added. No Seat Assignment,
Track-to-Seat, Candidate mapping, ROI, TSM/X3D, action recognition, Event,
Evidence, Appeal, or Report functionality was added.

## Remaining Blockers

1. Add independent supervisor-walk and stand/sit clips to the validation set.
2. Measure fragmentation and ID switches against dense MOT-style ground truth;
   current sparse review cannot establish identity accuracy.
3. Re-run performance and browser-to-CUDA validation after the NVIDIA driver is
   available.
