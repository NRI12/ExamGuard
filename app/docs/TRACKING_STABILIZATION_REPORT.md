# Tracking Stabilization Report

> Historical single-video stabilization report. The newer multi-video gate and
> current decision are documented in `TRACKING_VALIDATION_REPORT.md`.

Updated: 2026-09-21

## Scope and Test Material

This gate covers YOLO11n person detection, ByteTrack, the monitoring runtime,
WebSocket metadata, and Canvas rendering only. It adds no database schema and no
Seat Assignment, candidate mapping, ROI, TSM, action recognition, event,
evidence, report, or appeal behavior.

Representative source:

```text
data/uploads/4813700a-696b-44ee-81b4-657cca9ad895/source.mp4
1920x1080 H.264, 25 FPS, 644.68 seconds
```

Manual inspection at 0, 61.52, 120, and 180 seconds confirms six seated
candidates and one partially occluded person at the rear workstation. The
stabilized configuration was evaluated for 180 continuous seconds (2,251 AI
frames). A separate 30-second benchmark and 20-second realtime runtime probe
were run after the final code changes.

The host NVIDIA driver was unavailable during verification. All measurements in
this report therefore use an explicit benchmark-only `--device cpu` override.
The product profiles remain `cuda:0`, FP16, and fail clearly instead of silently
falling back to CPU.

## Root-Cause Findings

### 1. Duplicate boxes and increasing IDs

**Symptom:** One visible person could receive partial and full-body boxes, and
the tracker created more IDs than the number of visible people.

**Root cause:** At the original `conf=0.10`, `iou=0.70` detector setting, YOLO
returned low-confidence partial-person boxes as well as the correct full-person
boxes. With `new_track_thresh=0.25`, some partial boxes were eligible to start
new tracks. No evidence was found of two detector calls being merged or raw
detections and tracks being rendered together.

**Evidence:** At timestamp 0, manual inspection found 7 people, while the raw
detector returned 9 detections and baseline ByteTrack returned 8 active tracks.
Over 30 seconds the baseline produced 4,339 detections, created 17 tracks, and
peaked at 10 simultaneous tracks.

**Fix:** Keep the low association threshold `conf=0.10`, reduce YOLO NMS IoU to
`0.50`, and raise only ByteTrack `new_track_thresh` to `0.50`. The changes were
selected through controlled ablations, not used to conceal lifecycle bugs.

**Regression test:** Runtime profile tests assert `classes=[0]`, `iou=0.50`,
`new_track_thresh=0.50`, and disabled UI interpolation/smoothing. The benchmark
exports per-frame raw detections, confidences, tracked boxes, IDs, and suspicious
pairwise overlaps for manual comparison.

**Result:** The final 30-second run produced 3,318 detections, 9 created tracks,
a peak of 7 simultaneous tracks, and no frame above 7 active tracks. The
180-second run also had zero frames above 7 active tracks.

### 2. Pause/resume reset ByteTrack

**Symptom:** Resuming normal playback could change IDs even without a seek.

**Root cause:** `resume(timestamp_ms)` called the seek path, which advanced the
runtime generation and reset ByteTrack. The frontend always supplied the current
timestamp when resuming.

**Evidence:** The worker reset count increased during an ordinary pause/resume.

**Fix:** Resume now resumes the analysis clock without seeking. The frontend
sends a timestamp only for an explicit seek. Explicit seek still increments the
generation, clears stale work, and resets the tracker before processing the new
timeline.

**Regression test:** `test_worker_cadence_pause_seek_latest_drop_and_cleanup`
asserts that pause/resume preserves the tracker and that seek resets it.

**Result:** Pause/resume retains one worker and tracker. Seek starts sequence 1
in a new runtime generation.

### 3. Replaced runtime callbacks could affect a newer runtime

**Symptom:** A late callback from a stopped/replaced worker could theoretically
publish into or mark the state of a new runtime for the same session.

**Root cause:** Manager callbacks were keyed only by `session_id`.

**Evidence:** The callback API contained no worker/runtime identity check.

**Fix:** Each start creates a runtime UUID. Publish, ready, completion, and error
callbacks are accepted only when that UUID matches the active handle. A second
start is deterministically rejected while a runtime is active.

**Regression test:**
`test_manager_rejects_duplicate_start_and_ignores_replaced_worker_callbacks`
covers duplicate start and late publish/error callbacks from a replaced worker.

**Result:** Exactly one active runtime, detector, tracker, decoder loop, and
publisher exist for a session.

### 4. WebSocket messages had no stale-message identity

**Symptom:** After seek, restart, or reconnect, the browser could not prove that
an in-flight tracking message belonged to the current timeline.

**Root cause:** Tracking payloads had no runtime instance, generation, or
monotonic per-generation sequence.

**Evidence:** The prior tracking schema contained only session, frame,
timestamp, source dimensions, and tracks.

**Fix:** Runtime-only `runtime_instance_id`, `runtime_generation`,
`tracker_instance_id`, and `tracking_seq` fields are now included. The publisher
retains latest-value semantics; no sequence or tracking state is persisted.

**Regression test:** Backend schema/runtime tests validate identifiers and
sequence reset. Frontend buffer tests reject duplicate sequences, older
generations, and messages from retired runtime instances.

**Result:** One accepted latest state is rendered for the active generation.

### 5. Frontend stale rendering risks

**Symptom:** Canvas artifacts or stacked callback loops could look like duplicate
tracking even when the backend sent one state.

**Root cause:** No observed duplicate came from Canvas, but cleanup and stale
message protection were not strong enough to rule out restart/reconnect races.

**Evidence:** The existing frontend already rendered only `tracks`, selected one
nearest frame, and used a bounded buffer. It did not reject stale generations or
duplicate sequences. Canvas clearing needed an explicit backing-store guarantee
for device pixel ratio scaling.

**Fix:** The buffer is generation-aware, rendering remains raw and
non-interpolated, one stable callback chain reads the buffer, and every draw
clears the entire backing store under the identity transform. Seek clears frame
data. Unmount cancels the frame callback/animation callback, disconnects resize
and fullscreen handlers, clears Canvas, detaches socket handlers, and closes the
socket.

**Regression test:** Frontend tests cover one selected frame, generation and
sequence rejection, full backing-store clear, one render loop with cancellation,
geometry, reconnect policy, and bounded buffering.

**Result:** There is no raw-detection plus tracked-box double rendering and no
render-loop recreation on each WebSocket update.

## Configuration and Ablations

Final YOLO11n configuration:

```yaml
imgsz: 640
conf: 0.10
iou: 0.50
classes: [0]
max_det: 32
device: cuda:0
half: true
```

Final ByteTrack configuration:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.50
track_buffer: 20  # GTX 1650; RTX 3060 retains 30
match_thresh: 0.80
fuse_score: true
```

Controlled real-video results:

| Experiment | Result | Decision |
|---|---:|---|
| Baseline: IoU 0.70, new 0.25, 30 s | 17 created, peak 10 | Rejected |
| New-track 0.50 only, 30 s | 10 created, peak 8 | Improved |
| New-track 0.60 | 7 created, peak 7, but only 4-5 people often tracked | Rejected |
| Detector IoU 0.50 only, 30 s | 3,318 detections, 14 created, peak 9 | Improved |
| IoU 0.50 + new-track 0.50, 30 s | 9 created, peak 7 | Selected |
| Match 0.70, 75 s | 22 created | Rejected, fragmentation worse |
| Track-low 0.20, 75 s | A seated track was lost | Rejected |
| Track-buffer 30, 75 s | Did not prevent observed switch | Rejected |
| Detector IoU 0.35, 75 s | Peak returned to 8 | Rejected |

## Final Measurements

Final 30-second offline benchmark, excluding the first lazy-inference frame from
steady-state latency:

```text
frames analyzed: 376
offline throughput: 34.69 FPS
detector latency: average 18.02 ms, P95 20.85 ms
tracker latency: average 0.834 ms, P95 0.900 ms
pipeline latency: average 18.86 ms, P95 21.71 ms
first-frame warmup: 2,473 ms pipeline
detections: 3,318
created tracks: 9
peak simultaneous tracks: 7
suspicious raw overlap pairs at IoU >= 0.50: 0
```

Final 20-second realtime worker probe at a target of 12.5 FPS:

```text
actual analysis FPS: 12.40
tracking frames: 248
analysis lag: average 43.3 ms, P95 72 ms, maximum 171 ms
maximum queue size: 1
clean stop: true
runtime errors: none
```

The 252 dropped-frame count includes intentional sampling from native 25 FPS to
12.5 FPS plus latest-frame replacement. Lag stayed bounded and did not grow over
the probe.

## Tracking Quality Observations

For the 180-second, 2,251-frame reviewed segment:

```text
peak simultaneous tracks: 7
frames with more than 7 tracks: 0
frames with 7 tracks: 506
frames with 6 tracks: 1,688
frames with 5 tracks: 57
```

Five clearly visible seated people retained IDs 1-5 for the full 180 seconds.
The partially occluded rear person used ID 7 from 0.16 to 61.52 seconds, was
absent for about 4 seconds, and then used ID 14 from 65.76 to 180 seconds. At the
failure point the correct approximately 0.40-confidence detection remained, but
ByteTrack associated ID 7 to an incorrect large 0.181-confidence box. This is a
known occlusion/association limitation, not continuous ID churn.

No ground-truth MOT annotation was created, so this report does not claim MOTA,
HOTA, universal recall, or a universal ID-switch rate. The manual review CSVs in
`/tmp/examguard-tracking/` contain raw and tracked state for the exact timestamps,
including `final-candidate-180s.csv` and `stabilized-final.csv`.

## Scenario Coverage and Remaining Limitations

The real segment covers seated people, head/body movement, adjacent candidates,
partial occlusion, and a standing/rear person. Pause/resume, forward/backward
seek semantics, duplicate start, stale generation, stop cleanup, and reconnect
policy are covered by regression tests. Geometry tests cover contained-video
coordinate mapping; Canvas cleanup covers DPR-backed dimensions and fullscreen
resize callbacks.

Remaining limitations:

- The partially occluded rear person has one documented fragmentation/ID switch.
- No annotated multi-video ground truth exists, so recall and ID-switch metrics
  remain manual observations rather than benchmark claims.
- GPU FPS, VRAM, utilization, and full browser-to-CUDA end-to-end behavior could
  not be measured while the NVIDIA host driver was unavailable.
- ByteTrack cannot guarantee permanent identity through severe occlusion; Track
  ID remains runtime-only and is not a candidate identity.

## Verification

```text
Tracking backend tests: 16 passed
Backend Ruff check: passed
Ruff format check for tracking files: passed
Backend mypy: passed, 74 source files
Frontend Vitest: 4 files, 13 tests passed
Frontend typecheck and production build: passed
Full backend suite: 59 passed, 1 skipped, and 1 pre-existing unrelated settings
failure
```

The unrelated failure is
`tests/test_settings.py::test_old_secret_key_alias_is_not_accepted`: the Compose
environment supplies a valid current secret while the test checks rejection of
an obsolete alias. No tracking test failed.

A repository-wide optional `ruff format --check` also reports six accepted
Phase 5.5 files that predate this gate as unformatted. They were left untouched
to preserve the accepted working tree; all tracking Python files pass the format
check.
