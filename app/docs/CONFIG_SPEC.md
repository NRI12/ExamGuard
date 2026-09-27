# ExamGuard — Configuration Specification

Use `.env` for secrets/deployment, YAML for AI/runtime behavior and Pydantic Settings for validation.

## Seat-free identity defaults

Seat configuration is optional. Both hardware profiles default to `identity.mode: logical_track`; `seat` remains an advanced compatibility mode. Selected GTX1650 values are:

```yaml
identity:
  mode: logical_track
logical_tracking:
  center_history_size: 6
  raw_track_history_size: 8
  recovery:
    max_lost_ms: 5500
    max_position_distance: 0.18
    min_scale_similarity: 0.40
    min_score: 0.55
    ambiguity_margin: 0.08
    position_weight: 0.55
    motion_weight: 0.25
    scale_weight: 0.20
reid:
  enabled: false
  mode: on_demand
  encoder: hsv_histogram_v1
  max_batch_size: 16
  min_similarity: 0.72
  appearance_weight: 0.45
  min_combined_score: 0.68
  ambiguity_margin: 0.06
  prototype_seed_delay_ms: 800
  min_prototype_confidence: 0.50
  max_request_age_ms: 250
dynamic_neighbors:
  enabled: true
  max_neighbors_per_actor: 2
  max_horizontal_gap_ratio: 2.5
  max_vertical_gap_ratio: 0.75
  min_scale_similarity: 0.40
```

Motion thresholds were selected from the real-video ablation in `SEAT_FREE_TRACKING_REPORT.md`. Sparse appearance is implemented for EXP-TRACK-B but disabled because it did not improve identity metrics.

## .env example
```env
APP_ENV=development
APP_PROFILE=gtx1650
DATABASE_URL=postgresql+psycopg://examguard:examguard@db:5432/examguard
JWT_SECRET=CHANGE_ME
JWT_ALGORITHM=HS256
UPLOAD_ROOT=/data/uploads
EVIDENCE_ROOT=/data/evidence
MODEL_ROOT=/models
CUDA_VISIBLE_DEVICES=0
```

## GTX1650
```yaml
profile: gtx1650
runtime:
  device: cuda:0
  precision: fp16
video:
  realtime_analysis: true
  drop_stale_analysis_frames: true
analysis:
  target_fps: 12.5
  minimum_fps: 10.0
  queue_size: 1
detector:
  model: /models/detection/yolo11n.pt
  imgsz: 640
  conf: 0.10
  iou: 0.50
  classes: [0]
  max_det: 64
  half: true
  duplicate_suppression:
    enabled: true
    containment_threshold: 0.90
    max_area_ratio: 0.70
    max_center_distance_ratio: 0.35
    preferred_detection_confidence: 0.25
    larger_box_min_confidence_ratio: 0.65
tracker:
  config: /app/configs/tracking/bytetrack_exam.yaml
diagnostics:
  publish_hz: 2
seat_assignment:
  enabled: true
  overlap_weight: 0.70
  distance_weight: 0.30
  min_score: 0.35
  seat_expand_ratio: 0.08
  confirm_ms: 600
  release_ms: 1500
  switch_margin: 0.15
  switch_confirm_ms: 800
ui:
  tracking_interpolation: false
  tracking_smoothing: false
  show_confidence_default: false
```

## RTX3060
```yaml
profile: rtx3060
runtime:
  device: cuda:0
  precision: fp16
video:
  realtime_analysis: true
  drop_stale_analysis_frames: true
analysis:
  target_fps: 18.0
  minimum_fps: 12.0
  queue_size: 1
detector:
  model: /models/detection/yolo11n.pt
  imgsz: 640
  conf: 0.10
  iou: 0.50
  classes: [0]
  max_det: 32
  half: true
  duplicate_suppression:
    enabled: true
    containment_threshold: 0.90
    max_area_ratio: 0.70
    max_center_distance_ratio: 0.35
    preferred_detection_confidence: 0.25
    larger_box_min_confidence_ratio: 0.65
tracker:
  config: /app/configs/tracking/bytetrack_exam_3060.yaml
diagnostics:
  publish_hz: 2
seat_assignment:
  enabled: true
  overlap_weight: 0.70
  distance_weight: 0.30
  min_score: 0.35
  seat_expand_ratio: 0.08
  confirm_ms: 600
  release_ms: 1500
  switch_margin: 0.15
  switch_confirm_ms: 800
ui:
  tracking_interpolation: false
  tracking_smoothing: false
  show_confidence_default: false
```

Do not automatically use a larger YOLO model on RTX3060. Benchmark first.

## ByteTrack GTX1650
```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.40
track_buffer: 30
match_thresh: 0.80
fuse_score: true
```

## ByteTrack RTX3060
```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.40
track_buffer: 30
match_thresh: 0.80
fuse_score: true
```

At monitoring start, log active profile, detector model/config, tracker config and source metadata. Invalid config must fail fast. Do not silently fall back to CPU unless configured.

The stabilized `iou`, duplicate-suppression, and `new_track_thresh` values were selected from
controlled ablations on representative exam-room footage. Nested suppression is deliberately
limited to strongly contained, center-aligned boxes with materially different areas; adjacent
people that merely overlap are retained. Keep `conf=0.10` and `track_low_thresh=0.10` so
ByteTrack can still associate partially occluded people; do not raise either value merely to
reduce ID counts.

Seat-assignment weights must be non-negative with a positive sum. `min_score` is in `[0,1]`;
expansion, confirmation/release durations, switch margin and switch duration are non-negative.
Weights are applied exactly as configured and are not silently normalized.

## Phase 6 Action Recognition

The deployed runtime YAML files add `action_recognition` with the verified R3
checkpoint at `/models/action/r3/model.pth`, SHA-256 validation, TSM-ResNet50,
8 segments, 224×224, and a 4,000 ms timestamp window. Phase 6.5 measured that
TSM FP16 is about three times slower than FP32 on the GTX1650 TU117 and produces
non-finite B2/B8 output. The validated GTX profile therefore uses FP32 with:

```yaml
action_recognition:
  enabled: false
  precision: fp32
  scheduling:
    prediction_stride_ms: 1500
    max_batch_size: 2
    max_queue_size: 1
    max_prediction_age_ms: 2000
    min_inference_interval_ms: 200
```

The action branch remains disabled by default so runtime activation is an
explicit deployment choice; its enabled performance path passed on GTX1650.
The RTX3060 block remains a future, unvalidated profile and must not be treated
as benchmark evidence. Scheduler validation requires positive stride/batch/age,
batch `<=8`, queue size exactly one and non-negative minimum interval. Ready
requests use per-proposal timestamp stride, latest-equivalent replacement,
oldest-prediction-first selection and age expiry. These values do not change
the four-second clip span.

Single ROI: `expand_x=.04`, `expand_top=.03`, `expand_bottom=.08`, minimum
128×128 source pixels. Pair ROI: `.025`, `.02`, `.06`, minimum 128×128.
Adjacency uses row center-Y tolerance `.75`×Seat height and maximum horizontal
gap `2.5`×Seat width. The config validator fixes the R3 segment count, input
size and clip duration to the audited checkpoint contract. See
`docs/TSM_RUNTIME_CONTRACT.md` for formulas/sampling limits and
`docs/PHASE6_5_ACTION_PERFORMANCE_REPORT.md` for the full ablation.

## Phase 7 Event Detection

Both runtime profiles contain a centralized but disabled `event_detection`
block. A production-path pilot calibration now exists for S00–S04, but its
refined Macro Event F1 is 0.1064 with 1.01 false events/minute and zero
communicating true positives, so its searched values
remain an evaluation artifact and are not copied into deployment defaults.
Enabling an incomplete block fails validation, and event detection also
requires action recognition to be enabled.

After running `python -m app.cli.calibrate_events` on a versioned labeled
development/validation manifest, review the event metrics before separately
approving and copying the frozen `event_detection` mapping into a deployment
profile. Calibration artifacts never enable production automatically:

```yaml
event_detection:
  enabled: true
  max_discontinuity_ms: 5000
  dedup_temporal_iou: 0.30
  dedup_max_gap_ms: 2000

  suspicious_looking:
    proposal_types: [SINGLE]
    smoothing: {type: ema, alpha: CALIBRATED}
    start_threshold: CALIBRATED
    keep_threshold: CALIBRATED
    min_active_ms: CALIBRATED
    end_grace_ms: CALIBRATED
    merge_gap_ms: CALIBRATED

  communicating:
    proposal_types: [PAIR]
    # same calibrated fields
  exchange_object:
    proposal_types: [PAIR]
    # same calibrated fields
  using_phone_cheat_sheet:
    proposal_types: [SINGLE]
    # same calibrated fields
```

Validation requires `start_threshold > keep_threshold`, alpha in `(0,1]`,
positive minimum duration, non-negative grace/merge durations and the exact
Single/Pair routing above. Search accepts only `development` and `validation`
partitions and rejects `final`, `final_test` and `test`. The pilot manifest is
`docs/event_calibration/exam_dataset_pilot.yaml`; its generated artifacts are in
`docs/event_calibration/artifacts/`. The example manifest remains schema
documentation only and must not be used to select parameters.
