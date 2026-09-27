# TSM R3 runtime contract

The selected deployment checkpoint is
`ai_reference/r3_tsm_r50_k400_diff_final/model.pth` (SHA-256
`c5ef406cfe404d2883575d8dcf1306fd7bebc6d6b1ec723b5093f2a3e09b357c`).
The container mounts this package read-only at `/models/action/r3`.

## Verified training/evaluation artifacts

- `model.pth`: `model_state_dict`, architecture `TSM-ResNet50`, 8 segments,
  224×224 input, five ordered class names, `compact_actor_context_roi` view.
- `tsm_kinetics_model.py`: ResNet-50; TemporalShift wraps `conv1` in each of
  16 residual blocks (`blockres`, `fold_div=8`); frame logits are averaged over
  8 segments. `new_fc` is 5×2048. The checkpoint loads with strict state-key
  matching. Dropout 0.6 is inactive in eval mode. No Temporal Attention exists.
- `training_config.json`: Kinetics-400 initialization, differential fine-tuning.
  “diff” in the package name means differential fine-tuning, **not** an input
  temporal-difference branch. No external difference branch was added.
- `dataset_tsm_reference.py`: decode OpenCV BGR→RGB, sample the center frame of
  each of 8 equal frame-count segments, convert uint8 RGB to float32/255,
  bilinear stretch to 224×224 (`align_corners=False`), then normalize channels
  with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
  `(0.229, 0.224, 0.225)`. Model input is `[B,8,3,224,224]`; model returns
  logits; softmax is applied exactly once in the adapter. Eval has no random
  flip/brightness/contrast augmentation.
- The R3 clip manifest and `04d_generate_r3_r4_rois.py` verify 100 frames at
  25 FPS = **4,000 ms**, not an inferred window. The dataset generation
  configuration has 2-second window stride; runtime inference stride is
  likewise 2,000 ms. The original reference sampler uses frame-count segment
  centers; runtime uses video-timestamp segment centers for variable analysis
  FPS. These coincide for regularly sampled source clips within the configured
  sampling tolerance, but are not guaranteed identical under irregular drops.

Class order is fixed in one adapter constant:
`normal`, `suspicious_looking`, `communicating`, `exchange_object`,
`using_phone/cheat_sheet`.

## ROI and proposal contract

The frozen R3 ROI generator starts with the union of annotated actor boxes
over a complete short clip. Runtime starts with current ByteTrack boxes for
ASSIGNED identities and unions both boxes for a pair. This dynamic/static
geometry difference is an unavoidable deployment distribution shift and is
**not** claimed to be scientifically equivalent to the training ROI protocol.
For both, the exact crop geometry is:

1. Clamp actor union to source pixels; reject zero visible area.
2. Expand left/right by `expand_x × union_width`, top by
   `expand_top × union_height`, bottom by `expand_bottom × union_height`.
3. Enforce a centered minimum of 128×128 pixels, clamp to the source frame,
   and round origin/size down to even pixels as in the frozen generator.
4. Crop BGR source and stretch with Lanczos to 224×224; convert BGR→RGB.

Single R3 expansion is `(x=.04, top=.03, bottom=.08)`; pair expansion is
`(x=.025, top=.02, bottom=.06)`. These are from `r3_roi_config.json`, not the
larger engineering fallback in the Phase 6 prompt. The tracker bbox itself
is never changed. The training generator re-encoded its static ROI clips as
H.264; the runtime buffers uncompressed ROIs, another minor pixel-level shift.

Only an `ASSIGNED` SessionCandidate generates `single:<SC_UUID>`. Pairs require
two assigned candidates in consecutive left/right Seats in a geometry-derived
row; they use `pair:<lower_UUID>:<higher_UUID>`. Rows are grouped when center-Y
distance is at most 0.75×the larger Seat height. Consecutive seats connect only
when horizontal gap is at most 2.5×the wider Seat width. This is deterministic
for a fixed layout; inaccurate Seat geometry can make the adjacency wrong.
Track IDs are debug metadata only, never proposal identity.

## Buffer, lifecycle and resource bounds

Each proposal stores at most `clip_span_ms // capture_interval_ms + 3`
RGB 224×224 uint8 ROIs. GTX profile: capture every 200 ms, max 23 ROIs per
proposal (about 3.46 MiB before Python/container overhead); RTX profile:
every 160 ms, max 28 (about 4.21 MiB). At 30 singles plus 29 pairs the GTX
worst-case pixel buffer is roughly 204 MiB. Full 1080p frames are never copied
into per-proposal history. Timestamps must increase; duplicate timestamps are
ignored, backward timestamps reset that proposal, and a gap over 750 ms resets
it. An absent proposal expires after 4,750 ms of video time. Short Track
fragmentation can retain the same proposal buffer via SessionCandidate ID;
longer gaps restart it.

Eight target timestamps are the centers of equal 500 ms segments in the last
4,000 ms. Each target uses the nearest captured ROI only if within configured
tolerance (GTX 180 ms; RTX 140 ms). Otherwise no clip/prediction is fabricated.
Pause creates no frames and no wall-clock advancement. Seek increments runtime
generation and clears action buffers, queue, stride timers and old results;
the manager rejects stale-generation messages. Stop clears all session-local
state and joins the background threads.

The tracking thread offers only its latest source frame to a capacity-one
action-frame queue. A separate buffer thread crops/samples ROIs and offers the
latest ready proposal batch to a capacity-one inference queue. Old requests
are dropped; there is no unbounded work backlog. One lazily registered R3 model
per checkpoint/device/precision is shared across sessions and serialized by a
model lock. Startup validates SHA and strict state keys before monitoring
runs. CUDA uses FP16; CPU diagnostic mode uses FP32. The adapter internally
microbatches according to profile size (GTX 4, RTX 8). GPU/FP16 parity has
not yet been measured on the current host because its NVIDIA driver is not
available. Consequently `action_recognition.enabled` remains `false` in both
production profiles; enabling it requires a target-GPU freshness benchmark.

## Equivalence gate and limits

The production implementation is compared against the frozen dataset sampler
and TSM model source in tests. A 100-frame real R3 clip comparison reports
identical indices `[5,18,30,43,55,68,80,93]`, preprocessing max absolute
error `0.0`, softmax probability max absolute error `0.0`, and the same top-1.
This verifies inference math on an already-cropped R3 clip, **not** the
scientific accuracy of dynamically generated deployment ROIs. No F1 gain or
misconduct conclusion is inferred from system-integration smoke tests.
