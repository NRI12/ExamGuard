# ExamGuard Phase 6 — Action Recognition Runtime

Updated: 2026-09-23

> This report is the preserved Phase 6 pre-stabilization baseline. Phase 6.5
> resolves the GTX1650 performance failure with FP32 scheduling; see
> [PHASE6_5_ACTION_PERFORMANCE_REPORT.md](PHASE6_5_ACTION_PERFORMANCE_REPORT.md).

```text
PHASE 6 ACTION RECOGNITION: NOT PASSED
IMPLEMENTATION + CPU END-TO-END + REFERENCE EQUIVALENCE: PASSED
GTX1650 CUDA/FP16 INTEGRATION: PASSED
TARGET-GPU TRACKING-FRESHNESS GATE: FAILED (5.43 FPS < 10 FPS)
```

The Phase 6 raw-action branch is implemented and tested. CUDA/FP16 inference is
now verified through NVIDIA Container Toolkit on the target GTX1650 Max-Q, but
the three-clip production-pipeline benchmark drops aggregate offline throughput
from 38.97 FPS tracking-only to 5.43 FPS with action enabled. This is below the
configured 10-FPS minimum. Both production profiles therefore keep
`action_recognition.enabled: false`. This is a measured tracking-freshness
safety failure, not a missing action inference implementation.

## Training/runtime contract (report items 1–12)

1. Frozen sources: `ai_reference/r3_tsm_r50_k400_diff_final/{model.pth,tsm_kinetics_model.py,dataset_tsm_reference.py,r3_roi_config.json,training_config.json}`; external R3 clip generator and manifests in `exam_dataset_pilot` were inspected read-only.
2. Selected checkpoint: `model.pth`, SHA-256 `c5ef406cfe404d2883575d8dcf1306fd7bebc6d6b1ec723b5093f2a3e09b357c`.
3. Architecture: TSM-ResNet50 with 16 block-residual TemporalShift wrappers, `fold_div=8`, 5-way `new_fc`, per-segment logits averaged.
4. Temporal Difference: no input/auxiliary branch. `diff` means differential fine-tuning.
5. Temporal Attention: none in R3.
6. Class order: `normal`, `suspicious_looking`, `communicating`, `exchange_object`, `using_phone/cheat_sheet`.
7. Frames: 8 sampled RGB frames from a verified 4-second, 100-frame/25-FPS training clip.
8. Resolution: 224×224, stretched (no aspect-preserving crop).
9. Evaluation sampling: center of each of 8 equal frame-count segments; runtime takes nearest video-timestamp frame to each 500-ms segment center, subject to tolerance. The latter handles dropped/irregular analysis frames but is not mathematically identical for arbitrary frame cadence.
10. Normalization: float32 RGB/255, ImageNet mean `(0.485,.456,.406)`, standard deviation `(.229,.224,.225)`, layout `[B,8,3,224,224]`.
11. Color behavior: source OpenCV BGR → RGB before normalization. Logits receive one softmax in the adapter.
12. Detailed contract: [TSM_RUNTIME_CONTRACT.md](TSM_RUNTIME_CONTRACT.md).

## Implementation (report items 13–28)

13. Created: `backend/app/ai/action_recognition/{types,roi,proposals,buffer,preprocessing,model,adapter,runtime}.py`, package init, `backend/app/cli/validate_action_recognition.py`, `backend/tests/test_action_recognition.py`, this report, and the runtime contract.
14. Modified: monitoring config/worker/manager/publisher and diagnostics domain; both runtime YAML profiles; Docker Compose read-only R3 mount; monitoring frontend types/page; architecture/API/config/current-state docs; one monitoring publisher regression test.
15. Single proposal: only `ASSIGNED` Track with SessionCandidate UUID; stable `single:<SC_UUID>`. Current Track ID is debug metadata only. TENTATIVE/UNASSIGNED produce none.
16. Seat adjacency: geometrically group row center-Y within `.75 × larger Seat height`; sort each row by center-X and connect only consecutive Seats whose gap is `≤2.5 × wider Seat width`.
17. Pair proposal: only two assigned candidates in adjacent Seats; canonical `pair:<lower_UUID>:<higher_UUID>`, actor metadata sorted by UUID; no all-pairs combinatorics.
18. Single ROI: actor bbox, expand each horizontal side `.04 × width`, top `.03 × height`, bottom `.08 × height`; centered minimum 128×128 source pixels, clamp, even-pixel origin/size, Lanczos stretch to 224×224.
19. Pair ROI: union of both visible actor boxes and intervening space, then expand horizontal `.025`, top `.02`, bottom `.06`; otherwise same crop operation. This uses verified R3 ROI config.
20. Buffer: per stable proposal ID, timestamped RGB uint8 224×224 frames only; GTX max 23 frames/proposal, RTX max 28; no full-frame history or database writes.
21. Gap/reset: duplicate timestamps ignored, backward timestamps clear one buffer, >750-ms evidence gap clears it, absent proposals expire after 4,750 ms video time; seek clears all buffers/stride timers/queues and generation-gates old output.
22. Model lifecycle: one registry entry per checkpoint/device/precision shared by sessions; SHA and strict state loading before monitoring startup; serialized inference lock; cleanup remains session-local.
23. Device/FP16: CUDA profiles request FP16; CPU diagnostic uses FP32. Production YOLO11n and R3 both loaded and inferred through PyTorch `2.7.1+cu126` on `cuda:0` (GTX1650 Max-Q, compute capability 7.5) without OOM or runtime error. Host `nvidia-smi` still lacks `/dev/nvidia*`, but the NVIDIA Container Toolkit exposes the GPU correctly inside containers.
24. Batching: same-shape ready proposals are batched; adapter microbatches at max 4 (GTX) or 8 (RTX). Diagnostics report logical request size, which may exceed a microbatch.
25. Backpressure: independent capacity-one latest-frame and latest-inference queues; replace stale action requests, never build a multi-second backlog. Tracking thread only offers a frame reference; ROI and TSM run in background threads. Action latency is subordinate to tracking freshness.
26. ActionPrediction: stable proposal ID/type, SessionCandidate IDs, Seat codes, video timestamp, five raw probabilities, top class/confidence, model name. No full Candidate or tensor payload, no PostgreSQL persistence.
27. WebSocket: separate low-frequency `action_prediction` and `action_error` messages; publisher coalesces by message type so high-FPS tracking does not erase action metadata. RBAC `tracking.read` unchanged.
28. Diagnostics: single/pair counts, ready/active buffers, ROI frame count, total/rate of predictions, preprocess/inference/action-pipeline mean/P95, logical batch size mean/P95, queue depth, stale drops, device; developer drawer only.

## Verification (report items 29–35)

29. Tests added: ROI geometry/borders/union, deterministic adjacency, stable proposal IDs across Track fragmentation, UNASSIGNED exclusion, buffer coverage/duplicates/gap/pause/expiry/seek, config validation, frozen sampling/preprocessing/model equivalence, real checkpoint and real clip smoke, raw prediction lifecycle, and action-preserving WebSocket coalescing.
30. Preprocessing equivalence on actual 100-frame R3 ROI clip `S00_EASY_NORMAL_0013`: sampled indices `[5,18,30,43,55,68,80,93]`; max absolute tensor difference `0.0`.
31. Frozen-reference vs production inference on that clip: max absolute softmax probability difference `0.0`; same top-1 `normal`. This is an already-cropped R3 clip, not a dynamic-ROI accuracy claim.
32. Pytest: **106 collected, 105 passed, 1 skipped**. The regular suite skipped its opt-in CUDA test. The production GPU image does not ship pytest, so CUDA integration was instead exercised end-to-end by the production validation CLI: real YOLO11n, ByteTrack, Seat Identity and real R3 FP16 inference all completed on three videos without error.
33. Ruff: `ruff check backend/app backend/tests` **passed**; `git diff --check` **passed**.
34. Mypy: `mypy app` **passed, 101 source files**.
35. Frontend: Vitest **10 files / 28 tests passed**; TypeScript typecheck and Vite production build **passed** (124 modules).

## Real-video integration and throughput (report items 36–46)

The CLI ran production YOLO11n, ByteTrack, SeatAssignmentEngine, proposal/ROI
buffer and the real R3 checkpoint on three independently uploaded 60-second
camera clips, sampled at 5 FPS (301 frames per clip/mode). It compared the same
video interval with action disabled vs enabled on CPU. These are offline
throughput runs, not real-time browser playback measurements. The manifest's
scenario tags cover seated people, leaning/turning, adjacent people, partial
occlusion and extra people; frame-level action labels were not used to score F1.

| Clip | Tracking-only FPS | Tracking+action FPS | Single / pair IDs with prediction | Predictions | Preprocess mean/P95 ms | TSM mean/P95 ms | Action pipeline mean/P95 ms | Stale drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A adjacent/extra | 31.34 | 8.10 | 6 / 5 | 113 | 15.06 / 27.60 | 1586.37 / 2614.25 | 2139.57 / 3437.80 | 36 |
| B occlusion/extra | 30.73 | 8.50 | 6 / 5 | 107 | 30.91 / 59.46 | 3152.67 / 3763.93 | 3692.68 / 5090.56 | 24 |
| C seated/turn | 31.46 | 8.40 | 6 / 5 | 111 | 20.18 / 39.98 | 2296.76 / 3150.87 | 2901.75 / 4052.74 | 22 |

36. Clips: `clip_a_adjacent_extra_person`, `clip_b_partial_occlusion_extra_person`, `clip_c_stable_seated` from `docs/seat_identity_validation/manifest.yaml`; each 60 s.
37. Unique single proposals with ≥1 prediction: **18** across three sessions (6 each).
38. Unique pair proposals with ≥1 prediction: **15** (5 each). At clip C end, only 5 singles/3 pairs were currently active; historical counts are distinct.
39. Raw ActionPredictions: **331** (113+107+111); examples include `A01 normal=.7595` in clip A and `A03 normal=.9090` in clip C. These are model outputs, not behavior conclusions.
40. TSM preprocessing latency: measured per-clip mean/P95 in table; no unmeasured global percentile is claimed.
41. TSM inference latency: per-clip mean/P95 in table. CPU only.
42. Action-pipeline latency: queue wait + preprocess + model; per-clip mean/P95 in table. ROI crop cost is outside this metric and appears in overall tracking+action throughput.
43. Aggregate prediction rate: **3.05 predictions/s** over 108.41 s of combined processing time. Per-clip rates: 3.04, 3.02, 3.10/s.
44. Aggregate offline throughput: tracking-only **31.17 FPS**, tracking+action **8.33 FPS**. Mean tracking-frame compute time grew from ~24 ms to ~98–102 ms under concurrent CPU load. The CLI's offline video-time lag is not a valid live-playback lag metric and is **not reported as such**.
45. CPU-run VRAM was not applicable. The later GTX1650 CUDA/FP16 run measured VRAM, utilization, power and temperature as described below. CPU/RAM were not sampled by this CLI, so no values are claimed.
46. Stale action requests dropped: **82** (36+24+22); queue-depth snapshot at clip completion was 0. Queue capacities are one frame and one inference batch, but peak queue depth was not measured.

### GTX1650 CUDA/FP16 rerun — 2026-09-23

The same three 60-second clips and production components were rerun at 5-FPS
sampling in `examguard-backend-gpu:local`, using PyTorch `2.7.1+cu126`, CUDA
12.6, driver 580.178.04 and the real 4-GB GTX1650 Max-Q. A separate local CUDA
container sampled NVIDIA telemetry every 500 ms throughout the run.

| Clip | Tracking-only FPS | Tracking+action FPS | Predictions | TSM mean/P95 ms | Action pipeline mean/P95 ms | Errors / stale drops |
|---|---:|---:|---:|---:|---:|---:|
| A adjacent/extra | 39.07 | 5.76 | 260 | 742.50 / 1338.86 | 841.09 / 1373.46 | 0 / 0 |
| B occlusion/extra | 39.77 | 5.03 | 305 | 1479.55 / 1879.22 | 1516.38 / 1923.74 | 0 / 0 |
| C seated/turn | 38.12 | 5.56 | 272 | 1176.54 / 1858.35 | 1324.67 / 1906.51 | 0 / 0 |

- Aggregate: 903 frames per mode, **38.97 FPS tracking-only → 5.43 FPS tracking+action**.
- Raw predictions: **837**, or **5.04 predictions/s** over 166.22 seconds of action-enabled processing.
- Proposal coverage: 6 Single and 5 Pair stable IDs produced predictions in each clip.
- Model lifecycle: one R3 model load; no CUDA error, action error or OOM.
- Telemetry peak observed: **100% GPU**, **531 MiB / 4096 MiB VRAM**, **30.90 W**, **71°C**. Mean telemetry was not persisted, so no mean is claimed.
- The test is an offline production-component benchmark. Its `analysis_lag_ms`
  calculation is not a browser/live-scheduler lag measurement and is not used
  for acceptance. The achieved compute throughput itself is already below the
  10-FPS minimum, so the tracking-freshness gate fails.

## Remaining risks and scope (report items 47–50)

47. Known failure modes: dynamic runtime bbox crops differ from the training generator's clip-static annotation union and omit its H.264 re-encoding; temporal timestamp centers can diverge from frame-count centers under irregular frame drops; malformed Seat geometry can make wrong neighbors; short identity gaps can withhold predictions. GTX1650 CUDA/FP16 is now measured, but YOLO and TSM contend for the GPU and reduce throughput to 5.43 FPS, below the 10-FPS minimum. Live scheduler/playback lag and RTX3060 performance remain unmeasured. The earlier Seat-Stable Identity acceptance gaps (supervisor-walk/stand-leave source coverage and persisted room recalibration) also remain.
48. Head Pose was **not** added.
49. Desk/Object detector was **not** added.
50. Event generation, smoothing, thresholds, Evidence and misconduct conclusions were **not** added. Phase 6 stops at raw probabilities; Phase 7 was not started.

Activation gate: first reduce YOLO/TSM contention (for example by lowering
action cadence/batch pressure or scheduling inference outside tracking's
critical GPU path), then rerun target-rate live scheduler/playback validation
and verify at least 10 tracking FPS with bounded lag. Also verify FP16 numerical
tolerance and inspect dynamic ROI quality before setting
`action_recognition.enabled: true`. Do not infer improved F1 from this
integration benchmark; scientific evaluation requires the same split, ROI
protocol and checkpoint as the controlled R3 experiment.
