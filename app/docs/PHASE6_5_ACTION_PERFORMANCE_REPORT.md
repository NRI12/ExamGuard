# ExamGuard Phase 6.5 — Action Runtime Performance Stabilization

Updated: 2026-09-23

```text
PHASE 6: FUNCTIONAL PASS
PHASE 6.5: PERFORMANCE PASS
ACTION RECOGNITION: PASS (raw ActionPrediction runtime)
```

The selected GTX1650 configuration keeps the production profile opt-in
(`enabled: false`) but clears the performance gate when enabled: all three
60-second realtime-paced runs sustained 12.51 analysis FPS, had 100–104 ms lag
P95, non-increasing end-to-end lag, 160–320 ms ActionPrediction age P95, no
action error/OOM, and a final queue depth of zero. This phase does not claim
behavior-classification accuracy for dynamic runtime ROIs and does not create
Events.

Raw reproducibility artifacts are in `docs/action_performance/`.

## 1–7. Bottleneck and isolated TSM profile

1. **Existing bottleneck.** The Phase 6 runtime immediately inferred all ready
   proposals in FP16 microbatches. On this GTX1650 TU117, TSM FP16 is about
   three times slower than FP32, and B2/B8 produced non-finite outputs. Long,
   repeated TSM kernels contended with YOLO; the known Phase 6 result was 38.97
   FPS tracking-only versus 5.43 FPS tracking+action.
2. **Isolated B1/B2/B4/B8.** Real 8-frame 224×224 RGB samples from
   `S00_EASY_NORMAL_0013__compact_actor_context_roi.mp4`, the real R3 checkpoint,
   PyTorch 2.7.1+cu126/CUDA 12.6, 3 warmups + 8 measured iterations:

| Precision | B | Preprocess mean/P50/P95 ms | H2D mean/P95 ms | Forward mean/P50/P95 ms | Total mean/P95 ms | clips/s | allocated/reserved/peak MiB | Finite |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 | 1 | 1.54/1.53/1.63 | 0.90/0.95 | 58.71/58.80/58.98 | 61.35/61.59 | 16.30 | 102.67/228/182.30 | yes |
| FP32 | 2 | 3.34/3.42/3.53 | 1.76/1.84 | 114.99/114.99/115.28 | 120.32/120.82 | 16.62 | 107.26/350/268.51 | yes |
| FP32 | 4 | 7.04/6.91/7.62 | 3.48/3.66 | 217.20/217.16/217.77 | 228.00/228.79 | 17.54 | 116.45/962/434.95 | yes |
| FP32 | 8 | 14.54/14.45/15.85 | 6.49/6.66 | 428.49/429.01/429.76 | 449.79/452.34 | 17.79 | 134.83/980/771.83 | yes |
| FP16 | 1 | 1.63/1.63/1.69 | 0.98/1.09 | 182.49/182.49/182.54 | 185.37/185.54 | 5.39 | 58.52/116/98.33 | yes |
| FP16 | 2 | 3.54/3.47/4.24 | 1.85/1.98 | 344.72/344.71/344.76 | 350.39/351.09 | 5.71 | 60.82/172/140.44 | **no** |
| FP16 | 4 | 6.93/6.75/7.90 | 3.69/4.07 | 668.01/668.02/668.05 | 678.94/680.01 | 5.89 | 65.41/244/227.66 | yes |
| FP16 | 8 | 13.42/13.28/14.49 | 6.91/7.36 | 1308.45/1308.66/1309.05 | 1329.08/1331.05 | 6.02 | 74.60/426/393.10 | **no** |

3. **FP32 vs FP16 numerics.** B1 max logit difference
   `0.0028474331`, probability difference `0.0000871848`, same top-1. B4 is
   `0.0028476715`, `0.0000871867`, same top-1. B2 and B8 are not comparable
   because all FP16 output probabilities are non-finite; independent reruns
   reproduced the fault.
4. **Latency comparison.** FP16 forward is 3.11× slower at B1, 3.00× at B2,
   3.08× at B4 and 3.05× at B8. GTX therefore uses FP32. Production also
   rejects non-finite output before publishing it.
5. **Active Single load.** Tuning clip A averaged 5.43 active Singles (maximum
   6). Selected 60-second runs averaged 5.51, 5.90 and 5.73.
6. **Active Pair load.** Tuning clip A averaged 4.03 active adjacent Pairs
   (maximum 5). Selected runs averaged 4.11, 4.88 and 4.54. No all-pairs were
   introduced.
7. **Old inference frequency.** The Phase 6 setting had a nominal 2000-ms
   proposal stride, but every ready proposal was submitted immediately as one
   large logical request; there was no explicit fair compute budget between
   readiness and model inference.

The production component profile below is from selected clip B (60 seconds,
FP32/B2). ROI rows are per captured proposal frame; tensor/model rows are per
scheduled batch. Crop, ROI resize, RGB conversion, tensor assembly,
normalization and H2D are explicitly separated. `input_resize_ms` is zero
because runtime ROI preparation already produces 224×224 input; the unchanged
bilinear preprocessing operation detects that no spatial resize is required.

| Component | Placement | Mean / P50 / P95 ms |
|---|---|---:|
| ROI preparation total | CPU | 2.036 / 2.198 / 2.745 |
| CPU crop | CPU view | 0.009 / 0.008 / 0.016 |
| ROI resize | CPU OpenCV | 2.001 / 2.170 / 2.705 |
| BGR→RGB | CPU OpenCV | 0.027 / 0.021 / 0.048 |
| Tensor assembly | CPU | 2.372 / 2.134 / 4.172 |
| Input bilinear resize | CPU | 0 / 0 / 0 |
| Normalization | CPU | 1.712 / 1.660 / 2.852 |
| Host→device | CUDA transfer | 5.179 / 5.322 / 9.934 |
| TSM forward | CUDA | 104.674 / 117.373 / 126.561 |
| Softmax/postprocess | CUDA→CPU | 2.358 / 0.129 / 19.486 |
| Model batch total | mixed | 119.065 / 133.257 / 143.268 |

## 8–20. Runtime changes and verification

8. **Scheduler.** `ActionScheduler` sits between timestamp buffers and TSM. It
   owns per-proposal stride, latest-ready state, maximum age, minimum batch
   interval and maximum batch size.
9. **Fairness.** Deterministic never-predicted-first, then oldest last-prediction
   timestamp, then clip timestamp/proposal ID. The selected videos produced
   predictions for all 6 Single and all 5 adjacent Pair IDs per clip.
10. **Stale behavior.** A newer ready clip replaces the older equivalent clip;
    requests older than `max_prediction_age_ms` are discarded. Replacements
    and age expiry are separate metrics. The selected runs had 0 expired/stale
    requests and 5,568 latest replacements.
11. **Queue semantics.** Frame and inference queues remain capacity one. There
    is at most one model batch in flight; final depth was 0 for every selected
    and live run.
12. **Model instances.** `ActionModelRegistry` returns one adapter per resolved
    checkpoint/device/precision. A regression test checks reuse. Each validation
    process reported `model_load_count=1`.
13. **Worker instances.** One ActionRecognitionRuntime (one buffer thread and
    one inference thread) is created inside one monitoring worker. Existing
    manager duplicate-start prevention remains tested; WebSocket reconnect does
    not create a worker. Pause schedules no repeated timestamp, seek clears
    buffers/scheduler/queues and generation-gates in-flight output, and stop
    joins both action threads.
14. **Files modified.** Monitoring config/worker diagnostics/domain/schema,
    GTX/RTX YAML, action adapter/buffer/preprocessing/ROI/runtime, frontend
    diagnostics types, project architecture/config/API/current-state docs, and
    action tests/report.
15. **Files created.** `action_recognition/scheduler.py`,
    `cli/profile_action_model.py`, this report, and raw JSON files under
    `docs/action_performance/`.
16. **Tests added/revised.** Scheduler eligibility and timestamp stride,
    latest replacement, oldest-first fairness, B2 budget, Single/Pair coverage,
    max-age expiry/reset, runtime seek/stop, capacity-one config, model registry
    reuse, plus all existing pause/duplicate-worker/preprocessing/reference tests.
17. **Pytest.** Final result: **110 collected, 109 passed, 1 skipped**. The skip
    is the explicitly optional CUDA pytest; CUDA was exercised by the production
    profiler/validation CLIs.
18. **Ruff.** `ruff check backend/app backend/tests`: **passed**.
19. **Mypy.** `mypy app`: **passed, 103 source files**.
20. **Frontend.** Diagnostics types changed. Vitest: **10 files / 28 tests
    passed**; TypeScript check and Vite build passed, 124 modules transformed.

## 21–38. Controlled system experiments

All ablations use clip A, 30 seconds, 12.5-FPS sampling, the same detector,
tracker, Seat Identity, R3 checkpoint and seat layout. `min_inference_interval_ms`
is 200 and precision is FP32.

| ID | Stride ms | B | Tracking FPS | Action pred/s | Single mean/P95 ms | Pair mean/P95 ms | Forward P95 ms | Pipeline P95 ms | GPU mean/P95 | NVML peak MiB | Stale / final queue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYS-PERF-500-B2 | 500 | 2 | 17.12 | 7.13 | 1187/1440 | 1167/1440 | 115.15 | 130.64 | 80.76/89% | 889.94 | 0/0 |
| SYS-PERF-1000-B2 | 1000 | 2 | 17.26 | 6.93 | 1183/1440 | 1260/1444 | 128.22 | 142.73 | 79.13/88% | 889.94 | 0/0 |
| SYS-PERF-1500-B2 | 1500 | 2 | 19.15 | 5.63 | 1483/1680 | 1516/1640 | 129.32 | 142.58 | 77.38/87% | 889.94 | 0/0 |
| SYS-PERF-2000-B2 | 2000 | 2 | 22.21 | 4.27 | 1987/2160 | 2006/2204 | 129.28 | 140.73 | 74.50/88% | 889.94 | 0/0 |
| SYS-PERF-1500-B1 | 1500 | 1 | 24.24 | 3.57 | 2420/2768 | 2447/2720 | 58.18 | 72.34 | 73.68/— | 793.94 | 2/0 |
| SYS-PERF-1500-B4 | 1500 | 4 | 19.85 | 5.70 | 1468/1520 | 1512/1680 | 219.84 | 242.89 | 78.16/— | 1523.94 | 0/0 |

21. **Stride configurations:** 500, 1000, 1500 and 2000 ms were all executed.
22. **Batch configurations:** B1, B2 and B4 were executed at viable stride
    1500. B8 was isolated-profiled but rejected for the system matrix because
    its 429.76-ms FP32 forward P95 adds a long blocking kernel with little
    throughput benefit.
23. **Tracking-only:** 44.27 FPS on the same clip/30-second protocol.
24. **Previous bad result:** 38.97 FPS tracking-only → 5.43 FPS FP16 action,
    aggregated over the previous three 60-second Phase 6 runs at 5-FPS input.
25. **Selected tracking+action:** 60-second offline FPS was A 14.39, B 17.18,
    C 17.89 (minimum 14.39, simple mean 16.48). Realtime-paced FPS was A 12.508,
    B 12.509, C 12.512.
26. **Action rate:** A 6.17, B 6.80, C 6.42 predictions/video-second.
27. **Single cadence:** means 1554/1526/1566 ms; P95 1680/1680/1740 ms.
28. **Pair cadence:** means 1687/1484/1605 ms; P95 1680/1680/1680 ms.
    Large maxima in A/C include proposals that disappeared during identity
    gaps and later returned; they are not continuous-active starvation. P95 and
    all-ID coverage are the scheduler acceptance measures.
29. **TSM preprocess:** per-clip mean 9.82/9.26/6.74 ms; P95
    20.41/13.74/12.46 ms.
30. **TSM forward:** per-clip mean 122.43/104.67/101.45 ms; P95
    234.54/126.56/130.18 ms.
31. **Action pipeline:** per-clip mean 139.61/119.29/114.24 ms; P95
    266.15/143.49/145.42 ms.
32. **GPU:** offline selected mean 87.40/82.14/82.53%, with a 100% sampled peak;
    realtime mean 56.45/60.37/58.31%, P95 at most 71%. High utilization did not
    cause freshness failure.
33. **Memory:** maximum observed NVML VRAM was 1405.94 MiB in offline clip A;
    realtime peak was 887.94 MiB and PyTorch peak allocation about 337 MiB. No
    OOM. Process RSS peak was 1673.77 MiB. Buffers store only cropped 224×224
    RGB uint8 ROI frames, never duplicated 1080p frames.
34. **Analysis lag:** in 180 seconds of realtime-paced validation, lag P95 was
    100.73/99.98/104.20 ms; maximum 511.80/466.00/488.73 ms. First-to-last
    window growth was -2.48/-10.68/-12.34 ms, so lag did not accumulate.
35. **ActionPrediction age:** source-timestamp age mean 138.59/99.41/109.30 ms;
    P95 320/160/320 ms.
36. **Stale drops:** selected runs had 0 queue/generation/age-expired drops.
    Latest-equivalent replacements were counted separately and are expected.
37. **Final queue:** 0 for all runs; every runtime reached idle before stop.
38. **Selected GTX config:** FP32, stride 1500 ms, max batch 2, queue 1,
    max age 2000 ms and minimum inference interval 200 ms. B1 lost coverage;
    B4 doubled kernel P95 and raised VRAM without useful cadence gain.

## 39–45. Acceptance and limitations

39. **Live duration.** 180 seconds realtime-paced across three production-chain
    video runs. Native browser playback itself was not automated in this
    headless environment; the validator paced the same decoder/runtime against
    the video clock and measured source-vs-analysis lag and prediction age.
40. **Known limitations.** Dynamic tracker ROI remains distribution-shifted
    from clip-static training annotation ROI; timestamp sampling can differ on
    irregular cadence; proposal disappearance inflates interval maxima; no
    independent supervisor-walk or complete stand/move/sit source exists; no
    action accuracy/F1 claim is made; RTX3060 settings remain unbenchmarked.
41. **Scientific preprocessing:** unchanged: 8 RGB uint8 frames, `/255`,
    bilinear 224×224 with `align_corners=False`, ImageNet mean/std and
    `[B,8,3,224,224]`. Real-clip tensor max difference remains `0.0`; FP32
    probability max difference remains `0.0`, same top-1 `normal`.
42. **Head Pose:** not added.
43. **Desk/Object detector:** not added.
44. **Event FSM:** no smoothing, threshold, hysteresis, Event/Evidence write or
    misconduct conclusion was added.
45. **Decision:** **PHASE 6.5 PERFORMANCE PASS.** The minimum 10-FPS GTX1650
    gate, bounded-lag, bounded-queue, Single/Pair coverage, lifecycle,
    correctness and no-OOM requirements pass. The profile remains disabled by
    default so activation is explicit and cannot be confused with a scientific
    dynamic-ROI accuracy claim.

Stop at Phase 6.5. Phase 7 was not started.
