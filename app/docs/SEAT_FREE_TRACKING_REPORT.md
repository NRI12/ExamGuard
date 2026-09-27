# Seat-Free Stable Tracking Report

Updated: 2026-09-24

```text
SEAT-FREE STABLE TRACKING: NOT PASSED
```

The implementation, zero-Seat integration path, motion recovery, dynamic proposals, tests, and GTX1650 performance gate pass. The overall acceptance gate remains **NOT PASSED** because the available annotations are static actor regions rather than dense MOT ground truth, the clips do not cover a complete stand/move/sit sequence or representative supervisor crossing, and EXP-TRACK-B did not encounter an ambiguous recovery where appearance could demonstrate identity value. IDF1/HOTA and a learned ReID value claim would therefore be fabricated.

## 1–9. Seat dependency, architecture, domain, and recovery

1. Existing dependency: readiness required active Seats plus SessionCandidates; action Single/Pair proposals required assigned SessionCandidates and Seat adjacency; overlay labels assumed Seat/Candidate context.
2. Mandatory files found: `app/features/sessions/service.py`, `app/features/monitoring/service.py`, `app/ai/action_recognition/proposals.py`, `app/ai/action_recognition/runtime.py`, `app/monitoring/worker.py`, `SessionReadinessPanel.tsx`, `TrackingCanvas.tsx`, and `MonitoringPage.tsx`.
3. Seat optionality: readiness/start now require an active Room and source video only. Empty `SeatIdentityContext` is valid; Seat assignment can enrich a logical Track but cannot create or block it.
4. Default identity path: YOLO11n → ByteTrack → LogicalTrackManager → motion/spatial recovery → optional sparse appearance recovery → Stable Actor ID.
5. Domain: bounded `ActorMemory` stores actor ID, current raw ID, bounded raw-ID/center histories, source timestamps, bbox, confidence summary, velocity, and optional normalized prototype. Nothing is persisted.
6. State machine: new observation → ACTIVE; missing current raw ID → LOST; confident association → ACTIVE on the new raw ID; source-time age over the gate → EXPIRED and removal from recovery memory. Seek/reset clears all state.
7. Formula: `0.55*position + 0.25*motion_direction + 0.20*scale`, after hard time/position/scale gates. Constant velocity uses normalized centers per source millisecond.
8. Selected recovery thresholds: max lost 5,500 ms; max normalized position distance 0.18; minimum area-scale similarity 0.40; minimum score 0.55; ambiguity margin 0.08. These were selected from measured same-region fragmentation transitions (observed gap up to 5,440 ms, scale 0.432, score about 0.566), with lower-quality scale cases left separate.
9. Ambiguity: when the best and second motion candidates differ by less than 0.08, motion makes no assignment. Appearance, when enabled, must also clear similarity 0.72, combined score 0.68, and combined margin 0.06. Matching is deterministic one-to-one greedy.

## 10–18. Appearance / sparse ReID experiment

10. Native features: unavailable to the current custom predictor/adapter path. Ultralytics 8.4.155 supports feature pass-through in compatible predictor integration, but ExamGuard's `YOLO.predict()` result consumed by `ByteTrackAdapter` does not expose it.
11. Model: no neural ReID model was added. EXP-TRACK-B uses internal CPU `hsv_histogram_v1`; deployment profiles keep `reid.enabled: false`.
12. Selection reason: no OSNet/torchreid dependency or licensed checkpoint exists, and adding an unmeasured neural model would violate the GTX1650/value gate. The histogram is a low-cost appearance ablation, not an OSNet claim.
13. Preprocessing: clamp crop, reject crops below 8×8, resize to 64×128, BGR→HSV, split 2×2, compute 16×8 H/S histogram per cell, concatenate, L2 normalize.
14. Similarity: cosine similarity, never appearance alone; geometry/time/scale candidates are established first and the combined score uses appearance weight 0.45.
15. Batching: current-frame crops are passed in one batch, maximum 16. Work is synchronous/latest-only, so no unbounded queue exists; excess eligible requests are dropped rather than queued.
16. Prototype policy: one seed after 800 ms of stable ACTIVE tracking and confidence ≥0.50. No per-frame update is performed. Recovered/ambiguous tracks cannot update or poison a prototype; a unit test asserts this.
17. Appearance invocation: one-time stable prototype seeding and motion-ambiguous new-ID recovery when eligible LOST prototypes exist.
18. Appearance skipped: stable actors after seed, unambiguous motion recovery, impossible spatial/scale/time gates, missing/invalid crops, no LOST candidate, disabled config, and ambiguous/unsafe combined appearance.

No external checkpoint, model source, or third-party model license applies to `hsv_histogram_v1`; it is repository code. Across the three clips it processed 26 crops in 16 batches (0.144 crops/requests per source second), produced zero appearance recoveries, and changed no identity result versus motion-only. It is therefore **not worth enabling** on present evidence.

## 19–24. Proposals, payload, UI, and compatibility

19. Dynamic neighbors: gate ACTIVE actors by vertical-center difference ≤0.75×max height, horizontal gap ≤2.5×max width, and area-scale similarity ≥0.40; sort plausible edges by center distance and enforce degree `K=2` for both endpoints.
20. Pair identity: canonical `pair:<lower_actor_id>:<higher_actor_id>`. Single identity is `single:<actor_id>`. Raw-ID fragmentation therefore does not rename a recovered proposal.
21. WebSocket: each Track now sends `actor_id`, `actor_state`, and `recovered`; raw `track_id` remains for debug/compatibility. Seat/Candidate identity stays nullable. Diagnostics add logical actor, recovery, appearance latency/request, and dynamic-pair fields. No embedding is sent.
22. Frontend: normal fallback label is the stable Actor ID; mapped Candidate/Seat context remains richer when available. Debug overlay may show Actor ID, raw Track ID, recovery marker, and optional Seat score.
23. Preflight: active Room + source video + AI runtime remain required. Seat layout and Candidate assignment are informational/optional and no longer disable Ready/Start.
24. Compatibility: existing Seat models, APIs, calibration UI, assignment engine, and legacy Seat proposal mode remain. Existing data requires no migration. Phase 7 persistence is still guarded to Seat mode because durable EventActor rows require SessionCandidate UUIDs.

## 25–29. Verification

25. Tests added: same raw ID stability; near/gap/far/scale recovery; ambiguity; one-to-one assignment; sparse invocation/skip rules; prototype non-poisoning; dynamic neighbor gates/limit/canonical ID; pair identity across fragmentation; zero-Seat monitoring start.
26. Pytest: `133 passed, 5 skipped, 5 warnings in 6.45s`.
27. Ruff: `All checks passed!`
28. Mypy: `Success: no issues found in 117 source files`.
29. Frontend: Vitest `11 passed` files, `31 passed` tests; TypeScript check and Vite production build pass with 125 modules transformed.

## 30–48. Real-video experiment

Videos, each 60 seconds at 1,920×1,080/25 FPS and sampled at 12.5 FPS (751 analyzed frames each):

- clip A: `66275e02-f8fd-496a-a0a5-f1e367e3ac59`, 6 annotated seated actors plus extra-person movement; adjacency/lean/partial occlusion.
- clip B: `8eaf3ab1-5d40-430f-a48d-b365e66bdf5e`, 6 annotated seated actors plus extra-person movement; adjacency/partial occlusion.
- clip C: `a45616f0-6eb6-498c-ab9a-0e7f6a1f70f2`, 6 annotated seated actors; lean/turn/adjacency.

The prior manual detector review on these sources contains 60/60 visible-person observations (100% at nine sampled timestamps), zero duplicate-person observations, and no isolated non-person track. This is not dense detection GT.

| Config | Offline FPS mean | Raw tracks | Logical actors | Explicit recoveries | Wrong recoveries | ReID calls/s | GPU mean | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack raw | 44.48 | 34 | — | — | — | 0 | 61.53% | 547.94 MiB |
| EXP-TRACK-A motion | 45.12 | 34 | 30 | 4 | 0 | 0 | 63.43% | 547.94 MiB |
| EXP-TRACK-B sparse appearance | 44.52 | 34 | 30 | 4 | 0 | 0.144 | 63.39% | 547.94 MiB |

Per-clip offline FPS for raw / motion / sparse appearance:

- A: 45.11 / 46.02 / 44.60
- B: 45.60 / 45.45 / 45.34
- C: 42.72 / 43.90 / 43.64

30. Benchmark videos: the three IDs above; artifacts are under `docs/seat_free_tracking/`.
31. People/tracks: 18 annotated seated actors across clips, 20 visible people in the broader tracking manifest, 34 raw IDs, 30 logical IDs.
32. Raw ByteTrack FPS: 44.48 offline mean.
33. Motion-only FPS: 45.12 offline mean; production realtime 12.483 FPS on every 60-second probe.
34. Sparse appearance FPS: 44.52 offline mean; clip-A realtime 12.483 FPS.
35. Raw ID switches/fragmentation transitions: 105 annotation-matched transitions.
36. Logical ID switches: 97, an 8-transition reduction.
37. Raw fragmentation count: 105 under the documented per-transition evaluator; 34 total raw IDs for 20 visible people under the coarser manifest count.
38. After logical tracking: 97 annotation transitions; 30 logical IDs.
39. Manager recovery attempts accepted for assignment: 4. Ambiguous motion cases: 0 in these clips.
40. Manager recovery successes: 4; all were region-consistent. Eight repeated annotation transitions preserved the actor because historical raw-ID mapping is retained until expiry.
41. Explicit assignment success rate: 4/4 (100%); broader fragmentation-transition recovery: 8/105 (7.62%).
42. Wrong recovery: 0/4 (0%) and no actor crossed annotated region ownership in 13,211 matched samples. Static regions are weaker than dense MOT GT.
43. Appearance requests/crops: 26 / 180 source seconds = 0.144/s; 16 batches = 0.089/s.
44. Appearance latency: weighted mean about 0.62 ms/batch; worst per-clip P95 2.98 ms.
45. GPU utilization: raw/motion/appearance offline means 61.53/63.43/63.39%.
46. VRAM peak: 547.94 MiB in all three modes; no OOM.
47. Motion realtime lag across three clips: mean 21.15 ms; worst P95 33 ms; maximum 174 ms. Queue peak 1, clean stop on all clips, no runtime error. Sparse clip-A lag mean/P95/max 26.33/34/170 ms.
48. Dynamic pairs: 10,071 frame-pair observations over 2,253 frames, mean 4.47 active pairs/frame, peak 6; max degree 2 was enforced.

Mean actor lifetime increased from 34.52 s (raw IDs) to 39.56 s (logical actors). Host-wide CPU means were 13.03/13.15/13.81%; host RAM samples were noisy because they include unrelated processes (means 8.12/9.19/9.09 GiB), so they are not attributed solely to the pipeline. Dropped-frame counters include deliberate decoder discard while converting 25 FPS source to 12.5 FPS analysis; latest-frame queue remained capacity one and lag stayed bounded.

IDF1 and HOTA are not reported because the dataset lacks dense MOT trajectories.

## 49–54. Limits and decision

49. Known failures: large bbox-scale changes remain separate; recovery beyond 5.5 s remains separate; crossings with near-equal geometry deliberately create a new actor unless optional appearance is decisive; static region annotations cannot prove identity through arbitrary crossings; initial simultaneous duplicate raw tracks are not merged; no complete stand/move/sit or supervisor-crossing source is available.
50. Sparse ReID value: not demonstrated. The CPU histogram added calls/latency but no recovery, switch, or actor-count improvement, so deployment keeps it disabled. A lightweight learned model such as OSNet x0.25 requires a separately licensed checkpoint and measured A/B before adoption.
51. Seat requirement: removed from default monitoring, tracking, Actor ID, Single proposals, and Pair proposals. Zero-Seat API integration test passes.
52. TSM: checkpoint, architecture, 8 frames, 224×224 input, temporal span, normalization, and model inference were not changed. Only proposal identity metadata/routing was extended.
53. Phase 7: Event thresholds, EMA, FSM, calibration, and Pair-normal data were not modified. Event persistence remains disabled and Seat-mode guarded.
54. Decision: **SEAT-FREE STABLE TRACKING: NOT PASSED** for the strict full definition of done due to missing dense/challenging GT coverage and no demonstrated appearance-recovery value. The production default is nevertheless seat-free motion-only, passes the zero-Seat functional tests, sustains ≥10 FPS with bounded lag, and is safer than forcing uncertain identity.
