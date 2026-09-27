# ExamGuard Phase 7 — Calibration and Event Aggregation

Updated: 2026-09-23

```text
PHASE 7 IMPLEMENTATION/UNIT/INTEGRATION GATE: PASSED
PHASE 7 RUNTIME CALIBRATION: COMPLETED
PHASE 7 EVENT-ACCURACY GATE: FAILED
PHASE 7: NOT PASSED
```

Phase 7 stops at internal AI Event creation. The production-path pilot run is
complete and reproducible, but the measured event detector is not deployable:
refined Event F1 is 0.1216, Macro Event F1 is 0.1064, false events/minute is 1.01,
and `communicating` Event F1 is 0. Production profiles therefore remain
`event_detection.enabled: false`.

## 1–8. Calibration data, leakage control and raw metrics

1. **Calibration dataset used:** `exam_dataset_pilot-checkpoint5-runtime-events`.
   S00–S04 are the development partition: five full-room MP4s, 52.2407 minutes,
   106 annotated intervals (86 abnormal and 20 normal hard negatives). GT counts
   are suspicious-looking 36, communicating 16, exchange-object 16,
   phone/cheat-sheet 18 and normal 20. The exact event IDs and video names are in
   `event_calibration/artifacts/dataset_summary.json`.
2. **Final-test exclusion:** `checkpoint5_master_split.partition` puts S05–S08 in
   `final_test`. Collection filters to `development|validation`; search calls
   `assert_calibration_only` and rejects `final|final_test|test`; the prediction
   artifact contains only S00–S04. S05–S08 were not inferred or evaluated, and
   no final test was run. The checkpoint itself was trained on the 509 S00–S04
   development windows, so this threshold search is not an independent model
   validation and may be optimistic despite the poor result.
3. **Raw ActionPrediction count:** 20,210: 9,839 SINGLE and 10,371 PAIR, over 52
   stable proposal identities. Predictions are file-only; none are persisted to
   PostgreSQL.
4. **GT alignment:** exact Session UUID plus video ID; inclusive prediction
   timestamp inside `[start_ms,end_ms]`; exact canonical actor set when actors are
   annotated. Pair GT requires the exact two actors. The shortest interval then
   lexical event ID resolves multiple matches deterministically. Unmatched
   predictions are raw target `normal`. Result: 436 matched and 19,774 unmatched
   predictions; zero invalid annotations.
5. **Raw per-class Precision/Recall/F1:** normal 0.9831/0.5771/0.7273;
   suspicious-looking 0.1583/0.3607/0.2200; communicating
   0.0040/1.0000/0.0080; exchange-object 0.0442/0.4545/0.0806;
   phone/cheat-sheet 0.0952/0.2051/0.1301.
6. **Raw Macro/Weighted F1:** 0.2332 / 0.7171. Weighted F1 is dominated by the
   19,858 normal-target predictions and must not be used as the event-quality
   headline.
7. **Confusion matrix artifact:** `prediction_metrics.json`, row/column order
   `[normal, suspicious, communicating, exchange, phone]`:
   `[[11460,232,7431,431,304],[77,44,1,0,0],[0,0,30,0,0],
   [0,0,24,20,0],[120,2,1,1,32]]`.
8. **Probability distributions:** positive/negative P50 values are suspicious
   0.1882/0.0394, communicating 0.8202/0.1197, exchange 0.4040/0.0561 and phone
   0.1486/0.0262. The communicating negative P95 is 0.9529, explaining the very
   high false-event pressure. Full min/P50/P95/max values are in the artifact.
   The development training manifest exposes the cause: all 218 normal samples
   have one actor, while all 27 communicating and 33 exchange samples have two.
   There are no pair-normal negatives, so actor count/crop geometry is a class
   shortcut. Same-row runtime Pair proposals are top-1 communicating 89.0% of
   the time; their communicating probability P50/P95 is 0.8611/0.9713.

The derived manifest is `event_calibration/exam_dataset_pilot.yaml`. It rebases
stale `UBUNTU_DATA3` paths without changing the external dataset. The explicit
2D Seat neighbor graph comes from layout metadata, not event labels; 30/32
development Pair events are graph-reachable. The two intentionally retained
cross-diagonal GT events are valid false-negative opportunities.

## 9–17. Smoothing, search, FSM and deduplication

9. **Smoothing:** causal EMA only,
   `s_t = alpha*p_t + (1-alpha)*s_(t-1)`, using current/past predictions. It
   resets on seek/runtime reset, proposal expiry and large source discontinuity.
10. **Search space:** a one-time refined bounded search used 1,296 deterministic
    combinations per behavior: alpha `{0.4,0.6}`, start
    `{0.5,0.65,0.8,0.9,0.95,0.98}`, keep
    `{0.35,0.5,0.65,0.8,0.9,0.95}`, minimum active `{1500,3000,4500}` ms,
    end grace `{0,1500,3000}` ms and merge gap `{0,1500}` ms. Objective order is
    Event F1, Recall, then lower false-events/minute. The grid was frozen after
    this diagnostic refinement to avoid iterative overfitting on development.
11. **Selected search values:** suspicious `(alpha=.6,start=.65,keep=.5,
    min=3000,grace=0,merge=0)`; communicating `(.4,.98,.35,1500,0,0)`;
    exchange `(.6,.5,.35,4500,0,0)`; phone `(.4,.5,.35,4500,1500,0)`.
    These are frozen evaluation results, not
    deployment approval. The artifact retains `enabled: false`.
12. **FSM:** one in-memory domain FSM per `proposal_id + behavior`, with
    IDLE/CANDIDATE/ACTIVE/COOLDOWN. CANDIDATE requires sustained source-time
    evidence and at least two predictions. Evidence collapse suppresses it;
    cooldown recovery resumes the same Event.
13. **Start boundary:** first prediction timestamp of the sustained evidence
    sequence that later satisfies `min_active_ms`, not the four-second clip start.
14. **End boundary:** normal closure is `last_active_ms + end_grace_ms`; stop and
    discontinuity close at the last valid evidence timestamp.
15. **Confidence:** arithmetic mean of causal smoothed probabilities counted as
    active evidence. Peak probability/time and prediction count are also tracked;
    schema-supported mean and peak timestamp are persisted.
16. **Routing:** SINGLE only suspicious/phone; PAIR only
    communicating/exchange; `normal` has no FSM and cannot create Event.
17. **Dedup:** same behavior, exact canonical SessionCandidate actor set and
    qualifying temporal IoU or short gap. Different actors never merge.

## 18–24. Event-level evaluation

18. **Matching criterion:** deterministic one-to-one greedy matching by decreasing
    overlap, requiring same session, same behavior, exact compatible actor set and
    temporal IoU ≥ 0.30. The threshold was fixed before search and not tuned.
19. **Overall Event Precision/Recall/F1:** 0.1452 / 0.1047 / 0.1216
    (TP=9, FP=53, FN=77).
20. **Per-class Event F1:** suspicious 0.1600 (4/10/32 TP/FP/FN),
    communicating 0.0000 (0/6/16), exchange 0.1277 (3/28/13), phone 0.1379
    (2/9/16).
21. **Macro Event F1:** 0.1064.
22. **False events/minute:** 1.0145 over 52.2407 development minutes.
23. **Duplicate event rate:** 0.0 after exact-actor temporal dedup.
24. **Predicted Event counts:** suspicious 14, communicating 6, exchange 31,
    phone 11; total 62. Communicating false events still have duration P50
    47.12 seconds and maximum 288.24 seconds at start threshold 0.98; none match
    GT. For GT-aligned communicating predictions, score minus the same proposal's
    non-GT median has P50 −0.0241 and is positive only 43.3%, so causal
    per-proposal baseline subtraction is not a viable repair.

Artifacts are in `docs/event_calibration/artifacts/`: `predictions.jsonl`,
`dataset_summary.json`, `prediction_metrics.json`, `predicted_events.jsonl`,
`event_metrics.json` and `selected_event_thresholds.yaml`. The training/runtime
bias evidence is frozen in `event_calibration/pair_distribution_diagnosis.json`.

## 25–31. Persistence, lifecycle and tests

25. **DB/services modified:** no migration and no new entity. Existing Event,
    EventActor and AuditLog are used through `persist_ai_event`; runtime wiring is
    in the monitoring manager/worker and action callback.
26. **EventActor behavior:** actor UUIDs must belong to the Event session. SINGLE
    creates one actor; PAIR creates one Event with two actors, never two one-person
    Events.
27. **Pause:** no frame/source timestamp means no EMA/FSM/cooldown transition;
    wall clock is not used.
28. **Seek:** generation change clears every non-persisted smoother, FSM,
    cooldown and recent dedup item; finalized historical Events remain.
29. **Stop/flush:** ACTIVE/COOLDOWN close at last valid evidence; insufficient
    CANDIDATE is discarded and counted as suppressed; pending dedup output flushes.
30. **Idempotency:** `AI-` plus the first 40 hex characters of a SHA-256 semantic
    fingerprint over session, behavior, sorted actors and boundaries. Retry reads
    the existing Event and creates no duplicate actors or audit row.
31. **Tests added:** normal exclusion; sustained Single/Pair behaviors; spike,
    brief/long drop and recovery; invalid routing; seek/pause/stop/expiry;
    dedup/different actors; Pair persistence; AI/PENDING_REVIEW/audit/idempotency;
    final split guard; alignment; search reproducibility; normal interval metric
    exclusion; phone class-name mapping; explicit 2D neighbor graph; manifest
    path/split/neighbor derivation.

## 32–42. Verification, failure analysis and scope

32. **Pytest:** 129 collected; 124 passed and 5 environment-gated tests skipped
    (four unmounted R3 reference/checkpoint/clip tests and one optional CUDA
    integration test). Phase 7 aggregation tests pass 17/17 and the new
    manifest-builder test passes.
33. **Ruff:** `ruff check app tests` passes.
34. **Mypy:** `mypy app` passes for 111 source files.
35. **Frontend:** unchanged and therefore not rerun, as required by the prompt.
36. **False-positive modes:** the checkpoint's training split has no Pair-normal
    negatives, so actor cardinality and wide Pair crop geometry are confounded
    with interaction labels. Ordinary neighboring motion receives sustained high
    communication probability; seat/proposal identity fragmentation and
    background entrants add risk; long high-score spans have poor temporal IoU.
37. **False-negative modes:** missed/late seat identity; two Pair GT actors outside
    the declared neighbor graph; phone positives classified normal; short behavior
    suppressed by four-second clips, cadence, EMA or 3-second minimum duration.
38. **Boundary errors:** 1.5-second proposal prediction cadence quantizes edges;
    EMA delays crossing; grace extends ends; long false-positive spans fail the
    fixed IoU criterion. GT interval subjectivity remains a source of variance.
39. **Head Pose:** not added.
40. **Desk/Object:** not added.
41. **Evidence/Appeal/Report:** no Evidence generation, Review UI, Appeal,
    report engine, notification, retraining or new TSM architecture was added.
42. **Decision:** **PHASE 7: NOT PASSED.** Runtime calibration and all requested
    artifacts now exist with final-test isolation, but empirical event quality is
    unusable—especially zero communicating Event recall and 1.01 false events per
    minute after refined search. Threshold/FSM changes cannot recover a class
    whose real GT score is not separated from its own proposal baseline. Both
    profiles remain disabled. Do not run final test or tune on S05–S08. A future
    model-data phase must add Pair-normal/hard-negative crops under the same
    dynamic ROI contract before retraining and independent validation.

Stop after Phase 7. Do not begin Phase 8 automatically.
