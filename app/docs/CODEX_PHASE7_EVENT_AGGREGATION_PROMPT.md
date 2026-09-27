# CODEX PROMPT — EXAMGUARD PHASE 7 CALIBRATION + EVENT FSM

You are continuing the existing ExamGuard repository.

Existing pipeline:

```text
Video
→ YOLO11n
→ ByteTrack
→ Seat-Stable Identity
→ Proposal Builder
→ Temporal Buffer
→ TSM R3
→ ActionPrediction
```

Phase 6.5 PERFORMANCE PASS is complete on GTX1650.

Implement Phase 7 only:

```text
ActionPrediction
→ runtime calibration
→ temporal smoothing
→ hysteresis
→ Event FSM
→ cross-proposal deduplication
→ AI Event + EventActor
```

Do NOT implement Evidence, Review UI, Appeal, Head Pose, Desk/Object or Reports.

# 1. Read repository first

Read:

```text
AGENTS.md
CURRENT_STATE.md
PROJECT_CONTEXT.md
SYSTEM_ARCHITECTURE.md
DATABASE_SCHEMA.md
API_CONTRACT.md
CONFIG_SPEC.md
UI_UX_SPEC.md

TSM_RUNTIME_CONTRACT.md
PHASE6_ACTION_RECOGNITION_REPORT.md
PHASE6_5_ACTION_PERFORMANCE_REPORT.md

PHASE7_EVENT_AGGREGATION_CONTEXT.md
PHASE7_EVENT_FSM_SPEC.md
```

Inspect:

```text
backend/app/ai/action_recognition/
backend/app/services/
backend/app/models/
backend/app/schemas/
backend/app/api/
backend/tests/
configs/
scripts/
frontend/src/
```

Before editing:

```bash
git status
git log --oneline -15
git diff --stat
git diff
```

Preserve valid uncommitted work.

# 2. Do not modify working Phase 6 contract

Do not change without a concrete bug:

```text
TSM checkpoint
8-frame contract
224x224
RGB
normalization
class order
timestamp sampling
Single/Pair proposal identity
GTX1650 FP32 scheduler profile
```

# 3. Scope

Implement:

```text
7A Runtime Calibration
7B Smoothing + Event FSM
7C Dedup + Event Persistence
```

Do not invent thresholds before calibration.

# 4. Behavior routing baseline

Use:

```text
SINGLE
→ suspicious_looking
→ using_phone/cheat_sheet

PAIR
→ communicating
→ exchange_object
```

`normal` never creates Event.

Raw TSM remains 5-class.

# 5. Build calibration pipeline

Build/reuse a CLI/script that runs the actual production runtime path over labeled development/validation videos:

```text
YOLO
→ ByteTrack
→ Seat Identity
→ Proposal Builder
→ Dynamic Runtime ROI
→ TSM R3
→ ActionPrediction
```

Do not calibrate only with old annotation ROI evaluation.

Purpose: measure runtime distribution shift.

# 6. Development/final-test separation

Inspect manifests/splits.

Do not use final-test labels in threshold or FSM parameter search.

Record exactly which sessions/videos/events are calibration data.

# 7. Export ActionPrediction calibration artifacts

Export fields such as:

```text
session_id
video_id
timestamp_ms
proposal_id
proposal_type
session_candidate_ids
seat_codes

prob_normal
prob_suspicious_looking
prob_communicating
prob_exchange_object
prob_using_phone_cheat_sheet

top_class
top_confidence

matched_gt_event_id
matched_gt_behavior
```

Use JSONL/CSV/Parquet according to project conventions.

Do not persist raw predictions to PostgreSQL.

# 8. Ground-truth alignment

Define exact prediction-to-GT alignment using:

```text
session/video
timestamp
behavior
actor identity where available
```

Pair GT should require compatible actor pair when annotations support it.

Document unmatched predictions.

# 9. Raw metrics

Produce:

```text
per-class Precision
Recall
F1
Macro F1
Weighted F1
confusion matrix
```

Also inspect positive/negative probability distributions per behavior where practical.

# 10. Implement streaming smoother

Preferred baseline:

```text
EMA
```

Use only current/past predictions.

Make alpha configurable.

Reset on:

```text
seek
proposal expiry
large discontinuity
runtime reset
```

# 11. Add event detection config

Add centralized config:

```yaml
event_detection:
  enabled: false

  suspicious_looking:
    proposal_types: [SINGLE]
    smoothing:
      type: ema
      alpha: ...
    start_threshold: ...
    keep_threshold: ...
    min_active_ms: ...
    end_grace_ms: ...
    merge_gap_ms: ...

  communicating:
    proposal_types: [PAIR]
    ...

  exchange_object:
    proposal_types: [PAIR]
    ...

  using_phone_cheat_sheet:
    proposal_types: [SINGLE]
    ...
```

Final values must come from calibration.

# 12. Parameter search

Implement a bounded reproducible search over development/validation.

Tune per behavior:

```text
EMA alpha
start_threshold
keep_threshold
min_active_ms
end_grace_ms
merge_gap_ms optional
```

Primary objective:

```text
event-level F1 / Macro Event F1
```

Monitor:

```text
false events/min
Recall
duplicate rate
```

Avoid huge search spaces.

# 13. Hysteresis

Normally require:

```text
start_threshold > keep_threshold
```

Do not create Event from one score crossing a single threshold.

# 14. Event FSM

One FSM per:

```text
proposal_id + behavior
```

States:

```text
IDLE
CANDIDATE
ACTIVE
COOLDOWN
```

Use runtime domain types, not ORM as state.

# 15. IDLE → CANDIDATE

When smoothed score reaches start evidence.

Track:

```text
candidate_start_ms
peak probability
peak timestamp
evidence
```

No DB write.

# 16. CANDIDATE → ACTIVE

Require sustained evidence for:

```text
min_active_ms
```

Use source/action timestamps, never wall clock.

Effective Event start should reflect sustained evidence beginning.

# 17. CANDIDATE → IDLE

If evidence collapses before minimum duration:

discard candidate.

No Event row.

Increment suppression diagnostic.

# 18. ACTIVE → COOLDOWN

When score falls below keep evidence.

Record last valid active timestamp.

Do not close immediately.

# 19. COOLDOWN → ACTIVE

If evidence recovers before:

```text
end_grace_ms
```

resume same Event.

No second Event.

# 20. COOLDOWN → CLOSED

If low evidence persists through grace:

finalize candidate Event.

Use timestamp boundary rules from context.

# 21. Event confidence

Track:

```text
mean active probability
peak probability
peak timestamp
prediction count
```

Use:

```text
ai_confidence = mean active probability
```

unless project schema already specifies another documented rule.

# 22. Normal regression rule

Explicitly test:

```text
normal never creates Event
```

# 23. Proposal/behavior routing regression

Baseline:

```text
SINGLE communicating → no Event
SINGLE exchange_object → no Event

PAIR suspicious_looking → no Event
PAIR using_phone_cheat_sheet → no Event
```

Raw predictions remain available for calibration/debug.

# 24. Pair interaction persistence

For:

```text
pair:SC103:SC104
```

communicating/exchange creates:

```text
one Event
two EventActors
```

Do not create duplicate one-person Events.

# 25. Cross-proposal dedup

Implement deterministic dedup after candidate Event finalization.

Use:

```text
behavior
actor set
temporal overlap / short gap
```

Prefer exact actor-set equality baseline.

Do not merge unrelated nearby candidates.

# 26. Event persistence

Inspect existing `events` and `event_actors`.

Create AI Event using current schema:

```text
source = AI
status = PENDING_REVIEW
behavior
start_ms
end_ms
peak_ms
ai_confidence
session_id
```

Create EventActors from SessionCandidate IDs.

Never set CONFIRMED automatically.

# 27. No raw-prediction DB table

Do not create:

```text
action_predictions
model_predictions
frame_predictions
```

tables.

# 28. Idempotency

Protect Event persistence from duplicate callbacks/retries/runtime duplication.

Use existing service patterns.

Do not add arbitrary DB uniqueness constraints without analysis.

# 29. Pause

Pause must not advance CANDIDATE/COOLDOWN by wall clock.

# 30. Seek

On arbitrary seek reset:

```text
smoothers
FSMs
candidate event state
cooldowns
pending dedup state
```

Do not persist partial pre-seek candidate events.

Finalized historical Events remain.

# 31. Stop

Recommended policy:

```text
ACTIVE
→ finalize at last valid evidence timestamp

CANDIDATE below minimum
→ discard
```

Document exact behavior.

# 32. Event-level evaluation

Implement predicted-event matching against GT.

Require:

```text
same behavior
compatible actor set
sufficient temporal overlap
```

Define and document temporal criterion.

Report:

```text
TP
FP
FN
Precision
Recall
F1
per-class Event F1
Macro Event F1
false events/minute
duplicate event rate
```

Do not silently tune evaluation tolerance to improve metrics.

# 33. Selected config artifact

Create:

```text
selected_event_thresholds.yaml
```

or equivalent.

For each behavior report:

```text
alpha
start threshold
keep threshold
min active ms
end grace ms
merge gap if used

validation Precision
Recall
F1
false events/min
```

# 34. Final-test discipline

After selected config is frozen:

optionally run final test once if data/environment permit.

Clearly separate:

```text
calibration/development
vs
final test
```

Do not retune after final-test result.

# 35. RBAC

AI Event creation is internal system behavior.

Do not pretend AI is Supervisor/Reviewer.

Existing user permissions remain unchanged.

# 36. Diagnostics

Add:

```text
candidate_fsms
active_fsms
cooldown_fsms

events_created_total
events_suppressed_total
events_deduplicated_total

per_behavior_event_count
```

No frame-level DB logging.

# 37. Unit tests

Add tests for:

```text
normal never creates Event

single suspicious sustained → one Event
single phone sustained → one Event

pair communicating sustained → one Event + two actors
pair exchange sustained → one Event + two actors

one spike → no Event
brief drop → same Event
long drop → closes Event
cooldown recovery → same Event

invalid proposal/behavior routing → no Event

seek reset
pause timestamp behavior
stop ACTIVE flush
insufficient CANDIDATE discard

true duplicate → dedup
different actors → no dedup
```

# 38. Calibration tests

Verify:

```text
final test excluded from search
search reproducible
GT actor/time alignment correct
invalid annotations reported
```

# 39. Static checks

Run repository-standard:

```text
pytest
ruff
mypy
```

Only run frontend tests/build if frontend changed.

# 40. Reports

Create:

```text
docs/PHASE7_EVENT_AGGREGATION_REPORT.md
```

Update:

```text
CURRENT_STATE.md
SYSTEM_ARCHITECTURE.md
CONFIG_SPEC.md
```

Report calibration data, search space, selected parameters, raw metrics, Event metrics and known failure modes.

# 41. Do not implement

Do NOT add:

```text
Head Pose
Desk/Object detector
new TSM architecture
retraining
Evidence generation
Event Review UI
Appeal
Reports engine
notifications
```

Stop at AI Event creation.

# 42. Definition of Done

Set:

```text
PHASE 7: PASS
```

only if:

```text
runtime calibration completed on development/validation
per-class config selected reproducibly
normal never creates Event
proposal/behavior routing enforced
FSM timestamp-based
single spikes suppressed
jitter does not fragment Events
pair interaction creates one Event with two EventActors
dedup works
Event persisted source=AI
Event persisted PENDING_REVIEW
raw ActionPrediction not persisted PostgreSQL
seek/pause/stop correct
event-level metrics produced
no final-test leakage
tests/static checks pass
```

Otherwise:

```text
PHASE 7: NOT PASSED
```

with blockers.

# 43. Final report

Return:

1. Calibration dataset used.
2. Proof final test was excluded from tuning.
3. Raw ActionPrediction count.
4. GT alignment rule.
5. Raw per-class Precision/Recall/F1.
6. Raw Macro F1.
7. Confusion matrix artifact.
8. Probability distribution observations.
9. Smoothing method.
10. Search space.
11. Selected parameters per behavior.
12. FSM implementation.
13. Event start boundary rule.
14. Event end boundary rule.
15. Event confidence rule.
16. Proposal/behavior routing.
17. Dedup rule.
18. Event matching evaluation criterion.
19. Event Precision/Recall/F1.
20. Per-class Event F1.
21. Macro Event F1.
22. False events/minute.
23. Duplicate event rate.
24. Event counts per class.
25. DB/services modified.
26. EventActor behavior.
27. Pause behavior.
28. Seek behavior.
29. Stop/flush behavior.
30. Idempotency behavior.
31. Tests added.
32. Exact pytest result.
33. Ruff result.
34. mypy result.
35. Frontend result if changed.
36. False-positive modes.
37. False-negative modes.
38. Boundary errors.
39. Confirmation no Head Pose.
40. Confirmation no Desk/Object.
41. Confirmation no Evidence/Appeal/Report.
42. Final PHASE 7 PASS / NOT PASSED.

STOP after Phase 7.

Do not automatically begin Phase 8.
