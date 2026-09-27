# ExamGuard — Phase 7 Event FSM Specification

## 1. Scope

Implement:

```text
ActionPrediction
→ Calibration tooling
→ Smoothing
→ Hysteresis
→ Event FSM
→ Cross-proposal dedup
→ Event/EventActor persistence
```

Không implement Evidence/Review UI/Appeal/Head Pose.

## 2. Behavior routing

```text
SINGLE:
  suspicious_looking
  using_phone/cheat_sheet

PAIR:
  communicating
  exchange_object

normal:
  no Event
```

## 3. Calibration artifacts

Recommended:

```text
artifacts/event_calibration/
  predictions.jsonl
  prediction_metrics.json
  event_metrics.json
  selected_thresholds.yaml
```

Không yêu cầu PostgreSQL cho offline calibration nếu không cần.

## 4. Ground-truth alignment

Prediction align với GT theo:

```text
session/video
timestamp
behavior
actor identity where available
```

Pair behavior phải match actor pair nếu annotation hỗ trợ.

## 5. Parameter search

Tune trên development/validation:

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

Monitor false-events/minute và Recall.

Không dùng final test để search.

## 6. Config shape

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

## 7. Streaming smoother

Baseline EMA:

```text
s_t = alpha * p_t + (1-alpha) * s_prev
```

Reset trên seek, proposal expiry hoặc discontinuity lớn.

## 8. FSM states

```text
IDLE
CANDIDATE
ACTIVE
COOLDOWN
```

### IDLE → CANDIDATE
Smoothed score đủ `start_threshold`.

### CANDIDATE → ACTIVE
Evidence duy trì đủ `min_active_ms`.

### CANDIDATE → IDLE
Evidence mất trước minimum duration.

### ACTIVE → COOLDOWN
Score xuống dưới `keep_threshold`.

### COOLDOWN → ACTIVE
Score phục hồi trước `end_grace_ms`.

### COOLDOWN → CLOSE
Low evidence kéo dài qua grace.

## 9. Runtime event statistics

Track:

```text
start_ms
last_active_ms
end_ms
peak_ms
peak_probability
mean_active_probability
number_of_predictions
```

Persist schema-supported fields.

Default:

```text
ai_confidence = mean_active_probability
```

## 10. Dedup

Candidate Event dedup theo:

```text
behavior
actor set
time overlap / short gap
```

Config có thể gồm:

```text
dedup_temporal_iou
dedup_max_gap_ms
```

Baseline ưu tiên exact actor-set.

## 11. Persistence

Create:

```text
Event(
  source=AI,
  status=PENDING_REVIEW,
  behavior=...,
  start_ms=...,
  end_ms=...,
  peak_ms=...,
  ai_confidence=...
)
```

Create EventActor từ SessionCandidate IDs.

Không tạo EventReview.

Không tạo Event normal.

## 12. Idempotency

Chống persistence duplicate do retry/runtime duplicate.

Ưu tiên runtime fingerprint/idempotency mechanism hiện có.

Không thêm DB unique constraint mù quáng.

## 13. Lifecycle

Pause:
- không wall-clock transition.

Seek:
- reset non-persisted aggregation state.

Stop:
- ACTIVE close tại last valid evidence;
- insufficient CANDIDATE discard.

## 14. Diagnostics

```text
candidate_fsms
active_fsms
cooldown_fsms
events_created_total
events_suppressed_total
events_deduplicated_total
per_behavior_event_count
```

## 15. Unit tests

Phải có:

```text
normal never creates Event

single suspicious sustained → Event
single phone sustained → Event

pair communicating sustained → one Event + two actors
pair exchange sustained → one Event + two actors

single spike → no Event
brief drop → same Event
long drop → close
cooldown recovery → same Event

invalid proposal/behavior routing → no Event

seek reset
pause timestamp behavior
stop flush policy

true duplicate → dedup
different actors → no dedup
```

## 16. Event-level evaluation

Predicted Event match GT cần:

```text
same behavior
compatible actor set
sufficient temporal overlap
```

Document exact temporal criterion.

Report:

```text
TP
FP
FN
Precision
Recall
F1
per-class F1
Macro Event F1
false events/minute
duplicate event rate
```

## 17. Final-test protocol

Sau khi chọn config từ development/validation:

```text
freeze config
```

Sau đó mới chạy final test.

Không retune sau khi xem final-test result.

## 18. Definition of Done

PASS nếu:

```text
calibration artifacts exist
selected per-class config exists
FSM implemented/tested
dedup implemented/tested
Event/EventActor persistence works
normal never creates Event
AI events remain PENDING_REVIEW
event-level metrics produced
no final-test leakage
```
