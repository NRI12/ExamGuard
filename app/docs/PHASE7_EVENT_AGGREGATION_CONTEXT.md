# ExamGuard — Phase 7 Event Aggregation Context

## 1. Trạng thái hệ thống trước Phase 7

Các phase trước đã hoàn thành:

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

Phase 6.5 đã PASS về hiệu năng trên GTX 1650 Max-Q.

Runtime profile đã chọn:

```text
precision: FP32
prediction_stride_ms: 1500
max_batch_size: 2
queue_capacity: 1
max_prediction_age_ms: 2000
minimum_inference_interval_ms: 200
```

Realtime-paced validation:

```text
tracking FPS ≈ 12.5
action queue bounded
analysis lag P95 ≈ 100 ms
ActionPrediction age P95 <= 320 ms
no OOM
```

Training/runtime tensor preprocessing đã được xác nhận tương đương:

```text
8 frames
RGB uint8
/255
224x224
bilinear
align_corners=False
ImageNet mean/std
[B,8,3,224,224]
```

Lưu ý còn tồn tại:

```text
training ROI = annotation-derived static ROI
runtime ROI  = tracking-derived dynamic ROI
```

Do đó probability distribution ở runtime không được giả định giống training/evaluation distribution.

---

## 2. Mục tiêu Phase 7

Phase 7 chuyển:

```text
raw ActionPrediction
```

thành:

```text
reviewable AI Event
```

Pipeline:

```text
ActionPrediction
      │
      ▼
Runtime Calibration
      │
      ▼
Class-specific Temporal Smoothing
      │
      ▼
Threshold + Hysteresis
      │
      ▼
Event FSM
      │
      ▼
Cross-Proposal Deduplication
      │
      ▼
Event
status = PENDING_REVIEW
```

AI chỉ tạo sự kiện cần con người xem lại.

AI không tự xác nhận vi phạm.

---

## 3. Phase 7 chia thành ba phần

### Phase 7A — Runtime Calibration

Đánh giá đúng production runtime pipeline:

```text
Video
→ YOLO
→ ByteTrack
→ Seat Identity
→ Single/Pair Proposal
→ Dynamic ROI
→ TSM R3
→ probabilities
```

trên dữ liệu có ground truth.

Mục tiêu:

```text
quan sát probability distribution
chọn class-specific threshold
chọn smoothing/hysteresis parameters
đánh giá event-level metrics
```

Không lấy threshold trực tiếp từ training softmax nếu chưa kiểm chứng runtime.

### Phase 7B — Temporal Aggregation / Event FSM

Raw prediction không được tạo Event trực tiếp.

Mỗi:

```text
proposal_id + behavior
```

có một temporal state machine.

### Phase 7C — Cross-Proposal Deduplication

Hợp nhất các candidate event trùng semantic/time/actors.

---

## 4. Các behavior

Project có 5 class:

```text
normal
suspicious_looking
communicating
exchange_object
using_phone/cheat_sheet
```

`normal` không tạo Event.

Runtime Event chỉ áp dụng cho 4 abnormal behavior:

```text
suspicious_looking
communicating
exchange_object
using_phone/cheat_sheet
```

---

## 5. Proposal-type semantics

### SINGLE

Stable ID:

```text
single:<session_candidate_id>
```

Baseline Event eligibility:

```text
suspicious_looking
using_phone/cheat_sheet
```

### PAIR

Stable ID:

```text
pair:<session_candidate_id_A>:<session_candidate_id_B>
```

Baseline Event eligibility:

```text
communicating
exchange_object
```

Raw TSM vẫn output đủ 5 class. Routing chỉ quyết định class nào được Event layer tiêu thụ.

`normal` không bao giờ tạo Event.

---

## 6. Vì sao không dùng một threshold chung

Probability distribution của từng behavior khác nhau.

Do đó không dùng:

```text
if p > 0.7:
    create_event()
```

cho mọi class.

Dùng class-specific config:

```yaml
event_detection:
  suspicious_looking:
    start_threshold: ...
    keep_threshold: ...
    min_active_ms: ...
    end_grace_ms: ...

  communicating:
    ...

  exchange_object:
    ...

  using_phone_cheat_sheet:
    ...
```

Các giá trị phải đến từ calibration.

---

## 7. Calibration dataset

Ưu tiên dữ liệu có ground truth event:

```text
session
source video
actor(s)
behavior
event start_ms
event end_ms
```

Không calibrate threshold trên final test rồi dùng lại final test để báo kết quả.

Nếu project có:

```text
development
final test
```

thì calibration/tuning chỉ dùng development/validation.

Final test chỉ chạy sau khi khóa tham số.

---

## 8. Calibration output

Mỗi ActionPrediction nên export cho research/debug:

```text
session_id
video_id
timestamp_ms
proposal_id
proposal_type
session_candidate_ids
seat_codes
ground_truth_event_id
ground_truth_behavior
probabilities
top_class
top_confidence
```

Có thể lưu CSV/JSONL/Parquet.

Không cần persist mọi prediction vào PostgreSQL.

---

## 9. Metrics

Raw prediction-level:

```text
per-class Precision
per-class Recall
per-class F1
Macro F1
Weighted F1
confusion matrix
probability distributions
```

Event-level sau aggregation:

```text
event Precision
event Recall
event F1
per-class Event F1
Macro Event F1
false events / minute
duplicate event rate
missed event rate
```

---

## 10. Temporal smoothing

Smoothing phải dùng timestamps.

Baseline đơn giản:

```text
EMA
```

hoặc timestamp-aware rolling mean.

Không dùng future predictions trong live runtime.

Reset khi seek / proposal expiry / discontinuity lớn.

---

## 11. Hysteresis

Dùng hai threshold:

```text
start_threshold
keep_threshold
```

thông thường:

```text
start_threshold > keep_threshold
```

Điều này tránh Event bật/tắt liên tục khi score dao động quanh ngưỡng.

---

## 12. Event FSM

Baseline states:

```text
IDLE
CANDIDATE
ACTIVE
COOLDOWN
```

Luồng:

```text
IDLE
  │ sufficient start evidence
  ▼
CANDIDATE
  │ sustained for min_active_ms
  ▼
ACTIVE
  │ evidence drops
  ▼
COOLDOWN
  │ evidence recovers
  ├────────────→ ACTIVE
  │
  │ low evidence persists for end_grace_ms
  ▼
CLOSE EVENT
```

FSM key:

```text
proposal_id + behavior
```

Không key bằng Track ID.

---

## 13. Event boundary

Start:

```text
first sustained evidence
```

End:

```text
last sustained evidence + short grace
```

Không mở rộng event tùy tiện.

Không lấy toàn bộ 4-second TSM window làm event duration một cách máy móc.

---

## 14. Minimum duration

Không tạo Event từ một prediction đơn lẻ.

Mỗi class có:

```text
min_active_ms
```

được calibration.

---

## 15. Event confidence

Baseline:

```text
ai_confidence = mean probability over ACTIVE evidence
```

Có thể theo dõi thêm:

```text
peak_probability
peak_timestamp
```

trong runtime/debug artifact.

---

## 16. Cross-proposal deduplication

Sau FSM, deduplicate dựa trên:

```text
same behavior
actor set
temporal overlap / short gap
```

Baseline ưu tiên exact actor-set equality.

Không merge Event chỉ vì cùng behavior hoặc ngồi gần nhau.

---

## 17. Two-person interaction

Một interaction:

```text
SC103 ↔ SC104
```

phải tạo:

```text
one Event
two EventActors
```

không tạo hai Event riêng.

---

## 18. Event persistence

Event AI:

```text
source = AI
status = PENDING_REVIEW
```

Dùng schema hiện tại:

```text
event_code
session_id
source
behavior
start_ms
end_ms
peak_ms
ai_confidence
status
```

Không tạo Event `normal`.

Không tự CONFIRM.

---

## 19. Persistence boundary

Không persist raw ActionPrediction vào PostgreSQL.

Persist:

```text
final aggregated Event
EventActor
```

Raw prediction dùng file artifact cho research/debug.

---

## 20. Runtime lifecycle

### Pause
Không advance FSM theo wall-clock.

### Seek
Reset smoothing, FSM, candidate state, cooldown, dedup pending state.

### Stop
Recommended:
- ACTIVE → close tại last valid evidence timestamp.
- CANDIDATE chưa đủ min evidence → discard.

Historical finalized Event không tự delete khi seek.

---

## 21. Scientific requirement

Phân biệt:

```text
raw classification evaluation
```

và:

```text
event-level system evaluation
```

Không suy ra Event system tốt chỉ từ clip-level Macro F1.

---

## 22. Không bao gồm trong Phase 7

Không thêm:

```text
Head Pose
Desk/Object detector
multimodal Fusion
Evidence generation
Appeal
Report engine
new TSM model/retraining
```

---

## 23. Definition of Done

Phase 7 PASS khi:

```text
runtime calibration thực hiện trên development/validation

class-specific parameters có config reproducible

normal không tạo Event

Single/Pair routing rõ ràng

không tạo Event từ một prediction đơn lẻ

FSM dùng timestamp

jitter không tạo nhiều duplicate Event

Pair interaction tạo một Event với hai EventActors

seek/pause/stop lifecycle đúng

Event source=AI
Event status=PENDING_REVIEW

raw predictions không persist PostgreSQL

event-level metrics được báo cáo

không tuning trên final test

tests/static checks pass
```
