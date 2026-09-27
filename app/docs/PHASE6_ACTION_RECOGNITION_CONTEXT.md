# ExamGuard — Phase 6 Action Recognition Runtime Context

## 1. Trạng thái đầu vào

Các phase trước đã hoàn thành:

```text
Video
→ YOLO11n
→ ByteTrack
→ Seat-Stable Identity
```

Seat-Stable Identity hiện cung cấp:

```text
Track
→ Seat
→ SessionCandidate
→ Candidate
```

Kết quả validation gần nhất:

```text
Correct Seat Assignment: 95.146%
Wrong Seat Rate: 0%
Unassigned Rate: 4.854%
False Seat Switches: 0

seat_assignment_ms:
mean 0.667 ms
P95 0.770 ms
```

Seat-Stable Identity ưu tiên:

```text
correct identity
>
fast identity
```

và cho phép:

```text
UNASSIGNED
```

khi evidence không đủ mạnh.

Không thay đổi nguyên tắc này trong Phase 6.

---

# 2. Mục tiêu Phase 6

Phase 6 bổ sung khả năng:

```text
Candidate-aware temporal proposal
→ ROI clip
→ TSM R3
→ 5-class probabilities
```

Pipeline mục tiêu:

```text
Video
  │
  ▼
YOLO11n
  │
  ▼
ByteTrack
  │
  ▼
Seat-Stable Identity
  │
  ▼
Proposal Builder
  │
  ├───────────────┐
  ▼               ▼
Single Actor   Adjacent Pair
Proposal       Proposal
  │               │
  ▼               ▼
Actor Context     Pair ROI
ROI
  │               │
  └──────┬────────┘
         ▼
 Temporal Clip Buffer
         │
         ▼
    Temporal Sampling
         │
         ▼
       TSM R3
         │
         ▼
  5-class probabilities
```

Phase 6 chỉ kết thúc ở raw action probabilities.

Không tạo Event trong phase này.

---

# 3. Không triển khai trong Phase 6

Không thêm:

```text
Head Pose fusion
Desk/Object detector
High-resolution object cue
new Temporal Difference branch
new multimodal fusion
Event FSM
Evidence
Appeal
Report
automatic violation conclusion
```

Head Pose và Desk/Object sẽ là ablation sau khi TSM runtime baseline hoạt động đúng.

Nếu checkpoint R3 đã tích hợp Temporal Difference / Temporal Attention, không thêm lại branch temporal difference ở system level.

---

# 4. Các class

Giữ đúng 5 class của project:

```text
normal
suspicious_looking
communicating
exchange_object
using_phone/cheat_sheet
```

Runtime class order phải lấy chính xác từ training/checkpoint/config.

Không tự suy đoán index.

---

# 5. Phase 6.0 — Runtime Contract Audit

Trước khi load checkpoint vào production runtime, phải xác minh chính xác training contract.

Cần tìm từ:

```text
training notebook
training scripts
model config
checkpoint metadata
dataset transform
ROI generation
evaluation scripts
```

Xác định:

```text
model architecture
backbone
TSM configuration
Temporal Difference có/không
Temporal Attention có/không

checkpoint state_dict format

number of frames
input resolution

clip temporal span
sampling strategy
frame stride

RGB/BGR
tensor layout
dtype

resize policy
crop policy

normalization mean
normalization std

class order

logits / softmax behavior

ROI convention
```

Không được đoán.

Output của audit phải trở thành một runtime contract bất biến.

---

# 6. TSM Runtime Contract

Tạo tài liệu/config tương đương:

```text
TSM_RUNTIME_CONTRACT.md
```

và config runtime.

Ví dụ cấu trúc:

```yaml
action_recognition:
  enabled: true

  model:
    name: tsm_r3
    checkpoint: ...
    num_classes: 5

  input:
    num_frames: ...
    height: ...
    width: ...
    color_space: RGB

  sampling:
    strategy: ...
    clip_span_ms: ...
    stride_ms: ...

  normalization:
    mean: [...]
    std: [...]

  classes:
    - ...
```

Các giá trị phải đến từ training code thực tế.

Không copy giá trị ví dụ nếu không khớp checkpoint.

---

# 7. Training/runtime equivalence

Phase 6 phải có test hoặc smoke validation để chứng minh:

```text
same source clip
+
same ROI
+
same sampling
+
same preprocessing
+
same checkpoint
```

cho output tương đương giữa:

```text
training/evaluation inference
```

và:

```text
production runtime inference
```

Tolerance phải phù hợp FP32/FP16.

Nếu hai đường inference khác nhau đáng kể, không PASS.

---

# 8. Proposal Builder

Proposal Builder nhận đầu vào:

```text
TrackingFrame
+
Seat-Stable Identity
+
Seat adjacency
```

và tạo hai loại proposal:

```text
SINGLE
PAIR
```

---

# 9. Single Actor Proposal

Mỗi:

```text
ASSIGNED SessionCandidate
```

có thể tạo một Single Actor Proposal.

Stable proposal ID:

```text
single:<session_candidate_id>
```

Không dùng:

```text
single:<track_id>
```

vì Track ID không ổn định.

Ví dụ:

```text
single:SC103
```

metadata:

```text
proposal_type = SINGLE
primary_actor = SC103
seat = B03
track_id = current runtime track only
```

Track ID chỉ là runtime source.

---

# 10. Actor-Context ROI

Single ROI không chỉ là raw detector bbox.

Nó phải giữ:

```text
head
torso
hands
desk/context
```

Baseline:

```text
actor bbox
+
horizontal expansion
+
top expansion nhỏ
+
bottom/desk expansion
```

Config phải centralized.

Ví dụ engineering baseline:

```yaml
single_roi:
  expand_left: 0.15
  expand_right: 0.15
  expand_top: 0.05
  expand_bottom: 0.20
```

Không coi đây là giá trị khoa học tối ưu.

Clamp ROI vào frame.

---

# 11. Single Proposal class relevance

Single proposal đặc biệt phù hợp với:

```text
normal
suspicious_looking
using_phone/cheat_sheet
```

Tuy nhiên baseline TSM vẫn output đủ 5 classes.

Không hard-zero communicating/exchange_object ở Phase 6.

---

# 12. Adjacent Pair Proposal

Không tạo mọi cặp candidate.

Dùng Seat adjacency graph.

Ví dụ:

```text
A01 ↔ A02
A02 ↔ A03
A03 ↔ A04
```

Chỉ tạo pair nếu:

```text
Seat A adjacent Seat B
AND
both Seats have active ASSIGNED candidate identity
```

Proposal ID phải ổn định:

```text
pair:<session_candidate_id_A>:<session_candidate_id_B>
```

Canonicalize order để:

```text
pair:A:B
```

và:

```text
pair:B:A
```

không trở thành hai proposal khác nhau.

---

# 13. Seat adjacency

Adjacency phải là domain/runtime geometry, không all-pairs.

Có thể sinh từ:

```text
seat layout geometry
```

hoặc cấu hình adjacency rõ ràng.

Baseline ưu tiên:

```text
left/right neighboring seats
```

Nếu camera/layout có hàng trước/sau tương tác thực tế, adjacency có thể bổ sung.

Không thêm diagonal/front/back pair tùy tiện.

Document exact rule.

---

# 14. Pair ROI

Pair ROI:

```text
union(actor_bbox_A, actor_bbox_B)
+
interaction/context margin
```

Phải giữ:

```text
actor A
actor B
space between actors
hands
desk interaction region
```

Pair ROI dùng chủ yếu cho:

```text
communicating
exchange_object
look_at_neighbor_paper
```

nhưng baseline vẫn dùng cùng 5-class model.

---

# 15. Pair gating

Không tạo Pair Proposal khi:

```text
one seat empty
one actor UNASSIGNED
one actor TENTATIVE
track missing outside grace policy
```

Chỉ tạo pair khi identity đủ tin cậy.

Phase 6 ưu tiên proposal đúng hơn proposal nhiều.

---

# 16. Proposal lifecycle

Stable proposal identity không phụ thuộc Track ID.

Ví dụ:

```text
T17 → SC103
```

sau fragmentation:

```text
T24 → SC103
```

proposal vẫn:

```text
single:SC103
```

Do đó temporal buffer có thể tiếp tục theo business identity nếu temporal gap còn hợp lệ.

Nếu gap quá lớn:

```text
reset temporal buffer
```

theo config.

---

# 17. Temporal Buffer

Mỗi proposal có temporal buffer riêng.

Ví dụ:

```text
single:SC103
→ timestamped ROI frames
```

Pair:

```text
pair:SC103:SC104
→ timestamped Pair ROI frames
```

Không lưu temporal buffer vào PostgreSQL.

Runtime-only.

---

# 18. ROI geometry từng timestamp

Không sử dụng bbox từ frame đầu cho toàn clip.

Đúng:

```text
t0 → bbox0 → ROI0
t1 → bbox1 → ROI1
t2 → bbox2 → ROI2
...
```

Pair:

```text
t0 → A0+B0 → pairROI0
t1 → A1+B1 → pairROI1
...
```

Có thể smooth ROI geometry nhẹ để giảm jitter.

Không smooth quá mạnh làm mất motion cue.

---

# 19. Temporal sampling

Temporal sampling phải tuân theo TSM runtime contract.

Không tự chọn:

```text
8 frame
16 frame
2 seconds
```

nếu training không dùng như vậy.

Sampling nên timestamp-based thay vì phụ thuộc AI FPS.

Lý do:

```text
GTX1650
RTX3060
Colab T4
```

có inference FPS khác nhau.

---

# 20. Missing frame handling

Không duplicate frame tùy tiện nếu buffer thiếu data.

Phải xác định policy rõ:

```text
not enough temporal coverage
→ proposal not ready
```

hoặc training-compatible padding nếu training thực sự dùng padding.

Do not silently create fake temporal sequences.

---

# 21. TSM inference scheduling

YOLO/ByteTrack có thể chạy khoảng:

```text
10–12.5 FPS
```

nhưng TSM không cần chạy ở cùng rate.

Action inference có thể dùng:

```text
windowed inference
+
stride
```

Ví dụ concept:

```text
clip span X ms
prediction every Y ms
```

nhưng X/Y phải được cấu hình dựa trên runtime contract và compute benchmark.

Không infer mọi proposal ở mọi detector frame.

---

# 22. Shared TSM

Phase 6 sử dụng một shared TSM R3 checkpoint.

Không tạo:

```text
Single TSM
Pair TSM
```

thành hai model riêng.

Pipeline:

```text
Single clip ─┐
             ├→ shared TSM R3
Pair clip ───┘
```

Metadata giữ proposal type.

---

# 23. Batch inference

Khi nhiều proposal ready gần cùng timestamp:

```text
batch them
```

nếu VRAM cho phép.

Mục tiêu:

```text
reduce GPU launch overhead
```

Nhưng không tạo backlog.

Freshness ưu tiên hơn processing mọi proposal.

---

# 24. Backpressure

Không được để TSM queue tăng vô hạn.

Use:

```text
bounded queue
latest-ready semantics
drop stale inference requests
```

nếu runtime không theo kịp.

Track/Seat realtime path không được block lâu bởi action recognition.

---

# 25. GPU resource rule

GTX1650 Max-Q khoảng 4GB là machine hiện tại.

TSM phải chia GPU với YOLO.

Do đó:

```text
FP16 when safe
small dynamic batch
bounded queue
no duplicate model instance
```

Một runtime không được load nhiều TSM checkpoint giống nhau.

---

# 26. Model lifecycle

Load TSM:

```text
once per process/runtime manager
```

theo architecture hiện tại.

Không load checkpoint mỗi clip.

Không recreate model mỗi inference.

---

# 27. Inference output

Output runtime object:

```text
ActionPrediction
```

Concept:

```json
{
  "proposal_id": "single:SC103",
  "proposal_type": "SINGLE",
  "timestamp_ms": 53120,
  "actors": ["SC103"],
  "seat_codes": ["B03"],
  "probabilities": {
    "normal": 0.10,
    "suspicious_looking": 0.72,
    "communicating": 0.04,
    "exchange_object": 0.02,
    "using_phone_cheat_sheet": 0.12
  }
}
```

Pair:

```json
{
  "proposal_id": "pair:SC103:SC104",
  "proposal_type": "PAIR",
  "actors": ["SC103", "SC104"]
}
```

---

# 28. Raw prediction only

Phase 6 không tạo:

```text
Event
Violation
Alert
Confirmed behavior
```

ActionPrediction chỉ là model evidence.

Ví dụ:

```text
communicating = 0.76
```

không đồng nghĩa:

```text
Event created immediately
```

Phase 7 sẽ xử lý temporal aggregation.

---

# 29. UI

Normal Monitoring UI không cần spam raw probabilities cho operator.

Có thể có developer diagnostics:

```text
proposal ID
proposal type
top-1 class
confidence
TSM latency
buffer readiness
```

Normal mode nên giữ tập trung vào monitoring.

Do not show raw technical panel permanently.

---

# 30. Diagnostics

Collect:

```text
active_single_proposals
active_pair_proposals

ready_temporal_buffers

action_inference_fps
action_inference_latency_ms mean/P95

batch_size mean/P95

action_queue_depth

stale_requests_dropped

GPU memory if available

per-proposal readiness
```

Do not persist frame-level diagnostics to PostgreSQL.

---

# 31. Proposal tests

Unit tests:

```text
ASSIGNED candidate
→ Single Proposal

TENTATIVE
→ no Single Proposal

UNASSIGNED
→ no Single Proposal

two adjacent ASSIGNED candidates
→ one Pair Proposal

non-adjacent candidates
→ no Pair Proposal

pair order canonical
```

---

# 32. ROI tests

Test:

```text
single ROI expansion
pair union
frame clipping
tiny bbox
edge-of-frame bbox
adjacent actor ROI
```

No GPU required.

---

# 33. Buffer tests

Test:

```text
timestamp ordering
duplicate timestamp
out-of-order frame
buffer trimming
proposal identity continuity
Track fragmentation same SessionCandidate
large temporal gap reset
not-ready behavior
```

---

# 34. TSM preprocessing tests

Verify:

```text
frame count
tensor shape
dtype
color space
normalization
resize/crop
class order
```

against training contract.

This is mandatory.

---

# 35. Inference equivalence test

Use known clips/examples from training/evaluation pipeline.

Compare:

```text
reference inference
vs
runtime inference
```

Same checkpoint.

Report:

```text
max absolute logit difference
max probability difference
top-1 agreement
```

Do not claim equivalence without measurable result.

---

# 36. Runtime integration validation

Test complete:

```text
video
→ detection
→ tracking
→ seat identity
→ proposal
→ temporal buffer
→ TSM prediction
```

on real project videos.

Check:

```text
no unbounded lag
no duplicate proposal
no model reload
no runtime duplication
```

---

# 37. Performance benchmark

Measure on available machine:

```text
YOLO + ByteTrack only
```

versus:

```text
YOLO + ByteTrack + Seat Identity + TSM
```

Report:

```text
AI tracking FPS

action prediction rate

detector latency
tracker latency
seat latency
TSM preprocessing latency
TSM inference latency

GPU VRAM

analysis lag
dropped stale action requests
```

If NVIDIA driver unavailable, clearly report missing GPU measurements.

---

# 38. Validation dataset

Use representative clips containing:

```text
normal seated behavior
looking left/right/backward
neighbor interaction
communication
object exchange
phone/cheat-sheet if available
occlusion
fragmentation
```

Do not use only easy normal clips.

---

# 39. Scientific separation

Phase 6 baseline is:

```text
TSM R3
+
candidate-aware Single/Pair runtime proposal system
```

Head Pose and Desk/Object are separate future ablations.

Do not mix them now.

---

# 40. Definition of Done

Phase 6 PASS only when:

```text
TSM training/runtime contract is documented

runtime preprocessing matches training

Single Proposal stable by SessionCandidate ID

Adjacent Pair Proposal uses Seat adjacency

temporal buffers are timestamp-correct

TSM checkpoint loads once

raw 5-class predictions are produced

no Event is created

no unbounded backlog exists

real-video end-to-end inference runs

tests pass

performance benchmark is reported
```

If equivalence fails or temporal sampling is uncertain:

```text
PHASE 6: NOT PASSED
```

Do not hide the mismatch.

---

# 41. Output of Phase 6

Final architecture after Phase 6:

```text
Video
→ YOLO
→ ByteTrack
→ Seat-Stable Identity
→ Proposal Builder
   ├─ Single Actor Proposal
   └─ Adjacent Pair Proposal
→ ROI Temporal Buffer
→ TSM R3
→ ActionPrediction
```

Phase 7 will consume `ActionPrediction`.
