# ExamGuard — Seat-Stable Identity Context

## 1. Mục tiêu

ExamGuard sử dụng camera cố định trong phòng thi.

YOLO + ByteTrack hiện cung cấp:

```text
Track ID
bbox
confidence
timestamp
```

Nhưng:

```text
Track ID != Candidate Identity
```

ByteTrack ID chỉ là runtime identity.

Ví dụ:

```text
T17
→ occlusion
→ lost
→ T24
```

không có nghĩa đây là hai thí sinh khác nhau.

Trong phòng thi, mỗi thí sinh đã được gán trước vào một vị trí:

```text
Candidate
→ SessionCandidate
→ Seat
```

Do đó identity ổn định phải dựa trên:

```text
Track
→ Seat
→ SessionCandidate
→ Candidate
```

Lớp này được gọi là:

```text
Seat-Stable Identity
```

---

# 2. Seat Location và Seat-Stable Identity khác nhau

## Seat Location

Chỉ trả lời:

```text
Track T17 hiện đang gần Seat B03
```

Đây là geometric matching ở một thời điểm.

## Seat-Stable Identity

Trả lời:

```text
Track T17
→ ổn định tại B03
→ B03 được gán cho SessionCandidate SC103
→ Candidate SV103
```

và giữ assignment ổn định theo thời gian.

Nếu tracker fragmentation:

```text
T17
→ lost
→ T24
```

nhưng T24 xuất hiện lại tại B03:

```text
T24
→ B03
→ SV103
```

Candidate identity được phục hồi từ Seat.

---

# 3. Không dùng Track ID làm business identity

Không lưu:

```text
Track 17 = SV103
```

vĩnh viễn.

Không tạo database relationship:

```text
track_id → candidate_id
```

Track ID chỉ tồn tại trong runtime.

Business identity:

```text
Seat
→ SessionCandidate
→ Candidate
```

---

# 4. Database hiện tại

Không cần thêm bảng tracking.

Sử dụng:

```text
rooms
seats
candidates
exam_sessions
session_candidates
```

Quan hệ:

```text
ExamSession
      │
      ▼
SessionCandidate
   │        │
   ▼        ▼
Candidate  Seat
```

Ví dụ:

```text
ExamSession S01

B03
→ SessionCandidate SC103
→ Candidate SV103
```

Seat-Stable Identity là runtime state.

Không persist:

```text
bbox per frame
Track ID history
Track→Seat every frame
Seat occupancy every frame
```

vào PostgreSQL.

---

# 5. Seat geometry

Seat sử dụng normalized coordinates:

```text
x
y
width
height
```

range:

```text
0.0 → 1.0
```

Backend matching phải dùng source-frame normalized coordinates.

Không dùng browser pixel coordinates.

---

# 6. Ý nghĩa Seat Region

Seat region không phải chỉ là chiếc ghế vật lý.

Nó đại diện cho:

```text
expected occupancy zone
```

của candidate khi ngồi thi.

Nên bao phủ hợp lý:

```text
head
shoulders
torso
hands
desk interaction area
```

Không phụ thuộc vào chân vì trong camera phòng thi chân thường bị bàn che.

---

# 7. Calibration

Seat layout phải được cấu hình trên:

```text
real frame from room/session
```

không dùng background trắng.

UI:

```text
Video/reference frame
+
normalized seat rectangles
```

Người dùng có thể:

```text
drag
resize
rename
save
```

Seat rectangle.

---

# 8. Input từ Tracker

Input mỗi AI timestamp:

```text
TrackingFrame
{
    timestamp_ms,
    tracks[]
}
```

mỗi Track có:

```text
track_id
bbox_norm
confidence
```

Dùng bbox visible-person từ ByteTrack.

Không bắt buộc full-body.

Không crop upper-body trước Seat Assignment.

---

# 9. Seat matching

Với mỗi:

```text
Track T
Seat S
```

tính geometric score.

Baseline:

$$
Score(T,S)
=
w_o O(T,S)
+
w_d D(T,S)
$$

trong đó:

```text
O = overlap score
D = distance score
```

Initial values:

```yaml
overlap_weight: 0.70
distance_weight: 0.30
```

Đây là engineering baseline, không phải giá trị tối ưu khoa học.

---

# 10. Overlap score

Khuyến nghị ban đầu:

$$
O(T,S)
=
\frac{Area(T \cap S)}
{Area(T)}
$$

Ý nghĩa:

```text
bao nhiêu phần bbox người nằm trong seat occupancy region
```

Không nhất thiết dùng IoU vì Seat region và person bbox có scale khác nhau.

Exact formula phải được document và test.

---

# 11. Distance score

Tính center:

```text
track center
seat center
```

Distance:

$$
d =
\frac{
||c_T-c_S||
}{
diag(S)
}
$$

sau đó:

$$
D = clamp(1-d,0,1)
$$

Có thể thay normalization nếu benchmark chứng minh tốt hơn, nhưng phải nhất quán.

---

# 12. Seat expansion

Cho phép tolerance nhỏ:

```yaml
seat_expand_ratio: 0.08
```

để chịu:

```text
bbox jitter
candidate leaning
minor seat calibration error
```

Không expand quá lớn làm Seat B03 và B04 overlap mạnh.

---

# 13. Hard gating

Không bắt mọi Track phải thuộc một Seat.

Nếu best score thấp:

```text
UNASSIGNED
```

Điều này cần cho:

```text
supervisor
candidate đứng khỏi chỗ
extra person
false track
person walking
```

Sai Candidate identity nguy hiểm hơn temporary UNASSIGNED.

---

# 14. Track assignment states

Mỗi Track có state:

```text
UNASSIGNED
TENTATIVE
ASSIGNED
```

Luồng:

```text
UNASSIGNED
   │
   │ plausible seat
   ▼
TENTATIVE
   │
   │ stable for confirm_ms
   ▼
ASSIGNED
```

Không assign Candidate từ một frame duy nhất.

---

# 15. Time-based hysteresis

Dùng video/analysis timestamp.

Không dùng số frame.

Initial configuration:

```yaml
seat_assignment:
  min_score: 0.35

  confirm_ms: 600
  release_ms: 1500

  switch_margin: 0.15
  switch_confirm_ms: 800
```

Lý do:

```text
GTX1650 có AI FPS khác RTX3060
```

Identity behavior không được phụ thuộc trực tiếp FPS.

---

# 16. Stable assignment

Giả sử:

```text
T17 → B03
score 0.82
```

frame tiếp theo:

```text
B03 = 0.63
B04 = 0.67
```

Không switch ngay.

Seat switch chỉ được xem xét khi:

$$
Score_{new}
>
Score_{current} + switch\_margin
$$

và điều kiện giữ đủ:

```text
switch_confirm_ms
```

---

# 17. One-to-one constraint

Phải enforce:

```text
one Track
→ at most one Seat
```

và:

```text
one Seat
→ at most one active Track
```

Ví dụ:

```text
T17 → B03 score .84
T21 → B03 score .69
```

không được assign cả hai.

---

# 18. Conflict resolution

Baseline đủ cho thesis:

1. preserve valid strong confirmed assignments;
2. build all valid Track↔Seat candidates above min score;
3. sort descending by score;
4. greedily assign remaining Track and Seat;
5. apply temporal confirmation/hysteresis.

Không thêm SciPy/Hungarian lúc đầu.

Hungarian chỉ là future ablation nếu conflict benchmark cho thấy cần.

---

# 19. Seat runtime state

Seat có thể có:

```text
EMPTY
OCCUPIED
GRACE
```

Ví dụ:

```text
B03 OCCUPIED
↓
T17 lost
↓
B03 GRACE
↓
T24 recovered at B03
↓
B03 OCCUPIED
```

Nếu không recover sau thời gian:

```text
B03 EMPTY
```

---

# 20. Tracker fragmentation recovery

Core requirement:

```text
T17 → B03 → SV103
```

sau occlusion:

```text
T17 disappears
```

rồi:

```text
T24 appears at B03
```

kết quả phải trở lại:

```text
T24 → B03 → SV103
```

Không cố phục hồi old Track ID.

Candidate identity được phục hồi qua Seat.

---

# 21. Candidate rời ghế

Nếu:

```text
SV103 đứng dậy
```

Track có thể đi khỏi B03.

Sau khi geometry không còn phù hợp đủ lâu:

```text
T17 → UNASSIGNED
```

nhưng database vẫn:

```text
SV103 assigned to B03
```

Đây là hai khái niệm khác nhau:

```text
planned seat assignment
vs
current runtime occupancy
```

---

# 22. Supervisor

Supervisor đi qua phòng không được gán Candidate.

Ví dụ:

```text
T31
```

không đủ score với Seat nào:

```text
T31 → UNASSIGNED
```

Normal UI có thể:

```text
1 người chưa xác định
```

Không gọi đây là hành vi bất thường.

---

# 23. Pause

Pause không được làm:

```text
TENTATIVE
→ ASSIGNED
```

do wall-clock chạy.

Confirmation dùng:

```text
video timestamp
analysis timestamp
```

không dùng `time.time()` đơn thuần.

---

# 24. Seek

Arbitrary seek phải reset:

```text
ByteTrack state
Track→Seat state
tentative timers
assigned track states
seat occupancy states
identity runtime cache
```

Sau seek:

```text
rebuild identity from new frames
```

Không dùng old timeline assignment.

---

# 25. Resume

Normal Pause → Resume:

```text
do not duplicate runtime
do not create second tracker
```

Nếu timeline vẫn liên tục, có thể giữ Seat identity state.

Nếu seek xảy ra:

```text
reset
```

---

# 26. Runtime restart

Sau restart:

```text
Track IDs có thể thay đổi
```

điều này bình thường.

Seat identity phải tự rebuild.

---

# 27. Stable Proposal identity

Sau Phase này, downstream Proposal Builder không nên phụ thuộc Track ID.

Single Actor Proposal:

```text
single:<session_candidate_id>
```

ví dụ:

```text
single:SC103
```

Adjacent Pair Proposal:

```text
pair:<SC103>:<SC104>
```

Không dùng:

```text
pair:T17:T24
```

vì Track IDs không ổn định.

---

# 28. Runtime domain objects

Khuyến nghị các model không ORM:

```text
SeatMatchCandidate
TrackSeatState
SeatRuntimeState
TrackIdentity
SeatIdentitySnapshot
```

Ví dụ:

```python
TrackIdentity(
    track_id=17,
    assignment_state="ASSIGNED",
    seat_id="...",
    seat_code="B03",
    session_candidate_id="...",
    score=0.82,
)
```

Không truyền SQLAlchemy ORM object trực tiếp vào realtime pipeline.

---

# 29. WebSocket payload

Extend TrackingFrame:

```json
{
  "type": "tracking",
  "timestamp_ms": 53240,
  "tracks": [
    {
      "track_id": 17,
      "bbox_norm": [0.214, 0.182, 0.326, 0.784],
      "confidence": 0.91,
      "identity": {
        "state": "ASSIGNED",
        "seat_id": "uuid",
        "seat_code": "B03",
        "session_candidate_id": "uuid",
        "score": 0.82
      }
    }
  ]
}
```

Normal frontend không cần hiển thị `score`.

Debug mode có thể hiển thị.

---

# 30. Frontend lookup

Frontend load session assignments qua REST:

```text
session_candidate_id
→ candidate_code
→ candidate name
```

WebSocket không cần gửi full Candidate object ở AI FPS.

Normal overlay:

```text
B03 • SV103
```

Debug:

```text
B03 • SV103
T17
score=.82
```

---

# 31. Candidate panel

Không hiển thị technical Track list như giao diện chính.

Thay:

```text
T17
T21
T25
```

bằng:

```text
A01 • SV101   Đang theo dõi
A02 • SV102   Đang theo dõi
A03 • SV103   Tạm mất dấu
```

Unassigned:

```text
Người chưa xác định: 1
```

---

# 32. Operational statuses

Vietnamese labels:

```text
ASSIGNED
→ Đang theo dõi

TENTATIVE
→ Đang xác định

GRACE
→ Tạm mất dấu

UNASSIGNED
→ Chưa xác định

EMPTY
→ Ghế trống
```

Không dùng:

```text
gian lận
vi phạm
đáng ngờ
```

ở Seat Identity layer.

---

# 33. Performance

Typical room:

```text
6–30 Tracks
6–30 Seats
```

Seat matching complexity baseline:

$$
O(N_{tracks}N_{seats})
$$

vẫn rất nhỏ.

Seat Assignment chạy CPU.

Không dùng GPU.

Metric:

```text
seat_assignment_ms
```

phải nhỏ hơn đáng kể detector latency.

---

# 34. Diagnostics

Developer mode có:

```text
assigned_tracks
tentative_tracks
unassigned_tracks

occupied_seats
grace_seats
empty_seats

seat_assignment_ms
seat_switch_count

identity_recovery_count
```

Không đưa các metric này hết ra normal UI.

---

# 35. Configuration

Central config:

```yaml
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
```

Không hardcode rải rác.

Validate config.

---

# 36. Unit tests

Phải có:

```text
rectangle intersection
seat expansion
center calculation
distance score
overlap score
score calculation
```

Matching:

```text
one Track → correct Seat

far Track → UNASSIGNED

adjacent Tracks → different Seats

two Tracks conflict → one Seat only
```

Temporal:

```text
single-frame candidate
→ TENTATIVE

stable candidate
→ ASSIGNED

one-frame jitter
→ no switch

sustained new Seat
→ switch

Track disappears
→ GRACE

new Track appears at same Seat
→ same Candidate recovered
```

Lifecycle:

```text
pause
seek
resume
runtime restart
```

---

# 37. Real-video validation

Không PASS bằng unit tests בלבד.

Test trên video thật gồm:

```text
candidate seated normally
lean left/right
look backward
candidate adjacent interaction
partial occlusion
candidate standing
supervisor walking
ByteTrack ID fragmentation
```

---

# 38. Metrics

Đánh giá manually labeled validation segments.

### Correct Seat Assignment Rate

$$
CSA=
\frac{correct\ assigned\ candidate\ observations}
{eligible\ candidate\ observations}
$$

### Wrong Seat Rate

$$
WSR=
\frac{wrong\ seat\ assignments}
{evaluated\ assignments}
$$

### Unassigned Rate

Trong các candidate đáng lẽ assign được:

$$
UR=
\frac{UNASSIGNED}
{eligible\ candidate\ observations}
$$

### False Seat Switches

Số:

```text
B03 → B04 → B03
```

khi candidate thực tế không đổi ghế.

### Stabilization Delay

```text
Track first visible
→ ASSIGNED
```

### Fragmentation Recovery Delay

```text
new Track ID appears
→ correct Candidate recovered
```

---

# 39. Acceptance philosophy

Priority:

```text
correct Candidate
>
fast Candidate
```

Ưu tiên:

```text
temporary UNASSIGNED
```

thay vì:

```text
wrong Candidate
```

---

# 40. Definition of Done

Seat-Stable Identity PASS khi:

```text
1. Stable seated candidate
   → stable Seat/Candidate identity

2. Track fragmentation
   → Candidate recovered via Seat

3. Adjacent bbox jitter
   → no frequent candidate switching

4. Supervisor
   → UNASSIGNED

5. Candidate leaves seat
   → runtime assignment released

6. Seek
   → clean reset/rebuild

7. Pause/resume
   → no duplicate timer/runtime behavior

8. No frame-level tracking identity persisted to PostgreSQL

9. Normal UI displays Seat + Candidate instead of raw Track ID

10. Proposal Builder can use SessionCandidate identity rather than Track ID
```

Seat-Stable Identity is infrastructure for Action Recognition.

It does NOT classify behavior and does NOT create Event.
