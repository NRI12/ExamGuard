# ExamGuard RBAC Context

## Mục tiêu

RBAC của ExamGuard phải phục vụ đúng workflow phòng thi, dễ kiểm thử, dễ giải thích trong đồ án và không over-engineer.

Các role đăng nhập hiện tại:

```text
ADMIN
SUPERVISOR
REVIEWER
```

Không thêm role mới nếu chưa có yêu cầu nghiệp vụ rõ ràng.

## Tác nhân

### ADMIN — Quản trị viên
Quản trị user, phòng thi, seat layout, candidate, session, settings, reports, audit và diagnostics.

ADMIN vẫn phải tôn trọng lịch sử dữ liệu:
- không hard delete Event/EventReview/AuditLog;
- không overwrite AI prediction;
- không overwrite review lịch sử.

### SUPERVISOR — Giám thị
Workflow:

```text
Đăng nhập
→ chuẩn bị phiên thi
→ chọn phòng
→ gán thí sinh vào ghế
→ kiểm tra readiness
→ bắt đầu giám sát
→ pause/resume/stop
→ theo dõi
→ xem sự kiện
→ tạo Manual Event nếu AI bỏ sót
```

SUPERVISOR được vận hành nhưng không quyết định cuối cùng Event.

Nguyên tắc:

```text
Supervisor vận hành
Reviewer xác minh
```

### REVIEWER — Người xác minh
Workflow:

```text
Sự kiện chờ xác minh
→ xem context/evidence
→ xem candidate/session
→ Confirm / Dismiss / Needs Review
→ xử lý Appeal nếu có
```

Reviewer không được start/stop monitoring và không sửa dữ liệu setup như Room/Seat/Candidate.

### Candidate
Candidate là domain entity, không phải user đăng nhập.

### AI Runtime
AI Runtime không phải user RBAC. AI tạo Event nội bộ với `source=AI`.

### System
Internal actor cho evidence generation, audit và system jobs.

## Nguyên tắc RBAC

Backend là nguồn quyết định:

```text
request
→ authenticate
→ resolve user
→ resolve role
→ check permission
→ execute service
```

Frontend chỉ dùng RBAC để ẩn menu/button và tối ưu UX.

Không cần bảng `roles`, `permissions`, `role_permissions`, `user_permissions` với 3 role cố định. Dùng `users.role` + centralized permission map trong code.

## Permission identifiers

```text
dashboard.read

room.read
room.manage

candidate.read
candidate.manage

session.read
session.manage
session.monitor

media.read
media.upload

tracking.read

event.read
event.create_manual
event.review

evidence.read
evidence.manage

appeal.read
appeal.manage

report.read
report.export

user.manage

audit.read
audit.read_relevant

system.manage
diagnostics.read
```

## Role mapping

### SUPERVISOR

```text
dashboard.read
room.read
candidate.read
candidate.manage
session.read
session.manage
session.monitor
media.read
media.upload
tracking.read
event.read
event.create_manual
evidence.read
report.read
```

### REVIEWER

```text
dashboard.read
room.read
candidate.read
session.read
tracking.read
event.read
event.review
evidence.read
evidence.manage
appeal.read
appeal.manage
report.read
audit.read_relevant
```

### ADMIN

```text
all application permissions
```

## Permission matrix

| Resource | SUPERVISOR | REVIEWER | ADMIN |
|---|---|---|---|
| Dashboard | Read | Read | Read |
| Room | Read | Read | Manage |
| Seat Layout | Read | Read | Manage |
| Candidate | Create/Read/Update | Read | Manage |
| Exam Session | Create/Read/Update | Read | Manage |
| SessionCandidate | Manage | Read | Manage |
| Media | Upload/Read | Read | Manage |
| Monitoring | Control | View | Full |
| Tracking | View | View | View |
| Event | Read + Manual Create | Read | Read |
| Event Review | No | Create/Read | Create/Read |
| Evidence | Read | Read + Lock | Full |
| Appeal | Read | Create/Read/Update | Full |
| Report | Read | Read | Full |
| Audit | Basic/own | Relevant | Full |
| Users | No | No | Full |
| Settings | No | No | Full |
| Diagnostics | Basic only | No | Full |

## State-aware authorization

RBAC luôn kết hợp business state.

Ví dụ:
- `session.manage` không cho sửa mọi field khi session đã RUNNING/COMPLETED.
- `event.review` chỉ tạo review mới, không overwrite AI prediction hoặc review cũ.
- Evidence lock/unlock phải tuân theo Appeal/historical integrity.

## UI theo role

### SUPERVISOR

```text
Giám sát
Phiên thi
Thí sinh
Sự kiện
```

### REVIEWER

```text
Cần xác minh
Sự kiện
Thí sinh
Phiên thi
Khiếu nại
```

### ADMIN

```text
Giám sát
Phiên thi
Thí sinh
Sự kiện
Báo cáo

Cài đặt
  Phòng thi
  Người dùng
  Hệ thống
```

Không tạo fake page nếu feature chưa tồn tại.

## Audit

Audit business actions như:

```text
USER_CREATED
USER_UPDATED
USER_DEACTIVATED
ROOM_CREATED
ROOM_UPDATED
SEAT_LAYOUT_UPDATED
SESSION_CREATED
SESSION_UPDATED
SESSION_CANDIDATES_UPDATED
SESSION_STARTED
SESSION_PAUSED
SESSION_RESUMED
SESSION_STOPPED
MANUAL_EVENT_CREATED
EVENT_REVIEW_CREATED
EVIDENCE_LOCKED
EVIDENCE_UNLOCKED
APPEAL_CREATED
APPEAL_UPDATED
```

Không audit frame/bbox/WebSocket message.

## Definition of Done

```text
SUPERVISOR
→ vận hành session
→ không review Event
→ không quản lý Users/Settings

REVIEWER
→ review Event/Evidence
→ không control monitoring
→ không sửa Room/Candidate

ADMIN
→ quản trị đầy đủ
```

Restricted API phải trả `403`; frontend phải ẩn/disable action tương ứng nhưng backend vẫn là lớp quyết định cuối cùng.
