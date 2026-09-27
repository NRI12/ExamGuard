# ExamGuard RBAC Specification

## 1. Roles

```text
ADMIN
SUPERVISOR
REVIEWER
```

Không thêm role mới trong thesis scope hiện tại.

## 2. Permission constants

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

## 3. Role → Permission

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
All permissions.

## 4. API authorization matrix

### Auth

```text
POST /api/v1/auth/login         PUBLIC
GET  /api/v1/auth/me            AUTHENTICATED
```

### Rooms / Seats

```text
GET   /api/v1/rooms                 room.read
GET   /api/v1/rooms/{id}            room.read
POST  /api/v1/rooms                 room.manage
PATCH /api/v1/rooms/{id}            room.manage
GET   /api/v1/rooms/{id}/seats      room.read
PUT   /api/v1/rooms/{id}/seats      room.manage
```

### Candidates

```text
GET   /api/v1/candidates            candidate.read
GET   /api/v1/candidates/{id}       candidate.read
POST  /api/v1/candidates            candidate.manage
PATCH /api/v1/candidates/{id}       candidate.manage
```

### Sessions

```text
GET   /api/v1/sessions                  session.read
GET   /api/v1/sessions/{id}             session.read
POST  /api/v1/sessions                  session.manage
PATCH /api/v1/sessions/{id}             session.manage
PUT   /api/v1/sessions/{id}/candidates  session.manage
```

Monitoring:

```text
POST /api/v1/sessions/{id}/start   session.monitor
POST /api/v1/sessions/{id}/pause   session.monitor
POST /api/v1/sessions/{id}/resume  session.monitor
POST /api/v1/sessions/{id}/stop    session.monitor
POST /api/v1/sessions/{id}/seek    session.monitor
```

### Media

```text
POST /api/v1/media/videos               media.upload
GET  /api/v1/media/{id}                 media.read
GET  /api/v1/media/{id}/content         media.read
GET  /api/v1/media/{id}/frame           media.read
```

### WebSocket

```text
/ws/monitoring/{session_id}         tracking.read
```

WebSocket phải authenticate và authorize trước khi subscribe. Observer không được tự tạo runtime.

### Events

```text
GET  /api/v1/events                 event.read
GET  /api/v1/events/{id}            event.read
POST /api/v1/events/manual          event.create_manual
POST /api/v1/events/{id}/reviews    event.review
```

AI Event creation là internal service behavior.

Expected:

```text
SUPERVISOR POST review → 403
REVIEWER   POST review → allowed
ADMIN      POST review → allowed
```

### Evidence

```text
GET  /api/v1/events/{id}/evidence      evidence.read
GET  /api/v1/evidence/{id}             evidence.read
POST /api/v1/evidence/{id}/lock        evidence.manage
POST /api/v1/evidence/{id}/unlock      evidence.manage
```

### Appeals

```text
GET   /api/v1/appeals               appeal.read
GET   /api/v1/appeals/{id}          appeal.read
POST  /api/v1/appeals               appeal.manage
PATCH /api/v1/appeals/{id}          appeal.manage
```

### Reports

```text
GET  /api/v1/reports/...            report.read
POST /api/v1/reports/.../export     report.export
```

### Users

```text
GET   /api/v1/users                 user.manage
POST  /api/v1/users                 user.manage
PATCH /api/v1/users/{id}            user.manage
```

### Audit / Settings / Diagnostics

```text
GET audit                           audit.read
GET relevant event/evidence audit   audit.read_relevant
GET/PATCH system settings           system.manage
GET diagnostics                     diagnostics.read
```

## 5. Backend pattern

```text
get_current_user
→ require_permission(permission)
→ route
→ service business rules
```

Không duplicate raw role checks ở nhiều route.

## 6. Frontend pattern

Tạo/reuse:

```text
useAuth()
usePermissions()
can(permission)
```

Sidebar và action visibility dựa vào centralized permission map.

Direct URL unauthorized phải hiện:

```text
Bạn không có quyền truy cập chức năng này.
```

không redirect user đã đăng nhập về login.

## 7. Error contract

Authenticated but forbidden:

```http
403 Forbidden
```

Example:

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "You do not have permission to perform this action."
  }
}
```

Frontend:

```text
Bạn không có quyền thực hiện thao tác này.
```

## 8. Tests

### SUPERVISOR
Allowed:
- read rooms;
- candidate operations theo spec;
- session setup/assignment;
- media upload/read;
- monitoring control;
- read events;
- create manual event;
- read evidence.

Forbidden:
- room/seat manage;
- EventReview;
- user manage;
- settings;
- full audit.

### REVIEWER
Allowed:
- read rooms/candidates/sessions/events;
- create EventReview;
- evidence operations theo spec;
- appeal operations.

Forbidden:
- room/seat/candidate modify;
- monitoring control;
- user/settings manage.

### ADMIN
Allowed all application permissions, vẫn chịu business-state validation.

## 9. Database

Không thêm permission tables. Nếu `users.role` đã có 3 role thì không migration.

## 10. Definition of Done

PASS khi:
- Supervisor không review Event;
- Reviewer không control monitoring;
- Reviewer không sửa Candidate/Room;
- Admin quản lý Users/Settings;
- API trả 403 đúng;
- WebSocket được bảo vệ;
- frontend menu/action đúng role;
- direct URL được bảo vệ;
- audit ghi privileged actions.
