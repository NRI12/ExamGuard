# CODEX PROMPT — IMPLEMENT RBAC ON CURRENT EXAMGUARD SYSTEM

You are continuing an existing ExamGuard codebase.

Do NOT rebuild the application.

This task is only to implement and standardize Role-Based Access Control on the current system.

Read first:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `PROJECT_CONTEXT.md`
- `DATABASE_SCHEMA.md`
- `SYSTEM_ARCHITECTURE.md`
- `API_CONTRACT.md`
- `UI_UX_SPEC.md`
- `RBAC_CONTEXT.md`
- `RBAC_SPEC.md`

The repository is the source of truth.

Before changing code:

```bash
git status
git log --oneline -10
git diff --stat
```

Inspect current auth, user role enum, routers, services, frontend AuthProvider, route guards and sidebar.

Do not rewrite working authentication.

## 1. Goal

Implement RBAC for exactly:

```text
ADMIN
SUPERVISOR
REVIEWER
```

Do NOT add:
- Candidate login role;
- custom roles;
- dynamic permission tables;
- permission editor;
- ABAC engine.

## 2. Separation of duties

```text
SUPERVISOR
→ prepares and operates monitoring

AI
→ proposes events

REVIEWER
→ verifies events/evidence

ADMIN
→ manages the system
```

SUPERVISOR must not perform final EventReview decisions.

REVIEWER must not start/pause/resume/stop monitoring.

## 3. Database

Use existing `users.role`.

Do not create:

```text
roles
permissions
role_permissions
user_permissions
```

No migration is expected if the existing enum already supports the three roles.

## 4. Central permission model

Implement centralized permission constants equivalent to:

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

Map roles exactly according to `RBAC_SPEC.md`.

ADMIN has all application permissions.

## 5. Backend authorization

Preserve existing `get_current_user` or equivalent.

Add/reuse a centralized dependency/helper:

```text
require_permission(permission)
```

Desired route pattern:

```python
@router.post(...)
def endpoint(
    ...,
    current_user = Depends(require_permission(...)),
):
    ...
```

Do not scatter raw role checks throughout routers/services.

Backend is authoritative.

## 6. Business state rules

Permission does not bypass business rules.

Examples:
- session setup remains editable only in valid states;
- EventReview is append-only;
- reviewer cannot alter AI prediction;
- evidence lock/unlock still respects appeal/history rules.

## 7. Apply to current routes only

Inspect actual implemented routes.

Protect current endpoints for:
- rooms/seats;
- candidates;
- sessions;
- session candidate assignments;
- media;
- monitoring controls;
- tracking WebSocket;
- events if implemented;
- reviews if implemented;
- evidence if implemented;
- appeals if implemented;
- users;
- settings;
- diagnostics;
- audit.

Do not create fake future features merely to attach RBAC.

## 8. Critical cases

Verify:

```text
SUPERVISOR
POST EventReview
→ 403
```

```text
REVIEWER
POST session start/pause/resume/stop
→ 403
```

```text
REVIEWER
PATCH candidate
→ 403
```

```text
SUPERVISOR
PUT seat layout
→ 403
```

```text
ADMIN
manage users
→ allowed
```

## 9. WebSocket

Protect `/ws/monitoring/{session_id}`.

User must authenticate and have:

```text
tracking.read
```

before receiving metadata.

A WebSocket connection must never create an extra AI runtime merely because a viewer connects.

## 10. Frontend authorization

Inspect current:
- AuthProvider;
- router;
- sidebar;
- page actions.

Create/reuse centralized helpers such as:

```text
usePermissions()
can(permission)
```

Avoid scattered:

```javascript
role === "ADMIN"
```

Frontend permission map must match backend.

## 11. Role-aware UI

SUPERVISOR should primarily see:

```text
Giám sát
Phiên thi
Thí sinh
Sự kiện
```

REVIEWER:

```text
Cần xác minh
Sự kiện
Thí sinh
Phiên thi
Khiếu nại
```

ADMIN:

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

Only show pages that actually exist.

Do not create fake pages/data.

## 12. Action visibility

SUPERVISOR may see monitoring controls and manual event creation.

SUPERVISOR must not see final review actions.

REVIEWER may see:

```text
Xác nhận
Không vi phạm
Cần xem thêm
```

REVIEWER must not see monitoring controls or room/candidate edit actions.

ADMIN sees permitted administrative actions subject to business rules.

## 13. Direct URL protection

Hiding menu items is insufficient.

Authenticated user opening an unauthorized route directly must see:

```text
Bạn không có quyền truy cập chức năng này.
```

Do not redirect an authenticated forbidden user to login.

## 14. API errors

Authenticated but forbidden:

```http
403
```

Use existing standardized error response, preferably:

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "You do not have permission to perform this action."
  }
}
```

Frontend maps to Vietnamese.

## 15. User management

Only ADMIN may create/update/deactivate users/change roles.

Never expose `password_hash`.

Prefer deactivate instead of hard delete.

## 16. EventReview

If already implemented:

```text
REVIEWER
ADMIN
```

may create reviews.

SUPERVISOR receives `403`.

Review records remain append-only.

Do not add destructive review update/delete behavior.

## 17. Manual Event

If already implemented or currently in scope:

```text
SUPERVISOR
ADMIN
```

may create Manual Event.

Manual Event still enters normal review workflow and is not automatically confirmed.

## 18. Evidence / Appeal

If implemented, enforce permissions in `RBAC_SPEC.md`.

Do not broaden destructive data operations while implementing RBAC.

## 19. Audit

Use existing AuditService.

Audit privileged business actions such as:
- user create/update/deactivate;
- session start/pause/resume/stop;
- manual event create;
- event review create;
- evidence lock/unlock;
- appeal create/update.

Never log password, JWT or secrets.

Do not audit tracking frames/WebSocket messages.

## 20. Tests

Add permission-map tests and API integration tests.

At minimum verify:

### SUPERVISOR
Allowed:
- room read;
- candidate operations defined by spec;
- session setup/assignment;
- media upload/read;
- monitoring control;
- event read/manual create;
- evidence read.

Forbidden:
- room/seat management;
- EventReview;
- user management;
- settings;
- full audit.

### REVIEWER
Allowed:
- room/candidate/session/event read;
- EventReview;
- evidence operations;
- appeal operations if implemented.

Forbidden:
- room/seat/candidate modification;
- monitoring control;
- user/settings management.

### ADMIN
Verify privileged operations.

Also test WebSocket authorization and direct frontend-route protection where current tooling permits.

## 21. No feature expansion

Do NOT start unrelated work:
- TSM/X3D;
- new tracking changes;
- new Seat Assignment;
- Event AI pipeline;
- evidence generator;
- report engine;
- appeal redesign.

RBAC only.

## 22. Security review

Before completion verify:
- frontend cannot elevate role;
- direct API calls enforce RBAC;
- WebSocket enforces RBAC;
- inactive user cannot bypass authorization;
- password_hash never leaves backend;
- admin routes are protected;
- frontend hiding is not the only control.

## 23. Definition of Done

RBAC passes only when:

```text
SUPERVISOR
→ can operate sessions
→ cannot review events
→ cannot manage users/settings
```

```text
REVIEWER
→ can review events
→ cannot control monitoring
→ cannot modify rooms/candidates
```

```text
ADMIN
→ can perform administrative actions
```

and:
- restricted API calls return 403;
- WebSocket is protected;
- sidebar/actions reflect role;
- direct URL is protected;
- audit records privileged actions.

## 24. Final report

Return:
1. Existing auth architecture found.
2. Existing role enum.
3. Files created.
4. Files modified.
5. Permission constants.
6. Role→permission mapping.
7. Backend authorization helper.
8. Routes protected.
9. WebSocket authorization.
10. Frontend permission helper.
11. Sidebar/action changes.
12. Forbidden route handling.
13. Audit changes.
14. Migration status.
15. Tests added.
16. Exact pytest result.
17. Ruff result.
18. mypy result.
19. Frontend test/build result.
20. Supervisor verification.
21. Reviewer verification.
22. Admin verification.
23. Known limitations.
24. Confirmation that no unrelated feature was added.

Update `CURRENT_STATE.md`.

Write:

```text
RBAC: PASS
```

only if backend and frontend authorization genuinely pass.

Otherwise:

```text
RBAC: NOT PASSED
```

with blockers.

STOP after RBAC implementation.
