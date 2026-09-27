# ExamGuard — Greenfield Implementation Plan

## Phase 0 — Repository assessment
Before changing code:
1. print repository tree
2. identify working Docker/CUDA infrastructure
3. identify PostgreSQL setup
4. identify frontend build setup
5. identify legacy app/database areas to ignore
6. modify nothing yet

Output infrastructure to preserve, legacy areas to replace, risks and assumptions.

## Phase 1 — Clean skeleton + database
Create clean backend/frontend skeleton, settings, DB session, SQLAlchemy models, Alembic `initial_schema`, health endpoint and auth.

Acceptance: migrate from empty DB, backend starts, login works, no legacy schema required.

## Phase 2 — Rooms / seats / candidates / sessions
Implement CRUD and candidate-seat assignment with constraints.

## Phase 3 — Video upload
Implement upload, validation, ffprobe, metadata, media serving and native browser playback. Verify seek works before AI.

## Phase 4 — Monitoring frontend
Implement clean dark Monitoring page, session/room selection, assignment summary, player, controls and fullscreen.

## Phase 5 — YOLO11n
Add clean person-only detector wrapper, runtime config and latency metrics.

## Phase 6 — ByteTrack
Add tracker wrapper, normalized boxes and timestamped TrackingFrame. Do not persist frames.

## Phase 7 — Analysis runtime
Session-bound worker, target AI FPS, latest-frame semantics, pause/resume/seek/stop, cleanup and diagnostics. No backlog.

## Phase 8 — WebSocket + Canvas
Backend tracking/diagnostics/state messages. Frontend metadata buffer, Canvas overlay, requestVideoFrameCallback, letterbox/resize/fullscreen/seek correctness.

## Phase 9 — Manual Event
Add Mark button, manual event creation, optional behavior/actor association and audit entry.

## Phase 10 — Event Review
Event list/detail, confirm/dismiss/needs-review, review history and audit.

## Phase 11 — Evidence
Context clip/snapshot architecture, lock/unlock and evidence list.

## Phase 12 — Candidate History + Appeals
Candidate histories, appeal creation, related events, evidence locks, resolution and audit.

## Phase 13 — Reports
Only after real event data exists.

## Phase 14 — Seat runtime
Connect Track → Seat → SessionCandidate after tracking quality is acceptable.

## Phase 15 — TSM R3
Later integrate proposals → ROI → TSM → smoothing → Event.

Do not mix TSM into the initial rebuild.
