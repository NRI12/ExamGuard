# CODEX PROMPT — Rebuild ExamGuard from Scratch

You are rebuilding ExamGuard from a clean architecture. This is not a refactor of the legacy application.

Read these files first, in this exact order:
1. `AGENTS.md`
2. `PROJECT_CONTEXT.md`
3. `FUNCTIONAL_SPEC.md`
4. `DATABASE_SCHEMA.md`
5. `SYSTEM_ARCHITECTURE.md`
6. `UI_UX_SPEC.md`
7. `API_CONTRACT.md`
8. `CONFIG_SPEC.md`
9. `IMPLEMENTATION_PLAN.md`

Then inspect the current repository.

## Primary goal
Build a clean system with:
```text
Room → Seat → ExamSession → SessionCandidate → Event → Review → Evidence → Appeal
```
and current CV pipeline:
```text
Uploaded MP4 → native React playback → backend analysis → YOLO11n person-only → ByteTrack → timestamped WebSocket metadata → Canvas overlay
```

Hardware targets:
```text
Current: i7-11800H / 16 GB RAM / GTX 1650 Max-Q
Future: 16 GB RAM / RTX 3060 12 GB
```
Same codebase; runtime config changes only.

## Greenfield requirement
The repository contains legacy application/database code. Do not use it as the implementation base.

You may inspect legacy code only for:
- Docker
- Docker Compose
- NVIDIA Container Toolkit
- CUDA/PyTorch setup
- PostgreSQL deployment
- frontend build/dependencies
- environment variables

Do not copy legacy SQLAlchemy models, migrations, service structure, API routes, database table names or frontend page architecture.

Do not create compatibility code unless explicitly required.

Do not create `_v2`, `_new`, `_final` or `/api/v2`. Use `/api/v1` as the clean baseline.

## Stack
Preserve the current stack: Python 3.11, FFmpeg, Docker, Docker Compose, NVIDIA Container Toolkit, CUDA runtime, FastAPI, Uvicorn, Pydantic, pydantic-settings, python-dotenv, SQLAlchemy, Alembic, psycopg[binary], PyJWT, pwdlib[argon2], python-multipart, torch, torchvision, NumPy, opencv-python-headless, PyYAML, Pillow, pytest, httpx, ruff, mypy, React, react-dom, react-is, react-router-dom, axios, recharts.

Add only `ultralytics` for this phase. Use native WebSocket; do not add Socket.IO.

## Database
Create clean Alembic baseline `initial_schema` with these core tables:
```text
users
rooms
seats
candidates
media_assets
exam_sessions
session_candidates
events
event_actors
event_reviews
evidence_assets
appeal_cases
appeal_events
audit_logs
```
Do not create per-frame tracking tables. Do not import legacy migrations. Do not delete existing legacy data automatically; document a clean dev database/volume procedure.

## User workflows
Normal operation:
```text
Login → select/create session → select room → upload/select video → assign candidates to seats → start monitoring
```
Event review:
```text
AI/manual event → evidence → confirm/dismiss/needs-review
```
Appeal:
```text
search candidate → session → event → evidence → appeal
```
The user should not manually search a 90-minute video for a timestamp.

## Monitoring architecture
Use browser native HTML5 `<video>` as the primary player. Do not use server-generated MJPEG as the primary player.

Backend independently analyzes the same file:
```text
YOLO11n person-only
imgsz 640
conf 0.10
iou 0.70
FP16 CUDA
→ ByteTrack
```
GTX1650 initial analysis target 12.5 FPS; degrade to 10 or 8 FPS if needed.

Critical rule: never accumulate analysis backlog. Use latest-frame/drop-stale semantics.

## Frontend
Primary navigation:
```text
Monitoring
Sessions
Events
Candidates
Reports
Settings
```
Live Monitoring: 72–75% video, 25–28% contextual panel. Use HTML5 video + Canvas + requestVideoFrameCallback + WebSocket metadata. Overlay must remain aligned after resize/fullscreen/seek. High-frequency tracking metadata must not rerender the full React tree.

## Event semantics
AI findings are suspicious events, not final conclusions. Never implement candidate risk score or `candidate_is_cheater`. Preserve AI output and human review separately.

## Audit
Important actions must go through a centralized AuditService.

## Required first response — before coding
Return:
1. current repository assessment
2. infrastructure to preserve
3. legacy application areas to ignore
4. proposed new directory tree
5. new database implementation summary
6. implementation phases
7. risks/assumptions

Do not modify files in that assessment step.

## Implementation after assessment
Implement incrementally. Do not generate the entire system in one giant change.

Start with:
```text
clean backend/frontend skeleton
clean settings
clean SQLAlchemy models
clean Alembic baseline
auth
rooms
seats
candidates
exam sessions
session candidate assignment
```

Then continue according to `IMPLEMENTATION_PLAN.md`.

After each phase:
```text
run migrations
run tests
run ruff
run mypy where configured
run frontend build when relevant
```
Fix failures before continuing.

## Definition of done — first major milestone
- clean DB created from empty PostgreSQL
- no legacy tables required
- auth works
- room/seat management works
- candidate management works
- exam sessions work
- candidate-seat assignment works
- MP4 upload works
- ffprobe metadata works
- browser playback/seek works
- YOLO detects persons
- ByteTrack produces track IDs
- WebSocket sends timestamped metadata
- Canvas overlay aligns after resize/fullscreen/seek
- analysis lag does not accumulate
- GTX1650 profile works
- RTX3060 profile exists
- tests/lint/type checks pass for touched code

Do not implement TSM/action recognition until explicitly requested.
