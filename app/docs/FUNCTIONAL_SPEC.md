# ExamGuard — Functional Specification

## Roles
### SUPERVISOR
Can login, create/select session, select room, upload/select video, start/pause/resume/stop monitoring, see current persons/tracks, create manual marks and send events for later review.

### REVIEWER
Can search candidates/sessions/events, inspect evidence, confirm, dismiss, mark needs-review and add notes.

### ADMIN
Can manage users, rooms, seats, runtime settings, evidence lock/export and appeal resolution.

## Room management
- create room
- rename room
- activate/deactivate
- edit description

## Seat layout
For each room, show a real camera/video frame. User can add, drag, resize, rename and delete seats. Save normalized seat regions for reuse across sessions.

## Candidate management
Fields:
- candidate_code
- full_name
- optional class_name
- optional note

Support search by code/name and later session/evidence lookup.

## Exam sessions
User can create, select room, set exam name, attach video, assign candidates to seats, start/pause/resume/stop and reopen historical sessions.

States:
```text
DRAFT
READY
RUNNING
PAUSED
COMPLETED
CANCELLED
ERROR
```

## Candidate-seat assignment
Within one session, one candidate has one seat and one seat has one candidate.

Future identity path:
```text
Track → Seat → SessionCandidate → Candidate
```

## Video upload
Current target: MP4/H.264/1080p/25 FPS.

Flow:
```text
upload → validate → ffprobe → persist metadata → serve media → native browser playback
```

## Monitoring
Backend:
```text
sample frame → YOLO11n → ByteTrack → timestamped metadata → WebSocket
```

Frontend:
```text
HTML5 video + Canvas overlay
```

Display session status, person count, Track IDs, source FPS, AI FPS, latency and AI online/offline.

## Manual mark/event
Supervisor can click `Mark` when something needs later review. Optional behavior:
- suspicious looking
- communicating
- exchange object
- phone/cheat sheet
- other
- unspecified

This creates an Event with:
```text
source = MANUAL
status = PENDING_REVIEW
```
Do not create a separate Bookmark table.

## Automated event — future
TSM creates:
```text
source = AI
status = PENDING_REVIEW
```
AI does not directly confirm misconduct.

## Event actors
One event may involve one or many SessionCandidates. Interaction events should not be duplicated per actor.

## Review
Reviewer sees type, time, actors, seats, AI confidence, evidence, context and history. Actions:
```text
Confirm
Dismiss
Needs Review
```
Every action appends a review row.

## Evidence
May include:
- context video
- focused video
- snapshot
- report

Recommended default evidence window:
```text
5 sec before + event + 5 sec after
```
Keep full context because ROI-only evidence may be misleading during appeal.

## Evidence lock
Locked evidence must be excluded from future automated cleanup.

## Candidate history
Search candidate by code/name and show sessions, seat per session, events, review status and evidence availability.

## Appeal
Workflow:
```text
Candidate → Session → Event → Evidence → Appeal
```
States:
```text
OPEN
UNDER_REVIEW
UPHELD
OVERTURNED
CLOSED
```
An appeal may reference multiple events. Related evidence should be lockable.

## Audit trail
Record session start/stop, event create/open/confirm/dismiss, evidence lock/export, appeal create/resolve, room updates and seat-layout updates.

## Search
Quick search: candidate code/name, session code, event code.

Event filters later: date, session, room, candidate, seat, behavior, status, reviewer.

## Reports
Later session report: duration, candidates, alerts, reviewed, confirmed, dismissed, pending, per-class counts, timeline and event list.
