# AGENTS.md — ExamGuard Greenfield Rebuild Rules

## Mission
Build ExamGuard from a clean application architecture for the thesis **“Nghiên cứu và xây dựng hệ thống phát hiện hành vi bất thường trong phòng thi sử dụng camera giám sát”.**

This is a **greenfield rebuild** of the application and database. Legacy code may be inspected only for working infrastructure such as Docker, CUDA, PostgreSQL and environment configuration.

## Non-negotiable rules
- Use Python 3.11, FastAPI, React, PostgreSQL, SQLAlchemy 2.x and Alembic.
- Keep Docker Compose, NVIDIA Container Toolkit and CUDA support for the backend.
- Use native HTML5 `<video>` for uploaded MP4 playback.
- Use native WebSocket for realtime metadata and Canvas for overlays.
- Initial detector: YOLO11n, COCO pretrained, person-only.
- Initial tracker: ByteTrack.
- Use timestamp synchronization and bounded/latest-frame semantics.
- Drop stale analysis frames instead of accumulating latency.
- Realtime tracking state stays in memory, not PostgreSQL.
- AI findings are suspicious events, not final misconduct conclusions.
- Audit history is append-only.

## Must not
- Do not copy legacy SQLAlchemy models, Alembic migrations, service structure, API structure or frontend page architecture.
- Do not create compatibility tables for the legacy database.
- Do not create names like `_v2`, `_new`, `_final` or `/api/v2`.
- Do not create per-frame `detections`, `tracking_frames` or `bounding_boxes` tables.
- Do not use Track ID as permanent candidate identity.
- Do not use MJPEG as the primary uploaded-video player.
- Do not add Redis, Celery, Kafka, DeepStream, TensorRT, MediaMTX, WebRTC, pose estimation or ReID-heavy tracking in the current milestone.

## Core business model
```text
Room → Seat → ExamSession → SessionCandidate → Event → Review → Evidence → Appeal
```

Runtime CV chain:
```text
Video → YOLO11n → ByteTrack → Track → later Seat Assignment → later Candidate Context → later TSM → Event
```

## Core persistent entities
- User
- Room
- Seat
- Candidate
- MediaAsset
- ExamSession
- SessionCandidate
- Event
- EventActor
- EventReview
- EvidenceAsset
- AppealCase
- AppealEvent
- AuditLog

Do not add persistent entities without a concrete use case.

## AI defaults
YOLO11n:
```yaml
imgsz: 640
conf: 0.10
iou: 0.70
classes: [0]
device: cuda:0
half: true
```

ByteTrack:
```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 20
match_thresh: 0.80
fuse_score: true
```

## Hardware targets
Current:
```text
i7-11800H / 16 GB RAM / GTX 1650 Max-Q
Video playback: native 25 FPS
AI analysis target: 8–12.5 FPS
No accumulated lag
```

Future:
```text
16 GB RAM / RTX 3060 12 GB
AI analysis target: about 15–20 FPS
```

Keep GPU headroom for later TSM inference.

## UX rules
Normal operator workflow:
```text
Login → Create/select session → Select room → Upload/select video → Assign candidates to seats → Start monitoring
```

Primary navigation:
```text
Monitoring
Sessions
Events
Candidates
Reports
Settings
```

Do not expose YOLO, ByteTrack, CUDA, ROI or TSM as primary user navigation.

## Event semantics
Normal event lifecycle:
```text
PENDING_REVIEW → CONFIRMED
               → DISMISSED
```
Additional states:
```text
NEEDS_REVIEW
DISPUTED
RESOLVED
```
Never store `candidate_is_cheater = true`.

## Evidence
Evidence may include context video, focused ROI video, snapshot and exported report. Recommended default context is 5 seconds before and 5 seconds after the event interval. Evidence may be locked during investigation/appeal.

## Database rules
- PostgreSQL stores durable business data.
- UUID primary keys.
- TIMESTAMPTZ timestamps.
- Database-level FKs and unique constraints.
- Normalized seat geometry in [0,1].
- All schema changes through Alembic.
- Initial migration is `initial_schema`.
- No legacy migrations.

## Testing
At minimum test auth, room/seat constraints, candidate CRUD, session lifecycle, session candidate uniqueness, video metadata parsing, event creation, event actors, event reviews, evidence locking, appeals, audit logging, config validation, bbox normalization, stale-frame dropping, WebSocket schemas and seek/reset behavior.

Core tests must not require GPU.

## Implementation order
1. Clean skeleton.
2. Clean DB + Alembic baseline.
3. Auth.
4. Rooms + seats.
5. Candidates.
6. Sessions + candidate-seat assignment.
7. Video upload + ffprobe + native playback.
8. YOLO11n.
9. ByteTrack.
10. Realtime scheduler.
11. WebSocket.
12. Canvas overlay.
13. Monitoring UI.
14. Manual event/bookmark.
15. Review/evidence/audit.
16. Appeals/reports.
17. Seat assignment runtime.
18. TSM R3.
19. Automated abnormal events.
