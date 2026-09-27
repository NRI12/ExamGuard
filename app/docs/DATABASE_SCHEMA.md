# ExamGuard — Database Schema

## Principles
PostgreSQL stores durable business data. Do not store every frame, detection, track update or bbox. Use UUID primary keys, TIMESTAMPTZ and database-level constraints.

## ER overview
```text
users

rooms
 └── seats

candidates
media_assets

exam_sessions
 ├── room
 ├── media_asset
 └── session_candidates
      ├── candidate
      └── seat

events
 ├── exam_session
 ├── event_actors → session_candidate
 ├── event_reviews
 └── evidence_assets

appeal_cases
 ├── session_candidate
 └── appeal_events → event

audit_logs
```

## users
```text
id UUID PK
username VARCHAR UNIQUE NOT NULL
password_hash VARCHAR NOT NULL
full_name VARCHAR NULL
role VARCHAR NOT NULL
is_active BOOLEAN NOT NULL DEFAULT true
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```
Roles: SUPERVISOR, REVIEWER, ADMIN.

## rooms
```text
id UUID PK
code VARCHAR UNIQUE NOT NULL
name VARCHAR NOT NULL
description TEXT NULL
is_active BOOLEAN NOT NULL DEFAULT true
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```

## seats
```text
id UUID PK
room_id UUID NOT NULL FK rooms.id
code VARCHAR NOT NULL
x DOUBLE PRECISION NOT NULL
y DOUBLE PRECISION NOT NULL
width DOUBLE PRECISION NOT NULL
height DOUBLE PRECISION NOT NULL
sort_order INTEGER NULL
is_active BOOLEAN NOT NULL DEFAULT true
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```
Constraints:
```text
UNIQUE(room_id, code)
0 <= x <= 1
0 <= y <= 1
0 < width <= 1
0 < height <= 1
x + width <= 1
y + height <= 1
```

## candidates
```text
id UUID PK
candidate_code VARCHAR UNIQUE NOT NULL
full_name VARCHAR NOT NULL
class_name VARCHAR NULL
note TEXT NULL
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```

## media_assets
```text
id UUID PK
original_filename VARCHAR NOT NULL
stored_filename VARCHAR NOT NULL
storage_path VARCHAR NOT NULL
mime_type VARCHAR NULL
codec VARCHAR NULL
width INTEGER NULL
height INTEGER NULL
fps DOUBLE PRECISION NULL
duration_ms BIGINT NULL
size_bytes BIGINT NULL
sha256 VARCHAR NULL
created_by UUID NULL FK users.id
created_at TIMESTAMPTZ NOT NULL
```

## exam_sessions
```text
id UUID PK
session_code VARCHAR UNIQUE NOT NULL
exam_name VARCHAR NOT NULL
room_id UUID NOT NULL FK rooms.id
video_asset_id UUID NULL FK media_assets.id
status VARCHAR NOT NULL
scheduled_start TIMESTAMPTZ NULL
scheduled_end TIMESTAMPTZ NULL
actual_start TIMESTAMPTZ NULL
actual_end TIMESTAMPTZ NULL
runtime_profile VARCHAR NULL
created_by UUID NOT NULL FK users.id
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```
Status: DRAFT, READY, RUNNING, PAUSED, COMPLETED, CANCELLED, ERROR.

## session_candidates
```text
id UUID PK
session_id UUID NOT NULL FK exam_sessions.id
candidate_id UUID NOT NULL FK candidates.id
seat_id UUID NOT NULL FK seats.id
created_at TIMESTAMPTZ NOT NULL
```
Constraints:
```text
UNIQUE(session_id, candidate_id)
UNIQUE(session_id, seat_id)
```
Business validation: seat must belong to the same room as the session.

## events
```text
id UUID PK
event_code VARCHAR UNIQUE NOT NULL
session_id UUID NOT NULL FK exam_sessions.id
source VARCHAR NOT NULL
behavior_type VARCHAR NULL
start_ms BIGINT NOT NULL
end_ms BIGINT NOT NULL
peak_ms BIGINT NULL
ai_confidence DOUBLE PRECISION NULL
status VARCHAR NOT NULL
created_by UUID NULL FK users.id
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
```
Source: AI, MANUAL.
Behavior: SUSPICIOUS_LOOKING, COMMUNICATING, EXCHANGE_OBJECT, USING_PHONE_CHEAT_SHEET, OTHER.
Status: PENDING_REVIEW, CONFIRMED, DISMISSED, NEEDS_REVIEW, DISPUTED, RESOLVED.
Constraints:
```text
start_ms >= 0
end_ms >= start_ms
peak_ms is null or start_ms <= peak_ms <= end_ms
0 <= ai_confidence <= 1 when not null
```
Indexes:
```text
(session_id, start_ms)
(session_id, status)
(session_id, behavior_type)
```

## event_actors
```text
id UUID PK
event_id UUID NOT NULL FK events.id
session_candidate_id UUID NOT NULL FK session_candidates.id
role VARCHAR NULL
created_at TIMESTAMPTZ NOT NULL
```
Constraint: UNIQUE(event_id, session_candidate_id).

## event_reviews
```text
id UUID PK
event_id UUID NOT NULL FK events.id
reviewer_id UUID NOT NULL FK users.id
decision VARCHAR NOT NULL
note TEXT NULL
created_at TIMESTAMPTZ NOT NULL
```
Decision: CONFIRM, DISMISS, NEEDS_REVIEW. Keep history; do not overwrite rows.

## evidence_assets
```text
id UUID PK
event_id UUID NOT NULL FK events.id
kind VARCHAR NOT NULL
storage_path VARCHAR NOT NULL
mime_type VARCHAR NULL
start_ms BIGINT NULL
end_ms BIGINT NULL
sha256 VARCHAR NULL
locked BOOLEAN NOT NULL DEFAULT false
created_at TIMESTAMPTZ NOT NULL
```
Kind: CONTEXT_VIDEO, FOCUSED_VIDEO, SNAPSHOT, REPORT.

## appeal_cases
```text
id UUID PK
case_code VARCHAR UNIQUE NOT NULL
session_candidate_id UUID NOT NULL FK session_candidates.id
status VARCHAR NOT NULL
description TEXT NULL
resolution TEXT NULL
created_by UUID NOT NULL FK users.id
resolved_by UUID NULL FK users.id
created_at TIMESTAMPTZ NOT NULL
updated_at TIMESTAMPTZ NOT NULL
resolved_at TIMESTAMPTZ NULL
```
Status: OPEN, UNDER_REVIEW, UPHELD, OVERTURNED, CLOSED.

## appeal_events
```text
appeal_id UUID NOT NULL FK appeal_cases.id
event_id UUID NOT NULL FK events.id
PRIMARY KEY (appeal_id, event_id)
```

## audit_logs
```text
id UUID PK
actor_user_id UUID NULL FK users.id
action VARCHAR NOT NULL
entity_type VARCHAR NOT NULL
entity_id UUID NULL
metadata JSONB NULL
created_at TIMESTAMPTZ NOT NULL
```
Indexes:
```text
(entity_type, entity_id)
(actor_user_id, created_at)
created_at
action
```
Audit logs are append-only.

## Do not create initially
```text
detections
tracking_frames
bounding_boxes
model_predictions
notifications
jobs
queues
camera_health
candidate_risk_scores
```

## Alembic
Create one clean baseline migration named `initial_schema`. Do not import legacy migrations or rename legacy tables. Do not delete legacy volumes automatically.
