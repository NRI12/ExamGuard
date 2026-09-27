# ExamGuard — API Contract

Prefix: `/api/v1`

## Pagination
List endpoints use:
```text
?page=1&page_size=20&q=...
```
with `page_size` limited to 100. Paginated responses use:
```json
{"items": [], "page": 1, "page_size": 20, "total": 0}
```

Application errors use a stable code and may identify the affected field:
```json
{"error":{"code":"CANDIDATE_CODE_EXISTS","message":"Candidate code already exists","field":"candidate_code","details":{}}}
```

## Auth
```text
POST /api/v1/auth/login
GET  /api/v1/auth/me
```

## Rooms
```text
GET   /api/v1/rooms?page=1&page_size=20&q=...
POST  /api/v1/rooms
GET   /api/v1/rooms/{room_id}
PATCH /api/v1/rooms/{room_id}
```

## Seats
```text
GET /api/v1/rooms/{room_id}/seats
PUT /api/v1/rooms/{room_id}/seats
```
PUT validates unique codes and normalized geometry.

## Candidates
```text
GET   /api/v1/candidates?page=1&page_size=20&q=...
POST  /api/v1/candidates
GET   /api/v1/candidates/{candidate_id}
PATCH /api/v1/candidates/{candidate_id}
GET   /api/v1/candidates/{candidate_id}/sessions
GET   /api/v1/candidates/{candidate_id}/events
```

## Media
```text
POST /api/v1/media/videos
GET  /api/v1/media/{media_id}
GET  /api/v1/media/{media_id}/content
GET  /api/v1/media/{media_id}/frame?timestamp_ms=5000
```
Upload response should include filename, media URL, width, height, FPS, duration, codec and size. Media endpoint should support browser seeking/range requests when feasible.
The frame endpoint returns one authenticated `image/jpeg` reference frame for seat calibration;
it is not a realtime stream.

## Sessions
```text
GET   /api/v1/sessions?page=1&page_size=20&q=...&status=READY&room_id=uuid
POST  /api/v1/sessions
GET   /api/v1/sessions/{session_id}
PATCH /api/v1/sessions/{session_id}
PUT   /api/v1/sessions/{session_id}/candidates
```
Candidate assignment must reject duplicate candidates/seats and seats from another room.
Sessions may be edited only before monitoring starts. `DRAFT` or `READY` sessions may be
cancelled with `PATCH {"status":"CANCELLED"}`; cancelled and historical sessions remain read-only.

## Monitoring lifecycle
```text
POST /api/v1/sessions/{session_id}/start
POST /api/v1/sessions/{session_id}/pause
POST /api/v1/sessions/{session_id}/resume
POST /api/v1/sessions/{session_id}/stop
POST /api/v1/sessions/{session_id}/seek
GET  /api/v1/sessions/{session_id}/monitoring-status
```
Seek request:
```json
{"timestamp_ms": 53240}
```

## WebSocket
```text
/ws/monitoring/{session_id}
```
Tracking message:
```json
{
  "type": "tracking",
  "session_id": "uuid",
  "runtime_instance_id": "uuid",
  "runtime_generation": 1,
  "tracker_instance_id": "uuid",
  "tracking_seq": 138,
  "timestamp_ms": 53240,
  "frame_id": 1331,
  "source_width": 1920,
  "source_height": 1080,
  "tracks": [{
    "track_id": 17,
    "actor_id": "A0007",
    "actor_state": "ACTIVE",
    "recovered": false,
    "bbox_norm": [0.214,0.182,0.326,0.784],
    "confidence": 0.91,
    "identity": {
      "state": "ASSIGNED",
      "seat_id": "uuid",
      "seat_code": "B03",
      "session_candidate_id": "uuid",
      "score": 0.82
    }
  }],
  "seats": [{
    "seat_id": "uuid",
    "seat_code": "B03",
    "session_candidate_id": "uuid",
    "state": "OCCUPIED",
    "track_id": 17
  }]
}
```
Clients accept only increasing `tracking_seq` values for the active runtime instance/generation.
Seek increments the runtime generation; runtime restart changes the runtime instance identifier.
`actor_id` is the primary runtime identity; raw `track_id` is diagnostic and Seat/Candidate fields may be null. Tracking payloads do not include full Candidate objects. The frontend joins
`session_candidate_id` with the ordinary session REST response. Raw detections are available
only in development logs and benchmark CSV output.
Diagnostics additionally include active/lost logical actors, raw track count, motion/appearance recoveries, appearance request/batch/latency counters, logical tracking latency, dynamic pair count, and optional seat assignment latency, assigned/tentative/unassigned track
counts, occupied/grace/empty seat counts, seat switches, and identity recoveries.

Phase 6 publishes a separate low-frequency raw-action message (no Event write):
```json
{
  "type": "action_prediction",
  "session_id": "uuid",
  "runtime_instance_id": "uuid",
  "runtime_generation": 1,
  "timestamp_ms": 53240,
  "predictions": [{
    "proposal_id": "single:session-candidate-uuid",
    "proposal_type": "SINGLE",
    "session_candidate_ids": ["uuid"],
    "seat_codes": ["B03"],
    "timestamp_ms": 53240,
    "class_probabilities": {
      "normal": 0.10,
      "suspicious_looking": 0.72,
      "communicating": 0.04,
      "exchange_object": 0.02,
      "using_phone/cheat_sheet": 0.12
    },
    "predicted_class": "suspicious_looking",
    "confidence": 0.72,
    "model_name": "r3_tsm_r50_k400_diff_final"
  }]
}
```
The same message supports `PAIR` with two canonical SessionCandidate IDs and
Seat codes. `action_error` reports an inference failure without altering the
tracking lifecycle. Diagnostics carry proposal counts, action buffer/queue
depth, stale/expired and latest-replacement counts, scheduler ready/in-flight
counts, Single/Pair prediction rates and interval mean/P95/max, prediction age,
device, batch count, and mean/P95 preprocess/inference/forward/pipeline latency.
WebSocket authorization remains `tracking.read`.

## Manual events
```text
POST /api/v1/events/manual
```
Request may include session_id, timestamp_ms, optional behavior_type and optional session_candidate_ids.

## Events
```text
GET /api/v1/events
GET /api/v1/events/{event_id}
POST /api/v1/events/{event_id}/reviews
```
Filters: session, candidate, status, behavior, date.

## Evidence
```text
GET  /api/v1/events/{event_id}/evidence
POST /api/v1/events/{event_id}/evidence/lock
POST /api/v1/events/{event_id}/evidence/unlock
```

## Appeals
```text
POST  /api/v1/appeals
GET   /api/v1/appeals/{appeal_id}
PATCH /api/v1/appeals/{appeal_id}
```

## Search later
```text
GET /api/v1/search?q=SV103
```
Return typed results for candidate/session/event.
