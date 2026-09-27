# ExamGuard — Greenfield System Architecture

## Seat-free default identity (2026-09-24)

```text
Video → YOLO11n → ByteTrack → Logical Track Manager
   ├─ motion/spatial recovery
   └─ optional sparse appearance recovery when ambiguous
→ Stable Actor ID → Dynamic Neighbor Graph
   ├─ Single Proposal
   └─ Pair Proposal
```

Seat configuration is optional. Seat Assignment may enrich a Stable Actor with SessionCandidate context in advanced deployments, but detection, tracking, Actor ID, monitoring start, and proposal identity do not depend on Seat geometry. Runtime Actor IDs are in-memory identities and are never durable Candidate IDs.

## High level
```text
React
 ├── HTML5 Video
 ├── Canvas Overlay
 ├── Session/Candidate/Event UI
 ├── Axios REST
 └── Native WebSocket
          ↓
FastAPI
 ├── REST API
 ├── WebSocket
 ├── Auth
 ├── Business Services
 └── Monitoring Runtime
       Video Decoder → Scheduler → YOLO11n → ByteTrack → Logical Track Manager
                                                   ↓
                                       Actor Single/Dynamic Pair Proposals
                                                   ↓
                                       Bounded timestamped R3 ROI buffers
                                                   ↓
                                      Fair Action Scheduler (latest/age/budget)
                                                   ↓
                                      Shared TSM R3 → raw probabilities
          ↓
PostgreSQL + Media Storage
```

## Playback path
```text
Uploaded MP4 → media endpoint → browser HTML5 <video> → native H.264 decode
```
The player must not depend on AI inference FPS.

## AI path
```text
same MP4 → decoder → timestamp → frame sampler → YOLO11n → ByteTrack
         → Logical Track Manager → Stable Actor ID → TrackingFrame → WebSocket
         → stable proposals → R3 ROI buffers → Action Scheduler
         → background TSM R3 → ActionPrediction
```

GTX1650 target: 12.5 FPS, degrade to 10 or 8 FPS if needed. Never accumulate old frames.

## Runtime data
Recommended typed objects:
```text
FramePacket
Detection
Track
TrackingFrame
RuntimeDiagnostics
MonitoringRuntimeState
```

TrackingFrame contains session/runtime identifiers, frame/timestamp, source dimensions,
tracks, and runtime seat occupancy. Each Track contains an ephemeral track ID, normalized
bbox, confidence, and a small runtime identity result. It never contains a full Candidate.

## Realtime state
Keep tracking state in memory. Runtime components may include MonitoringRuntimeManager, VideoAnalysisWorker, LatestFrameBuffer, TrackStateStore and WebSocketPublisher.

## Optional Seat identity enrichment
```text
Track → Seat → SessionCandidate → Candidate
```
At runtime start the backend loads the session Room, active Seats, SessionCandidates and
Candidates once. The CPU-side matcher uses source-video normalized coordinates and keeps
UNASSIGNED/TENTATIVE/ASSIGNED track state plus EMPTY/OCCUPIED/GRACE seat state in memory.
Track IDs remain ephemeral; a replacement track can resolve to the same Candidate through
the same Seat. Seek resets ByteTrack and all identity state. Pause advances no identity timer
because every transition uses video timestamps rather than wall clock.

## Action-recognition runtime (Phase 6 + 6.5)
```text
Seat-Stable Identity → Proposal Builder → R3 ROI → 4-second timestamp buffer
                     → latest-ready Action Scheduler → shared TSM R3
                     → raw five-class ActionPrediction
```
Single proposals require an ASSIGNED SessionCandidate. Pair proposals require two
ASSIGNED candidates in adjacent Seats. A validated explicit 2D room-layout graph
is used when the immutable runtime context supplies one; otherwise the legacy
geometry-derived left/right-row fallback remains. Proposal IDs use
SessionCandidate UUIDs, not ephemeral Track IDs. A queue of capacity one keeps
the latest action request; ROI preparation and TSM inference run in separate
background threads, preserving the tracking worker's latest-frame path.
The scheduler keeps one latest ready clip per stable proposal, applies a video-
timestamp stride and maximum age, chooses never-predicted then oldest-predicted
proposals deterministically, and limits each dispatch to a small batch. GTX1650
uses FP32/1500-ms/B2 because measured FP16 is slower and can be non-finite on
that GPU. Pause cannot advance scheduling by wall clock; seek resets buffers,
ready/in-flight state and last-prediction timestamps; stop joins both action
threads. The R3 model registry shares one checkpoint/device/precision instance.

The raw ActionPrediction remains available for diagnostics and offline
calibration. When both action recognition and a calibrated event profile are
explicitly enabled, Phase 7 connects it to the aggregation path below. No raw
prediction is written to PostgreSQL.

The pilot calibration loader builds the explicit graph from versioned Seat
layout metadata, never from event labels. The live database context does not
yet persist that graph and therefore still uses the fallback; this is a known
deployment gap and one reason event detection remains disabled.

## Event aggregation runtime (Phase 7 implementation)

```text
ActionPrediction → causal per-class EMA → hysteresis FSM
                 → exact-actor temporal dedup → AI Event + EventActor
```

There is one in-memory FSM per stable `proposal_id + behavior`. `SINGLE` routes
only to suspicious-looking and phone/cheat-sheet; `PAIR` routes only to
communicating and exchange-object. Normal is never consumed by the Event layer.
FSM states are IDLE/CANDIDATE/ACTIVE/COOLDOWN and advance only with source-video
timestamps. Tracking timestamps expire inactive proposal state; pause provides
no timestamp and therefore cannot advance it. Seek clears every non-persisted
smoother/FSM/dedup state in the new runtime generation. Stop closes ACTIVE or
COOLDOWN state at its last valid evidence timestamp and discards an insufficient
CANDIDATE.

Finalized events are deduplicated only for the same behavior and exact
SessionCandidate actor set when their temporal IoU or configured short gap
qualifies. Persistence uses a deterministic semantic fingerprint as the event
code, making a repeated callback idempotent without adding a database unique
constraint. AI events use `source=AI`, `status=PENDING_REVIEW`, `created_by=NULL`,
and one EventActor per SessionCandidate. Pair interactions therefore create one
Event with two actors. Event creation also appends an `AI_EVENT_CREATED` audit
row. Review, Evidence, Appeal, reporting, Head Pose and object detection are not
part of this runtime.

The implementation is intentionally disabled in both profiles. S00–S04 pilot
calibration produced 20,210 real dynamic-ROI predictions, but refined Macro
Event F1 is 0.1064 and communicating has zero true-positive Events. The
checkpoint's 509-sample development training split contains no Pair-normal crop:
all 218 normal samples have one actor, whereas all communicating/exchange samples
have two. Runtime Pair geometry is therefore a learned class shortcut rather
than reliable interaction evidence. This is an upstream data/model limitation;
FSM thresholds must not be presented as a fix. `app.cli.calibrate_events`
remains the production-path collector/search/evaluator, and S05–S08 remain an
untouched final-test partition.

## Frontend synchronization
```text
video.currentTime → timestamp_ms → nearest TrackingFrame → Canvas
```
Use requestVideoFrameCallback where supported. Handle resize, fullscreen and letterboxing correctly.

## Seek
```text
clear overlay → clear metadata buffer → seek video → backend seek → reset analysis clock → wait for new metadata
```
Never show old boxes after seek.

## Diagnostics
Publish about 2 Hz: source FPS, analysis FPS, detector/tracker/pipeline latency,
dropped frames, GPU/VRAM if available, CPU and RAM. Action diagnostics add
ready/in-flight/replaced/expired requests, Single/Pair prediction rates and
interval mean/P95/max, prediction age, batch count and TSM forward latency.
Event diagnostics add CANDIDATE/ACTIVE/COOLDOWN FSM counts, created/suppressed/
deduplicated totals and per-behavior event counts.
Do not crash if NVML is unavailable.

## Docker services
Initial services:
```text
frontend
backend
postgres
```
Backend gets GPU access. Do not split detector/tracker into microservices.
