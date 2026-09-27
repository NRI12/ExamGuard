# ExamGuard Seat-Stable Identity Report

Updated: 2026-09-22

```text
SEAT-STABLE IDENTITY: NOT PASSED
IMPLEMENTATION/UNIT/INTEGRATION GATE: PASSED
```

The implementation is complete through `Track → Seat → SessionCandidate → Candidate` and
stops before Proposal Builder. The overall acceptance gate remains `NOT PASSED` because the
available independent videos contain neither a supervisor walking through the aisle nor a
complete stand/leave/move sequence, and the persisted H9302 layout has not yet been recalibrated
by an operator against the actual selected camera view.

## Existing architecture and integration

The repository already had one monitoring runtime and one retained ByteTrack instance per
session, ordered video timestamps, a one-slot latest-frame buffer, seek generations, tracker
reset on seek, pause/resume using an analysis clock, native WebSocket delivery and one Canvas
render loop. Seat identity is added after normalized ByteTrack bboxes without rewriting that
tracking lifecycle.

At monitoring start, one query loads the session's active Seats and optional
SessionCandidate/Candidate bindings into immutable non-ORM runtime types. The per-frame path is
CPU-only and does not query or write PostgreSQL. No Alembic migration or persistent tracking
entity was added.

```text
Video → YOLO11n → ByteTrack → normalized Track bbox
      → Seat matcher → temporal assignment → Seat-Stable Identity
      → TrackingFrame/WebSocket

Seat-Stable Identity → Proposal Builder [future]
```

## Geometry and scoring

For normalized track rectangle `t` and stored Seat rectangle `s`, expansion ratio `r` expands
each side by `r × width(s)` and `r × height(s)`, then clamps the result to `[0,1]`. Stored Seat
geometry is never changed.

```text
overlap(t, s') = area(t ∩ s') / area(t)
```

Invalid, zero-area, non-finite and out-of-frame rectangles safely produce a score of zero after
clipping.

Using track center `c_t`, expanded-seat center `c_s'`, and expanded-seat diagonal
`diag(s')`:

```text
d = ||c_t - c_s'|| / diag(s')
distanceScore = clamp(1 - d, 0, 1)
```

The final score is applied exactly as configured; weights are not normalized:

```text
score = 0.70 × overlap + 0.30 × distanceScore
```

Pairs below `min_score=0.35` are discarded. Remaining pairs are sorted descending by score and
overlap and greedily accepted only while both Track and Seat are free. Existing confirmed
assignments reserve their Seat before new proposals are considered.

## Runtime configuration

```yaml
seat_assignment:
  enabled: true
  overlap_weight: 0.70
  distance_weight: 0.30
  min_score: 0.35
  seat_expand_ratio: 0.08
  confirm_ms: 600
  release_ms: 1500
  switch_margin: 0.15
  switch_confirm_ms: 800
```

Weights must be non-negative with a positive sum. Score threshold is in `[0,1]`; expansion,
durations and switch margin must be non-negative.

## State machines

Track assignment:

```text
UNASSIGNED
  → plausible Seat
TENTATIVE(Seat)
  → same plausible Seat for confirm_ms of video time
ASSIGNED(Seat)
  → evidence absent for release_ms
UNASSIGNED
```

A confirmed switch requires the new score to exceed the current score by more than
`switch_margin` continuously for `switch_confirm_ms`. A single adjacent-seat score spike does
not switch identity.

Seat occupancy:

```text
EMPTY → OCCUPIED → GRACE → EMPTY
                   ↘ OCCUPIED (track/geometry recovers)
```

When an assigned track disappears, the Seat stays in GRACE. A new track near that Seat becomes
TENTATIVE and then resolves through the same SessionCandidate after confirmation. The old Track
ID is not restored. If a visible candidate walks away and remains below the gate for
`release_ms`, that track becomes UNASSIGNED while the planned database seat assignment remains
unchanged. An extra person can remain UNASSIGNED; identity is never forced to the nearest Seat.

All transition durations use analysis/video timestamps. Pause produces no analysis frames and
cannot advance confirmation. Seek increments the runtime generation and resets ByteTrack,
track-seat state, seat state, tentative/switch timers and identity counters. Stop destroys the
worker-local assignment engine; a restart constructs a new one.

## API and frontend

Tracking WebSocket items retain `track_id`, `bbox_norm` and `confidence` and add:

```json
{
  "identity": {
    "state": "ASSIGNED",
    "seat_id": "uuid",
    "seat_code": "B03",
    "session_candidate_id": "uuid",
    "score": 0.82
  }
}
```

Each TrackingFrame also has compact runtime Seat snapshots (`EMPTY`, `OCCUPIED`, `GRACE`). Full
Candidate objects are not sent at AI FPS. The frontend joins `session_candidate_id` with the
existing session assignments loaded over REST.

Normal Canvas labels show `B03 • SV103`, `Đang xác định…`, or no identity label. Track IDs,
scores and confidence are development/diagnostic-only. The side panel is candidate/seat-centric
and includes tracked, grace, empty, tentative, and unidentified counts. Existing RBAC remains
authoritative: REVIEWER receives no monitoring control buttons.

Seat calibration can load an authenticated JPEG from:

```text
GET /api/v1/media/{media_id}/frame?timestamp_ms=5000
```

or use a locally selected MP4. Seat rectangles are drawn over the real source image, never a
blank white calibration canvas. The endpoint decodes one frame and is not streaming.

## Diagnostics

Runtime diagnostics add:

```text
seat_assignment_ms
assigned_tracks / tentative_tracks / unassigned_tracks
occupied_seats / grace_seats / empty_seats
seat_switches / identity_recoveries
```

## Automated verification

- Backend pytest: `85 passed, 1 skipped`; the skipped test explicitly requires CUDA.
- Ruff: passed.
- mypy: `Success: no issues found in 91 source files`.
- Frontend Vitest: `10 passed` files, `28 passed` tests.
- Frontend TypeScript and production Vite build: passed, 124 modules transformed.

Unit/integration coverage includes clipping/intersection/expansion/centers, overlap, distance,
combined score, clear/far/adjacent/conflict matching, tentative confirmation, wall-clock pause,
supervisor/extra-person UNASSIGNED behavior, adjacent jitter, genuine switch, fragmentation
handoff, candidate leaving, release/grace, reset/restart, worker seek reset, WebSocket schema,
identity context loading and media-frame authorization.

## Real-video validation

The production detector, tracker and assignment engine were run unchanged on three independent
60-second clips at 5 analysis FPS on CPU. Human-calibrated evaluation regions are stored in
`docs/seat_identity_validation/manifest.yaml`. The set covers stable seating, leaning/turning,
adjacent people, partial occlusion, tracker fragmentation and a seated extra person outside the
six candidate Seats.

Aggregate measured result over 903 analyzed frames:

| Metric | Result |
|---|---:|
| Expected candidate actor-samples | 5,418 |
| Correct Seat Assignment | 5,155 / 5,418 = 95.146% |
| Wrong Seat rate | 0 / 5,763 = 0.000% |
| Unassigned candidate rate | 263 / 5,418 = 4.854% |
| Extra/duplicate track samples | 345 |
| False identity on extra tracks | 0 |
| False Seat switches | 0 |
| Mean/P95 stabilization delay | 711.1 / 600.0 ms |
| Track-change recovery heuristic | 32 / 59 = 54.24% |
| In-GRACE engine identity recoveries | 2 |
| Mean/P95 observed recovery delay | 112.5 / 600.0 ms |
| `seat_assignment_ms` mean/P95 | 0.667 / 0.770 ms |

P95 uses the validation CLI's rounded-index percentile implementation; one clip has a 2,600 ms startup
outlier, while 17 of 18 initial candidate stabilizations are 600 ms. “Track-change recovery” is
an evaluation-region heuristic and is reported separately from the engine's stricter GRACE
handoff counter. No formal MOT/ReID metric is claimed.

## Known limitations and blockers

- No available independent clip contains a supervisor walking through the aisle.
- No available independent clip contains a complete stand/leave/move/sit sequence, so real-video
  release behavior is covered only by deterministic runtime tests.
- Current stored H9302 Seat rectangles were created before real-frame calibration and do not
  align consistently with the different uploaded camera views. They must be recalibrated through
  the corrected UI; this implementation deliberately does not rewrite user data automatically.
- Room-level Seat geometry assumes a stable camera viewpoint. Moving the camera between sessions
  requires recalibration; per-camera calibration is not a persistent entity in the current model.
- Long occlusion beyond `release_ms` intentionally causes temporary UNASSIGNED state and requires
  a fresh `confirm_ms` before identity returns. This conservative behavior avoids wrong identity.
- GPU/VRAM and end-to-end browser-to-CUDA performance were not measured because host NVIDIA
  driver access is unavailable.

No TSM, action recognition, pair ROI, behavior classification, Event, Evidence, Appeal or Report
functionality was added.
