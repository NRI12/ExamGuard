# Seat-Free Stable Tracking Context

Updated: 2026-09-24

## Goal

The normal monitoring path must start from a video and an AI-ready runtime. A Seat layout and Candidate mapping are optional deployment metadata, not prerequisites for person detection, tracking, stable runtime identity, or action proposals.

```text
Video → YOLO11n → ByteTrack → LogicalTrackManager
      → motion/spatial recovery
      → optional sparse appearance recovery when ambiguous
      → Stable Actor ID → Dynamic Neighbor Graph
      → Single/Pair proposals
```

The legacy Seat path remains available for configured rooms:

```text
Stable Actor → optional Seat Assignment → optional SessionCandidate context
```

It enriches an actor but does not create the actor and cannot block monitoring.

## Existing dependency found

Before this phase, `SessionReadiness.can_mark_ready` required at least one active Seat and one SessionCandidate, monitoring start reused that aggregate check, action proposals excluded unassigned tracks, pair proposals required Seat adjacency, WebSocket/UI labels were Seat/Candidate-first, and the documented workflow required Seat calibration and assignment.

The mandatory assumptions were concentrated in:

- `backend/app/features/sessions/service.py`
- `backend/app/features/monitoring/service.py`
- `backend/app/ai/action_recognition/proposals.py`
- `backend/app/ai/action_recognition/runtime.py`
- `backend/app/monitoring/worker.py`
- `frontend/src/features/sessions/SessionReadinessPanel.tsx`
- `frontend/src/features/monitoring/TrackingCanvas.tsx`
- `frontend/src/features/monitoring/MonitoringPage.tsx`

## Runtime identity boundary

`actor_id` is a process/runtime identity such as `A0007`. It is not a Candidate ID and is not persisted. A restart or arbitrary seek may reuse the display sequence from `A0001` because both create a fresh runtime generation. Raw ByteTrack IDs remain in debug payloads only.

States are `ACTIVE`, `LOST`, and `EXPIRED`. Tracklet memory retains bounded center and raw-ID histories. LOST actors older than `max_lost_ms` are expired and removed from recovery memory. Full frames and embeddings are never written to PostgreSQL or sent over WebSocket.

## Recovery contract

For a new raw ID and each eligible LOST actor:

```text
predicted_center = last_center + velocity_per_ms × gap_ms
position_score = 1 - distance(predicted_center, new_center) / max_position_distance
scale_score = min(old_area, new_area) / max(old_area, new_area)
motion_score = direction cosine mapped to [0,1]

recovery_score =
    0.55 × position_score
  + 0.25 × motion_score
  + 0.20 × scale_score
```

Candidates are gated before scoring by source-time gap, normalized position distance, and scale similarity. Matching is deterministic greedy, score-descending, and one-to-one. If the top two scores differ by less than the configured ambiguity margin, motion does not guess.

The selected GTX1650 motion thresholds are recorded in `CONFIG_SPEC.md` and the measured rationale/results are in `SEAT_FREE_TRACKING_REPORT.md`.

## Sparse appearance experiment

Ultralytics 8.4.155 can pass through native detector features only when its tracker is driven by a compatible predictor feature hook. ExamGuard's custom `YOLO.predict()` → `ByteTrackAdapter` path does not expose those features. No OSNet/torchreid dependency or licensed checkpoint exists in the repository.

EXP-TRACK-B therefore uses an internal CPU `hsv_histogram_v1` appearance baseline: a valid crop is resized to 64×128, converted BGR→HSV, divided into a 2×2 grid, encoded with 16×8 H/S histograms per cell, concatenated, and L2-normalized. Similarity is cosine similarity combined with the already-gated motion score. This is not represented as a learned person-ReID model.

Appearance work is batched, bounded by `max_batch_size`, and never runs per actor per detector frame. Prototypes are seeded once under stable/high-confidence conditions and are not updated during recovery or ambiguity. The experiment is disabled in deployment profiles because it produced no identity improvement on the available clips.

## Lifecycle

- Pause advances no actor age because the manager sees only source/video timestamps.
- Seek resets ByteTrack, logical actors, LOST/appearance memory, action buffers, and proposal mappings in the new generation.
- Stop destroys the worker-owned tracker, logical manager, appearance state, dynamic graph, and bounded queues.
- Restart creates fresh Actor IDs; they are not durable Candidate identities.

## Evaluation boundary

The available three real 60-second clips cover seated candidates, adjacency, leaning/turning, partial occlusion, and extra-person movement. They do not contain a complete stand/move/sit sequence or a representative supervisor aisle crossing. Actor-region annotations support conservative recovery consistency checks, but they are not dense MOT ground truth, so IDF1/HOTA are not claimed.
