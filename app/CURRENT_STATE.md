# ExamGuard Current State

Updated: 2026-09-24

## Seat-Free Stable Tracking

The default runtime identity path is now YOLO11n → ByteTrack → LogicalTrackManager → Stable Actor ID. Seat layout and Candidate mapping are optional enrichments; a READY session with an active Room and source video can start with zero Seats. Dynamic Actor-neighbor proposals replace mandatory Seat adjacency in logical mode. The GTX1650 motion-only path sustained 12.483 realtime FPS with bounded lag and queue size one. Sparse appearance produced no identity improvement and remains disabled. The strict overall gate is NOT PASSED because dense MOT/challenging crossing ground truth is unavailable; see `docs/SEAT_FREE_TRACKING_REPORT.md`.

## Checkpoint

- Greenfield baseline starts at `e6870d2`; `0805e64` preserves the reusable CUDA environment while removing inherited legacy application files.
- PostgreSQL uses the single `20260920_0001_initial_schema` migration and the approved domain tables only.
- Docker Compose, CUDA/PyTorch, NVIDIA runtime, FFmpeg, PostgreSQL volume, media storage, YOLO11n and ByteTrack infrastructure remain unchanged.
- The earlier Seat-Stable Identity milestone remains as optional advanced enrichment (`Stable Actor → Seat → SessionCandidate`). It is no longer the default identity prerequisite.

## Phase 5.5 UX Audit

The operator UI now uses an AdminLTE-inspired application shell: a dark responsive
sidebar, compact top navigation, content header, white cards/tables/forms, status
accents, mobile sidebar backdrop, and a matching login card. Existing routes,
permissions, validation, and monitoring behavior are unchanged.

| Page | CRUD / validation | Loading / empty / error | Phase 5.5 result |
|---|---|---|---|
| Login | Login with normalized username and field validation | Submit and connection errors | Vietnamese field-level validation and duplicate-submit protection |
| Monitoring | Select and operate a ready session | Explicit loading, empty, retryable error | Vietnamese UI, bounded search, real readiness panel, stop confirmation |
| Rooms | Create, read, update, deactivate/reactivate | Loading, empty, retryable error | Pagination, debounced search, field errors, confirmation and toast |
| Seat layout | Create, update and soft-deactivate omitted seats | Empty, error and saving states | Local validation, dirty protection, removal consequence and toast |
| Candidates | Create, read, update and search; no destructive delete | Loading, empty, retryable error | Pagination, debounced search, field errors and toast |
| Sessions | Create, read, search/filter and retain history | Loading, empty, retryable error | Pagination, localized status, validation and consistent date display |
| Session detail | Editable before monitoring; cancel and preserve history | Loading, empty, retryable error | Pre-flight checks, metadata edit, dirty assignment protection and confirmations |
| Navigation/settings | Existing roles only | Honest unavailable states | Role-oriented Vietnamese navigation and localized role names |

## Contracts

- Rooms, candidates and sessions use `{items, page, page_size, total}` list responses.
- List endpoints accept `page`, `page_size` and `q`; sessions also accept `status`.
- API errors keep English machine codes and messages, add an optional `field`, and are mapped to Vietnamese in the frontend.
- Frontend input validation improves UX; Pydantic, services and PostgreSQL remain authoritative.
- Session edits and assignment changes are rejected after monitoring starts. Pre-start cancellation is append-only audited as `SESSION_CANCELLED`.
- Existing seat rows omitted from a saved layout are deactivated, not hard deleted.

## Verification

- Frontend unit tests, TypeScript checks and production build pass.
- Backend Ruff passes.
- Alembic remains at the single initial-schema head; Phase 5.5 requires no schema migration.
- Backend mypy passed for the 76 source files present before the RBAC user-management
  additions; the expanded backend still requires a fresh container mypy run.
- The 20 targeted detector/tracker/runtime tests from the prior tracking gate pass.
  The full backend suite has not been rerun for the expanded RBAC tree because
  volume-backed Docker execution is blocked by the current approval service.
- No database volume reset or environment-file change was performed.

## Tracking Validation and Benchmark Gate

```text
TRACKING STATUS: PASSED ON THE AVAILABLE MINI VALIDATION SET
```

- Runtime correctness passes: one runtime/worker/tracker, ordered tracker input,
  latest-frame queue size 1, bounded lag, correct pause/resume, clean seek generation,
  stale WebSocket rejection, one Canvas state, and deterministic cleanup.
- The final balanced profile uses YOLO `conf=0.10`, `iou=0.50`, `classes=[0]`,
  `max_det=64` plus conservative nested partial/full-body suppression; ByteTrack
  uses new-track `0.40` and buffer 30.
- Three independent 60-second videos were evaluated. Three other uploads were exact
  SHA-256 duplicates and were excluded from the mini validation set.
- Nine manually reviewed timestamps contain 60 visible-person observations: recall
  60/60, zero duplicate-person observations, no isolated non-person false track,
  and zero active-count error at the reviewed timestamps.
- Across all 2,253 analyzed frames, no clip exceeds its reference person count;
  the mean absolute active-count error is 0.424. This all-frame number uses a
  constant per-clip visible-person reference and is not a formal MOT metric.
- Track fragmentation during long/partial occlusion remains: 34 track IDs are
  created for 20 people across the three clips. Track ID remains runtime-only and
  must not be used as permanent candidate identity.
- Final CPU realtime probe: target 12.5 FPS, actual 12.50 FPS, lag mean/P95/max
  23.53/27/111 ms, queue peak 1, and clean stop.
- Native host `nvidia-smi` remains unavailable because `/dev/nvidia*` is not
  created, but NVIDIA Container Toolkit GPU access now works. Phase 6 records
  containerized CUDA/VRAM measurements below; browser/live-scheduler lag is
  still not claimed.
- Supervisor-walk and complete stand/sit source clips are not available in the current
  independent uploads and remain coverage gaps, so this is not a universal accuracy
  claim.
- Full evidence, ablations, metrics, commands, and blockers are in
  `docs/TRACKING_VALIDATION_REPORT.md`.

## RBAC

```text
RBAC STATUS: NOT PASSED
RBAC UI STATUS: PASSED
RBAC USER MANAGEMENT: IMPLEMENTED, BACKEND TEST EXECUTION PENDING
```

- Fixed permissions are centralized for the existing `ADMIN`, `SUPERVISOR`, and
  `REVIEWER` roles. No role/permission tables or database migration were added.
- All implemented room, candidate, session, media, monitoring, and tracking
  WebSocket routes now enforce named permissions after authentication.
- The backend returns `403` with code `FORBIDDEN` for authenticated users who
  lack a required permission. Inactive-user and existing business-state checks
  remain authoritative.
- Frontend routes, role landing pages, sidebar navigation, editing controls,
  media upload, monitoring controls, and development diagnostics use the
  centralized permission map. Forbidden direct routes show an explicit
  Vietnamese `403` state without redirecting an authenticated user to Login.
- The ADMIN landing page has a compact operational overview backed by real
  session and active-user counts. SUPERVISOR lands on Monitoring; REVIEWER lands
  on the implemented read-only Sessions page because Event Review is not yet present.
- Navigation exposes only implemented routes. The top navbar localizes the role
  and provides account details plus logout without exposing raw permissions.
- ADMIN now has `Cài đặt → Người dùng` backed by real `/api/v1/users` endpoints:
  paginated search/filter, account creation, full-name/role editing, deactivation,
  and reactivation. Passwords require at least 12 characters with a letter and a
  number, are hashed immediately, and are never returned by the API.
- Role/status changes take effect on the next authenticated request because the
  backend resolves the current user from PostgreSQL for every request. Self role
  changes, self deactivation, and removal of the last active ADMIN are rejected.
- User creation, profile changes, role changes, deactivation, and reactivation
  append dedicated audit records. ADMIN has a read-only `Nhật ký hệ thống` page
  backed by `/api/v1/audit-logs`; other roles cannot access either admin route.
- Supervisor candidate management now matches the RBAC specification;
  reviewers retain genuinely read-only candidate/session views and cannot
  trigger monitoring lifecycle or seek synchronization APIs.
- Page/resource `403` responses use a consistent Vietnamese content message;
  mutation `403` responses retain the action-specific message.
- Existing append-only audit calls remain in the room, candidate, session,
  assignment, media, and monitoring services. RBAC adds no per-frame or
  WebSocket-message audit records.
- Added backend permission-map and authorization regression tests. Frontend tests
  cover the permission map, role landings/navigation, direct-route authorization,
  `PermissionGate`, and the Vietnamese Forbidden page.
- Frontend verification: `npm test` passes 10 files and 27 tests; `npm run
  typecheck` passes; `npm run build` passes with 124 modules transformed; `git
  diff --check` passes. No frontend lint script is configured in `package.json`.
- Backend Ruff and format checks pass, and backend user-management/API regression
  tests were added. Fresh backend pytest and mypy execution are blocked because
  the host virtualenv targets container Python 3.11, while the Docker volume run
  requires an approval service that currently returns `MODEL_NOT_FOUND`. The
  overall RBAC status therefore remains `NOT PASSED`.
- Events/reviews, evidence, appeals, reports, and system settings do not yet
  expose application APIs in this milestone. Event/review,
  evidence, appeal, report, and system-setting navigation/pages remain omitted
  instead of fabricating data or placeholder workflows.
- No detector, tracker, scheduler, WebSocket schema, or other AI behavior was
  changed by the RBAC UI implementation.

## Next Boundary

```text
SEAT-STABLE IDENTITY: NOT PASSED
IMPLEMENTATION/UNIT/INTEGRATION GATE: PASSED
```

- Runtime identity context is loaded once at monitoring start. PostgreSQL is not queried per
  analysis frame, and no tracking/identity history table or migration was added.
- Matching uses normalized source-video coordinates, explicit overlap/distance scoring, greedy
  one-to-one assignment, temporal confirmation/release, switch hysteresis, and seat grace.
- Pause uses no wall clock. Seek resets ByteTrack and Seat Assignment in the same generation
  transition. A restarted runtime constructs fresh state.
- WebSocket tracking messages now include compact identity and seat occupancy metadata. The
  frontend joins `session_candidate_id` with the existing session REST assignments and shows
  `Seat • Candidate` instead of Track ID in normal mode.
- Seat calibration now uses an authenticated real media frame or a locally selected MP4 rather
  than a blank background. Existing H9302 seat coordinates were not automatically overwritten.
- Three real 60-second clips were evaluated at 5 FPS using production YOLO11n, ByteTrack, and
  SeatAssignmentEngine: 903 frames, 5,418 expected actor samples, 95.146% correct, 0% wrong,
  4.854% unassigned, zero false seat switches, and zero false identities across 345 extra-track
  samples. CPU seat assignment mean/P95 was 0.667/0.770 ms.
- The overall gate remains NOT PASSED because the available independent sources do not contain
  a supervisor walking through the aisle or a complete stand/leave/move sequence, and the
  persisted H9302 layout still needs operator recalibration for the selected camera view. These
  missing real-world acceptance scenarios must not be replaced by unit-test claims.
- Backend verification: 85 passed, 1 skipped (explicit CUDA-only test); Ruff passes; mypy passes
  for 91 source files. Frontend verification: 10 files/28 tests pass and production build passes.
- Full formulas, state machines, validation definitions and limitations are in
  `docs/SEAT_STABLE_IDENTITY_REPORT.md`.

## Phase 6 — Action Recognition Runtime

```text
PHASE 6 FUNCTIONAL IMPLEMENTATION: PASSED
PRE-STABILIZATION PERFORMANCE: FAILED (5.43 FPS < 10 FPS)
RESOLUTION: SEE PHASE 6.5 BELOW
```

- Added stable SessionCandidate-based Single and Seat-adjacent Pair proposals,
  verified R3 ROI geometry, bounded timestamped 4-second RGB ROI buffers,
  a shared SHA-validated TSM-ResNet50 R3 adapter, capacity-one action queues,
  generation-safe seek/reset and low-frequency raw ActionPrediction WebSocket
  messages. The normal operator UI remains unchanged; raw action diagnostics
  appear only in the developer drawer. No database migration or per-frame DB
  writes were added.
- Frozen evaluation vs production on one 100-frame R3 clip: identical sample
  indices, preprocessing max absolute difference 0.0, probability max absolute
  difference 0.0 and matching top-1. Dynamic runtime bbox ROIs are not claimed
  to reproduce the training dataset's clip-static annotation ROI exactly.
- Three real 60-second uploads were run through production YOLO11n → ByteTrack
  → Seat Identity → proposal/ROI/buffer → R3 checkpoint on CPU: 903 frames per
  mode, 18 unique Single IDs, 15 unique Pair IDs and 331 raw predictions. No
  inference error; one model load; 82 stale action requests were dropped.
- Full backend suite: 106 collected, 105 passed, 1 optional CUDA test skipped.
  Ruff passes; mypy passes for 101 app source files. Frontend Vitest passes
  10 files/28 tests and production typecheck/build passes.
- The CPU offline throughput comparison fell from 31.17 FPS tracking-only to
  8.33 FPS tracking+action. A later three-clip GTX1650 Max-Q CUDA/FP16 rerun
  measured 38.97 FPS tracking-only and 5.43 FPS tracking+action, with 837 raw
  predictions, no CUDA/action error, and observed peaks of 100% GPU, 531 MiB
  VRAM and 71°C. That baseline failed the configured 10-FPS minimum; Phase 6.5
  below resolves the runtime bottleneck. Existing Seat Identity field
  acceptance gaps remain independent of Action Runtime performance.
- Contract, numeric equivalence, per-clip latency/performance and all 50 Phase
  6 report items are in `docs/TSM_RUNTIME_CONTRACT.md` and
  `docs/PHASE6_ACTION_RECOGNITION_REPORT.md`.

## Phase 6.5 — Action Runtime Performance Stabilization

```text
PHASE 6: FUNCTIONAL PASS
PHASE 6.5: PERFORMANCE PASS
ACTION RECOGNITION: PASS (raw prediction runtime)
```

- Added an explicit oldest-prediction-first Action Scheduler between ready
  timestamp buffers and TSM. It maintains one latest ready clip per stable
  SessionCandidate proposal, per-proposal source-timestamp stride, capacity-one
  dispatch, B2 compute budget, 2000-ms max age and separate replaced/expired
  metrics. Pause remains timestamp-driven; seek resets buffers, scheduler and
  stale output generation; stop joins action threads.
- Isolated real-checkpoint profiling found the GTX1650's TSM FP16 path is about
  three times slower than FP32 and produces non-finite output at B2/B8. The GTX
  profile now uses validated FP32, stride 1500 ms, max batch 2, queue 1 and
  minimum inference interval 200 ms. A finite-output guard prevents invalid
  probabilities from being published.
- Same-video ablations executed stride 500/1000/1500/2000 at B2 and B1/B2/B4
  at stride 1500. Selected B2 preserves approximately 1.5–1.7-second Single
  and Pair cadence without B4's longer kernel/VRAM cost or B1's 2.7-second P95
  coverage loss.
- Three 60-second representative clips passed offline at minimum 14.39 FPS.
  An additional 180 seconds of realtime-paced validation sustained 12.51 FPS
  on all clips with lag P95 100–104 ms, non-increasing lag, ActionPrediction
  age P95 160–320 ms, all Single/Pair IDs covered, zero action errors/OOM and
  final queue depth zero.
- The scientific contract is unchanged. A real 100-frame R3 clip still has
  preprocessing max difference 0.0 and FP32 probability max difference 0.0.
  No Head Pose, object branch, Event FSM, Event or Evidence functionality was
  added.
- `action_recognition.enabled` remains false by default so deployment activation
  is explicit and is not confused with an action-accuracy claim. RTX3060 is
  structured separately but remains unbenchmarked.
- Full results and raw artifact locations:
  `docs/PHASE6_5_ACTION_PERFORMANCE_REPORT.md`.

Stop after Phase 6.5. Do not automatically begin Phase 7/Event generation.

## Phase 7 — Event Aggregation

```text
PHASE 7 IMPLEMENTATION/UNIT/INTEGRATION GATE: PASSED
PHASE 7 RUNTIME CALIBRATION: COMPLETED
PHASE 7 EVENT-ACCURACY GATE: FAILED
PHASE 7: NOT PASSED
```

- Added causal class-specific EMA, timestamp hysteresis and IDLE/CANDIDATE/
  ACTIVE/COOLDOWN FSM state per stable proposal and behavior. Normal is excluded;
  Single/Pair routing is enforced; one score spike cannot create an Event.
- Added exact-actor temporal deduplication and idempotent internal persistence to
  the existing Event/EventActor schema. AI events remain `PENDING_REVIEW`, use
  `created_by=NULL`, append `AI_EVENT_CREATED`, and never create EventReview or
  Evidence. One Pair interaction produces one Event with two EventActors.
- Pause uses no wall clock. Tracking source timestamps expire discontinuous
  proposal state. Seek resets every non-persisted smoother/FSM/cooldown/dedup
  state; historical finalized events remain. Stop flushes ACTIVE/COOLDOWN at the
  last valid evidence timestamp and discards insufficient CANDIDATE state.
- Added production-path calibration CLI, pilot-manifest builder, explicit 2D Seat
  neighbor graph support, JSONL export, actor/time GT alignment, raw metrics,
  bounded deterministic search and Event evaluation. Raw predictions remain
  file-only.
- GPU collection ran YOLO11n → ByteTrack → Seat Identity → proposal/dynamic ROI →
  TSM on S00–S04: 52.2407 minutes, 106 GT intervals and 20,210 predictions.
  S05–S08 stayed excluded final test and were not inferred or evaluated.
- Raw Macro F1 is 0.2332. A one-time refined bounded search produces Event
  Precision/Recall/F1 0.1452/0.1047/0.1216, Macro Event F1 0.1064, 1.01 false
  events/minute and zero
  communicating Event F1. Calibration completed, but the empirical accuracy gate
  failed; both production profiles remain disabled and final test was not run.
- Root-cause diagnosis found no pair-normal examples in the 509-sample development
  training set: all 218 normal samples contain one actor, while all 27 communicating
  and 33 exchange samples contain two. Runtime same-row Pair proposals are therefore
  top-1 communicating 89.0% of the time outside event intervals. Per-proposal
  baseline normalization also fails (GT delta P50 −0.0241).
- Reproducible artifacts are under `docs/event_calibration/artifacts/`; the
  searched config is evaluation evidence with `enabled: false`, not a deployment
  approval. The checkpoint was trained on S00–S04 windows, so this development
  search is not an independent model-validation result.
- Final backend verification and the full 42-point report are in
  `docs/PHASE7_EVENT_AGGREGATION_REPORT.md`. Frontend code was unchanged and was
  not rerun. Backend verification: 129 collected, 124 passed and 5
  environment-gated tests skipped; Ruff and mypy (111 source files) pass.

Stop after Phase 7. Do not automatically begin Phase 8.
