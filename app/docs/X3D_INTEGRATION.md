# X3D-L behaviour classifier and AI events

The monitoring runtime classifies every tracked person with the X3D-L model of `cheating_detection_v1`
(4 classes: normal, looking, interaction, phone_cheatsheet; 43 of 52 DOAN1 S07/S08 events caught at
15.6 false alarms per person-hour). Alerts of candidates resolved to a seat become `PENDING_REVIEW`
Events that a reviewer confirms or dismisses.

## Runtime path

```text
tracking frame (YOLO11n -> ByteTrack -> Stable Actor -> Seat -> SessionCandidate)
  -> CheatingClassifierRuntime.update()             first frame of every 200 ms, RGB + mapped actor box
  -> every 1 s: 16 samples (3.2 s) per actor         square crop 356 px, same geometry as training
  -> X3D-L on its own thread (capacity 1, newest window wins; tracking never waits)
  -> smoothing (3 windows) + hysteresis (start 0.85 / keep 0.70) + min 2 windows
  -> cheat_prediction WebSocket message (per-actor probabilities, alert, label)
  -> closed alert with a SessionCandidate -> AggregatedEvent -> persist_ai_event (Event + EventActor + audit)
```

- Code: `backend/app/ai/cheating_classifier/` (model, runtime), wired in `monitoring/worker.py` and
  `monitoring/manager.py`; configuration `cheating_classifier` in `configs/runtime/rtx3060.yaml`.
- The model is loaded and `torch.compile`d once at backend start-up (about 70 s), shared by all sessions.
- Label -> behaviour: looking -> SUSPICIOUS_LOOKING, interaction -> COMMUNICATING (one actor per event:
  the classifier flags each participant), phone_cheatsheet -> USING_PHONE_CHEAT_SHEET.
- Alerts of people not assigned to a seat are shown live but not stored (an Event needs a SessionCandidate).
- The R3/TSM event FSM stays in the code base; a profile enables one of the two event sources.

## API and UI

- `GET /api/v1/events` (filters `session_id`, `status`, `behavior`, pagination), `GET /api/v1/events/counts`,
  `GET /api/v1/events/{id}` (with review history), `POST /api/v1/events/{id}/reviews`
  (`CONFIRM` | `DISMISS` | `NEEDS_REVIEW`, append-only `EventReview` + `EVENT_REVIEWED` audit).
- Monitoring page: boxes coloured by the live decision with seat and behaviour label, "AI alerts" and
  "recorded events" panels. Events page ("Sự kiện"): list, counts, video clip around the event, review.
- Tracking results may arrive up to 1.2 s after the video frame they describe (remote GPU server); results
  ahead of the player by more than 250 ms are still rejected.

## Measured on Vast.ai RTX 3060 12 GB (S07, 1080p, 6 candidates)

| | value |
|---|---|
| tracking analysis | 16.5 FPS, lag 29 ms |
| X3D-L window (up to 8 actors, crops + compiled model) | about 360 ms per second of video |
| S07 02:00 to 05:05 live | 7 events: P1/P2 interaction 02:04-02:11, P4 interaction 03:01, P6 phone 04:07-04:20, P3/P5 interaction 04:52-05:04 |

## Deployment

- Any Windows or Linux server with an RTX 3060: see `DEPLOY.md` (`deploy/windows/*.cmd`, `deploy/linux/*.sh`).
  One process: uvicorn serves the API, the WebSocket and the built frontend (`FRONTEND_DIST`).
- Vast.ai instance (container, nginx in front): `scripts/vast/setup_instance.sh`, `deploy.sh`, `tunnel.sh`,
  `seed_demo.py` (rooms with seats calibrated on the DOAN1 videos, candidates, READY sessions).
- `COOKIE_SECURE=false` allows a production server over plain HTTP on a LAN; keep it true behind HTTPS.
- Without Triton (usual on Windows) `torch.compile` is skipped and X3D runs in eager mode. RTX 3060,
  torch 2.7.1, 6 actors: 305 ms per window eager vs 245 ms compiled; a 60 s live run in eager mode dropped
  no window (tracking 15 FPS).
