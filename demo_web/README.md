# Realtime demo website

Replays the real outputs of the detector on the validation sessions S07 and S08 (never used for training) as if the
video were a live camera feed. At time T the page only uses decisions made at or before T: every second the model
decides on the last 3.2 s of each tracked person, the decision rule (smoothing + hysteresis) runs in the browser.

```bash
python demo_web/build_demo.py   # once: 720p videos + JSON data from the projects' eval logs (~2 min)
python demo_web/serve.py        # http://127.0.0.1:8000 (Range support, needed for seeking)
```
No internet or GPU needed; plain HTML / CSS / JS, no libraries.

## Features
- Video with live overlay: box per tracked person, predicted class, cheat score, class probability bars.
  Optional ground-truth boxes (dashed) with the annotated behaviour.
- LIVE / PAUSED / END indicator, speed 0.5x to 4x, previous / next ground-truth event, next alert.
- Decision rule sliders (alert on / off thresholds, smoothing, minimum alert length): the whole replay, metrics and
  timeline are recomputed instantly, so the precision / recall trade-off can be shown on stage.
- Timeline per participant: ground-truth events, model alerts, cheat score curve; 2 min or whole-session view,
  click to seek, option to reveal future alerts.
- Per-person status cards: NORMAL / ALERT, score meter with thresholds, class probabilities.
- Live metrics up to the current time: events caught / missed, alerts, false alarms, precision, false alarms per
  person-hour (same rules as `tune_thresholds.py`: v1 on S07 + S08 gives 43 / 52 at 15.6, as in the README).
- Alert log (matches event / false alarm / ongoing) and ground-truth list (caught with delay / missed / in progress),
  CSV export of the alerts raised so far.
- Whole-session comparison table of all models (v1, v2, v3 fusion, v3 late fusion on S07 only, v1 variant A).

Keyboard: Space play / pause, Left / Right 5 s, `,` / `.` previous / next event, A next alert, Home restart.
