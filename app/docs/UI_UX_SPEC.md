# ExamGuard — UI/UX Specification

## Goal
Users should operate the system without understanding AI internals.

Normal workflow:
```text
Login → Monitoring → Select/Create Session → Select Room → Upload Video → Start
```

Appeal workflow:
```text
Search Candidate → Session → Event → Evidence
```

## Navigation
```text
ExamGuard
▣ Monitoring
◷ Sessions
⚠ Events
👤 Candidates
▤ Reports
────────────
⚙ Settings
```
Do not expose YOLO/ByteTrack/GPU/ROI/TSM as primary navigation.

## Theme
Professional dark surveillance UI.
```text
Main background #0F172A
Sidebar         #0B1120
Panel           #111827
Secondary       #1E293B
Border          #334155
Primary text    #F8FAFC
Secondary       #94A3B8
Muted           #64748B
```
Semantic colors:
- normal tracking: cyan/blue
- healthy: green
- warning: amber
- confirmed critical event: red
- inactive/lost: gray

Do not use red for normal tracking boxes. Avoid neon/cyberpunk/rainbow tracks.

## Login
Minimal centered form. No charts or AI metrics.

## Monitoring — no active session
Show create/select session, select room, upload/select video and one primary button: Start Monitoring. Seat layout and Candidate assignment are optional advanced context and never block the default flow. No AI settings here.

## Candidate assignment
Simple table:
```text
Seat   Candidate
A01    SV001 - Nguyễn Văn A
A02    SV002 - Trần Văn B
A03    [Select candidate]
```
Prevent duplicate assignments before save.

## Live Monitoring
Target layout: 72–75% video, 25–28% side panel.

Main area:
- HTML5 video
- Canvas overlay
- playback controls
- fullscreen

Normal labels use mapped business context such as `B03 • SV103` when available; otherwise they use the Stable Actor ID such as `A0007`. Tentative Seat matches may show `Đang xác định…`, but an unmapped actor still receives its Actor label. Track ID and score are visible
only in development diagnostics.

## Side panel
Current phase:
```text
STATUS
● Running
6 persons detected

THÍ SINH
A01 • SV101  Đang theo dõi
A02 • SV102  Tạm mất dấu

Người chưa xác định: 1
```
Seat/identity states are operational tracking states, never suspicious-behavior labels. Later
show events needing attention only after the event milestone. Do not fabricate event data.

## Status bar
Compact:
```text
● AI Online | Video 25 FPS | AI 11.8 FPS | Latency 81 ms
```
Detailed metrics belong to Diagnostics drawer.

## Diagnostics drawer
Show source/analysis FPS, detector/tracker/pipeline ms, GPU, VRAM, CPU, RAM, dropped frames, queue, active profile. Missing values show `—`, never mocked values.

## Seat calibration (optional advanced setup)
Seat calibration is not a monitoring prerequisite. When used, use a real video/camera frame plus editable seat rectangles. User can add/drag/resize/rename/delete/save. Never use a blank white canvas.

## Sessions
List exam name, room, date/time, status, candidate count, event count later and open/review action. No unnecessary charts.

## Events
Recommended three-pane design:
```text
Event list | Evidence viewer | Event details/review
```
Actions: Confirm, Dismiss, Needs Review.

## Candidate detail
Show candidate code/name, sessions, seat per session, events, status and evidence access. Goal: find evidence in a few clicks.

## Appeal
Show candidate, session, related events, evidence, status, notes, resolution and audit history.

## Reports
Only after real events exist. Use Recharts here, not on live monitoring.

## Performance
Do not set React state for tracking at frame frequency if it rerenders the whole page. Prefer WebSocket → buffer/ref → Canvas renderer. Status cards may update 2–4 times/sec.

## Error/loading UX
Examples:
```text
Checking video...
Initializing AI...
Synchronizing analysis...
```
Do not show raw exceptions to users.
