# ExamGuard — Project Context

## Thesis
**Nghiên cứu và xây dựng hệ thống phát hiện hành vi bất thường trong phòng thi sử dụng camera giám sát**

The system combines exam-room video monitoring, person detection, tracking, seat/candidate context, video action recognition, suspicious-event review, evidence storage, historical lookup and appeal support.

## Research context
Video:
- 1920×1080
- 25 FPS
- H.264

Classes:
- normal
- suspicious_looking
- communicating
- exchange_object
- using_phone/cheat_sheet

Action-recognition research includes TSM and X3D. TSM R3 is the later deployment priority. The current milestone is not action recognition yet.

Current milestone:
```text
clean business system
+ clean database
+ upload video
+ native playback
+ YOLO11n
+ ByteTrack
+ WebSocket
+ Canvas overlay
```

## Hardware
Current development:
- Intel Core i7-11800H
- 16 GB RAM
- GTX 1650 Max-Q

Future:
- 16 GB RAM
- RTX 3060 12 GB

## Existing infrastructure stack
Backend/runtime:
- Python 3.11
- FFmpeg / ffprobe
- Docker / Docker Compose
- NVIDIA Container Toolkit
- CUDA runtime inside Docker
- FastAPI / Uvicorn
- Pydantic / pydantic-settings / python-dotenv
- SQLAlchemy / Alembic / psycopg[binary]
- PyJWT / pwdlib[argon2]
- python-multipart
- torch / torchvision
- NumPy / opencv-python-headless / PyYAML / Pillow
- pytest / httpx / ruff / mypy

Frontend:
- React
- react-dom
- react-is
- react-router-dom
- axios
- recharts

New dependency:
- ultralytics

## Greenfield rule
Legacy code may be inspected only for Docker/CUDA/PostgreSQL/frontend-build configuration. Do not preserve legacy schema/API/service structure merely for compatibility.

## Product questions the system must answer
1. Is monitoring running?
2. Who is being monitored?
3. Who needs attention?
4. What happened?
5. Which candidate and seat are involved?
6. What evidence exists?
7. Who reviewed the event?
8. If a candidate appeals later, can the exact evidence be found quickly?
