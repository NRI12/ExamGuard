"""Create the DOAN1 validation sessions (S07, S08) in a running ExamGuard, through its REST API.

For each session: a Room whose six Seats are calibrated on the video itself (median YOLO11n person box of
each annotated participant over sampled frames, matched to the CVAT ground truth), six Candidates, the
1080p video upload, and a READY exam session with every candidate assigned to its seat, ready to be
monitored. Idempotent: existing rooms / candidates / sessions are reused.

  backend/.venv/bin/python scripts/vast/seed_demo.py --videos /workspace/live_demo/media \
      --gt /workspace/live_demo/data --api http://127.0.0.1:8001
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import httpx
import numpy as np
from ultralytics import YOLO

BOX_MAP = dict(dx1=-0.101, dx2=0.094, dy1=-0.131, h=1.021)  # YOLO box -> annotated upper body + desk


def mapped(box):
    x1, y1, x2, _ = box
    w = x2 - x1
    top = y1 + BOX_MAP["dy1"] * w
    return [x1 + BOX_MAP["dx1"] * w, top, x2 + BOX_MAP["dx2"] * w, top + BOX_MAP["h"] * w]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def calibrate_seats(video: Path, gt: dict, detector: YOLO, samples: int = 60) -> dict[str, list[float]]:
    """pid -> normalized [x, y, w, h] seat: median YOLO box of that participant, slightly padded."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes: dict[str, list[list[float]]] = {pid: [] for pid in gt["persons"]}
    index = {pid: {round(r[0], 1): r[1:] for r in rows} for pid, rows in gt["persons"].items()}
    for frame_no in np.linspace(0, n - 1, samples).astype(int):
        frame_no = int(round(frame_no / 5) * 5)  # ground truth every 5 frames (0.2 s)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        t = round(frame_no / fps, 1)
        result = detector.predict(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
        detections = result.boxes.xyxy.cpu().numpy().tolist()
        for pid, rows in index.items():
            truth = rows.get(t)
            if truth is None or not detections:
                continue
            best = max(detections, key=lambda d: iou(mapped(d), truth))
            if iou(mapped(best), truth) >= 0.3:
                boxes[pid].append(best)
    cap.release()
    seats = {}
    for pid, found in sorted(boxes.items()):
        if len(found) < 5:
            print(f"  {pid}: only {len(found)} matched frames, seat skipped")
            continue
        x1, y1, x2, y2 = np.median(np.array(found), axis=0)
        pad_x, pad_y = 0.04 * (x2 - x1), 0.04 * (y2 - y1)
        x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        seats[pid] = [round(x1 / width, 4), round(y1 / height, 4),
                      round((x2 - x1) / width, 4), round((y2 - y1) / height, 4)]
        print(f"  {pid}: seat {seats[pid]} from {len(found)} frames")
    return seats


class Api:
    def __init__(self, base: str, username: str, password: str) -> None:
        self.http = httpx.Client(base_url=base.rstrip("/") + "/api/v1", timeout=600)
        token = self.http.post("/auth/login", json={"username": username, "password": password})
        token.raise_for_status()
        self.http.headers["Authorization"] = f"Bearer {token.json()['access_token']}"

    def call(self, method: str, path: str, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise SystemExit(f"{method} {path} -> {response.status_code} {response.text}")
        return response.json() if response.content else None

    def find(self, path: str, key: str, value: str):
        page = self.call("GET", path, params={"q": value, "page_size": 100})
        return next((item for item in page["items"] if item[key] == value), None)


def seed_session(api: Api, sid: str, video: Path, gt: dict, detector: YOLO) -> None:
    print(f"== {sid}")
    room_code = f"DOAN1-{sid}"
    room = api.find("/rooms", "code", room_code)
    if room is None:
        room = api.call("POST", "/rooms", json={
            "code": room_code, "name": f"Phòng DOAN1 (camera {sid})",
            "description": "Seats calibrated on the video (YOLO11n + CVAT ground truth)"})
    seats = api.call("GET", f"/rooms/{room['id']}/seats")
    if not seats:
        layout = calibrate_seats(video, gt, detector)
        seats = api.call("PUT", f"/rooms/{room['id']}/seats", json={"seats": [
            {"code": pid, "x": x, "y": y, "width": w, "height": h, "sort_order": i}
            for i, (pid, (x, y, w, h)) in enumerate(layout.items())]})
    seat_by_code = {seat["code"]: seat for seat in seats}

    candidates = {}
    for pid in sorted(seat_by_code):
        code = f"{sid}-{pid}"
        candidate = api.find("/candidates", "candidate_code", code)
        if candidate is None:
            candidate = api.call("POST", "/candidates", json={
                "candidate_code": code, "full_name": f"Thí sinh {pid} ({sid})", "class_name": "DOAN1"})
        candidates[pid] = candidate

    session_code = f"DOAN1-{sid}"
    session = api.find("/sessions", "session_code", session_code)
    if session is None:
        session = api.call("POST", "/sessions", json={
            "session_code": session_code, "exam_name": f"DOAN1 {sid} (kiểm định)",
            "room_id": room["id"], "runtime_profile": "rtx3060"})
    if session.get("video") is None:
        print(f"  uploading {video.name} ({video.stat().st_size / 1e6:.0f} MB) ...")
        with video.open("rb") as fh:
            media = api.call("POST", "/media/videos", files={"file": (f"{sid}.mp4", fh, "video/mp4")})
        session = api.call("PATCH", f"/sessions/{session['id']}", json={
            "source_type": "VIDEO_UPLOAD", "video_asset_id": media["id"]})
    if not session.get("assignments"):
        session = api.call("PUT", f"/sessions/{session['id']}/candidates", json={"assignments": [
            {"candidate_id": candidates[pid]["id"], "seat_id": seat_by_code[pid]["id"]}
            for pid in sorted(candidates)]})
    if session["status"] == "DRAFT":
        session = api.call("PATCH", f"/sessions/{session['id']}", json={"status": "READY"})
    print(f"  session {session_code}: {session['status']}, {len(session['assignments'])} candidates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True, help="folder with S07_src.mp4, S08_src.mp4")
    ap.add_argument("--gt", type=Path, required=True, help="folder with gt_S07.json, gt_S08.json")
    ap.add_argument("--api", default="http://127.0.0.1:8001")
    ap.add_argument("--sessions", nargs="+", default=["S07", "S08"])
    args = ap.parse_args()
    api = Api(args.api, os.environ["ADMIN_USERNAME"], os.environ["ADMIN_PASSWORD"])
    detector = YOLO(str(Path(os.environ["MODEL_ROOT"]) / "detection" / "yolo11n.pt"))
    for sid in args.sessions:
        gt = json.loads((args.gt / f"gt_{sid}.json").read_text())
        seed_session(api, sid, args.videos / f"{sid}_src.mp4", gt, detector)


if __name__ == "__main__":
    main()
