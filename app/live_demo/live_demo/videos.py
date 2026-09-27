"""Videos the service can analyse.

Built-in sessions (S07 / S08, with ground truth):
  media/<id>_src.mp4   original 1080p video, read by the detector (what the models were evaluated on)
  media/<id>.mp4       720p H.264 faststart copy, played by the browser
  data/gt_<id>.json    ground-truth boxes and events (from demo_web/build_demo.py)
Uploads (no ground truth): media/uploads/<id>/{source.*, play.mp4, meta.json}
"""
import json
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from .config import DATA, MEDIA, SESSIONS


@dataclass
class Video:
    id: str
    name: str
    source: str            # file decoded by the detector
    media: str             # URL played by the browser
    fps: float = 25.0
    frames: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    gt: str | None = None  # URL of the ground-truth JSON
    persons: int = 0
    events: int = 0
    status: str = "ready"  # ready | processing | error
    error: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)

    def public(self):
        d = asdict(self)
        d.pop("source")
        d.pop("extra")
        return d


def probe_cv(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or w <= 0:
        raise ValueError("no video stream")
    return fps, n, w, h


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt:format=format_name", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    j = json.loads(out)
    s = (j.get("streams") or [{}])[0]
    return s.get("codec_name", ""), s.get("pix_fmt", ""), j.get("format", {}).get("format_name", "")


def browser_copy(src, dst):
    """H.264 / yuv420p MP4 is remuxed (seconds), anything else is transcoded to 720p H.264."""
    codec, pix, fmt = ffprobe(src)
    tmp = dst.with_suffix(".part.mp4")
    if codec == "h264" and pix == "yuv420p" and "mp4" in fmt:
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-c", "copy", "-an",
               "-movflags", "+faststart", str(tmp)]
    else:
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-vf", "scale=-2:720",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "50", "-pix_fmt", "yuv420p", "-an",
               "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=6 * 3600)
    tmp.replace(dst)


class Library:
    def __init__(self):
        self.videos: dict[str, Video] = {}
        self.lock = threading.Lock()
        (MEDIA / "uploads").mkdir(parents=True, exist_ok=True)
        self._load_builtin()
        self._load_uploads()

    def _load_builtin(self):
        for sid in SESSIONS:
            play, src, gt = MEDIA / f"{sid}.mp4", MEDIA / f"{sid}_src.mp4", DATA / f"gt_{sid}.json"
            if not play.exists() or not gt.exists():
                continue
            g = json.loads(gt.read_text())
            source, note = src, ""
            if not src.exists():  # fall back to the 720p copy (boxes are in 1080p coordinates: scaled later)
                source, note = play, "1080p source missing: analysing the 720p copy (results differ from eval)"
            fps, n, w, h = probe_cv(source)
            self.videos[sid] = Video(
                id=sid, name=f"{sid}  (validation, ground truth)", source=str(source), media=f"media/{sid}.mp4",
                fps=fps, frames=n, width=g["width"], height=g["height"], duration=g["duration"],
                gt=f"data/gt_{sid}.json", persons=len(g["persons"]), events=sum(e["cheating"] for e in g["events"]),
                note=note, extra=dict(src_w=w, src_h=h))

    def _load_uploads(self):
        for meta in sorted((MEDIA / "uploads").glob("*/meta.json")):
            try:
                v = Video(**json.loads(meta.read_text()))
            except (TypeError, ValueError):
                continue
            if v.status == "processing":  # interrupted by a restart
                v.status, v.error = "error", "processing interrupted, upload again"
            self.videos[v.id] = v

    def list(self):
        with self.lock:
            return [v.public() for v in self.videos.values()]

    def get(self, vid):
        with self.lock:
            return self.videos.get(vid)

    def _save(self, v):
        (MEDIA / "uploads" / v.id / "meta.json").write_text(json.dumps(asdict(v)))

    def add_upload(self, name, write_body):
        """write_body(path) stores the uploaded bytes; returns the Video (processing in the background)."""
        vid = "U" + uuid.uuid4().hex[:8]
        d = MEDIA / "uploads" / vid
        d.mkdir(parents=True)
        ext = Path(name).suffix.lower()[:8] or ".mp4"
        src = d / f"source{ext}"
        try:
            write_body(src)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        v = Video(id=vid, name=Path(name).name[:80], source=str(src), media=f"media/uploads/{vid}/play.mp4",
                  status="processing")
        with self.lock:
            self.videos[vid] = v
        self._save(v)
        threading.Thread(target=self._prepare, args=(v, src, d / "play.mp4"), daemon=True).start()
        return v

    def _prepare(self, v, src, play):
        try:
            v.fps, v.frames, v.width, v.height = probe_cv(src)
            v.duration = round(v.frames / v.fps, 2)
            browser_copy(src, play)
            v.status = "ready"
        except Exception as e:  # noqa: BLE001 - reported to the page
            v.status, v.error = "error", str(e)[:300]
        self._save(v)

    def delete(self, vid):
        with self.lock:
            v = self.videos.get(vid)
            if not v or not vid.startswith("U"):
                return False
            del self.videos[vid]
        shutil.rmtree(MEDIA / "uploads" / vid, ignore_errors=True)
        return True
