"""Fetch the videos of the built-in sessions on the server (fast datacenter network, nothing to upload).

  media/S07_src.mp4, media/S08_src.mp4   1080p originals, from the DOAN1 task backups on Google Drive
  media/S07.mp4, media/S08.mp4           720p H.264 copies for the browser (skipped if already uploaded)

  python -m live_demo.fetch_media
"""
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from .config import MEDIA, SESSIONS

DOAN1 = "<DOAN1_DATASET_URL>"
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def remote_size(sess, fid):
    with sess.get(URL.format(fid), headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"{fid}: Google Drive returned a web page (quota exceeded or file not public)")
        return int(r.headers["Content-Range"].split("/")[-1]) if "Content-Range" in r.headers \
            else int(r.headers.get("Content-Length", 0))


def download(sess, fid, path, size):
    done = os.path.getsize(path) if os.path.exists(path) else 0
    for attempt in range(6):
        if done >= size:
            return
        try:
            headers = {"Range": f"bytes={done}-"} if done else {}
            with sess.get(URL.format(fid), headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if done and r.status_code != 206:
                    done = 0
                with open(path, "ab" if done else "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        done += len(chunk)
        except Exception as e:  # noqa: BLE001 - retried
            print(f"  retry {attempt + 1} {os.path.basename(path)}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
            done = os.path.getsize(path) if os.path.exists(path) else 0
    if done < size:
        raise SystemExit(f"incomplete download {path}")


def main():
    import gdown
    import requests
    MEDIA.mkdir(parents=True, exist_ok=True)
    todo = {sid: name for sid, name in SESSIONS.items() if not (MEDIA / f"{sid}_src.mp4").exists()}
    if todo:
        print("listing the DOAN1 Google Drive folder ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            listing = gdown.download_folder(DOAN1, output=tmp, skip_download=True, quiet=True)
        files = {sid: next(f for f in listing if f.path.replace(os.sep, "/").endswith(f"backup/{name}_task_backup.zip"))
                 for sid, name in todo.items()}
        sess = requests.Session()
        with ThreadPoolExecutor(4) as pool:
            sizes = dict(zip(files, pool.map(lambda f: remote_size(sess, f.id), files.values())))
        dl = MEDIA / "_download"
        dl.mkdir(exist_ok=True)
        t0 = time.time()
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda sid: download(sess, files[sid].id, dl / f"{sid}.zip", sizes[sid]), files))
        total = sum(sizes.values())
        print(f"downloaded {total / 1e9:.2f} GB in {time.time() - t0:.0f} s "
              f"({total * 8 / 1e6 / max(time.time() - t0, 1e-3):.0f} Mbit/s)", flush=True)
        for sid, name in todo.items():
            with zipfile.ZipFile(dl / f"{sid}.zip") as zf, zf.open(f"data/normalized/{name}.mp4") as src, \
                    open(MEDIA / f"{sid}_src.mp4.part", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
            os.replace(MEDIA / f"{sid}_src.mp4.part", MEDIA / f"{sid}_src.mp4")
            print(f"{sid}: 1080p source ready", flush=True)
        shutil.rmtree(dl)
    for sid in SESSIONS:
        play = MEDIA / f"{sid}.mp4"
        if play.exists():
            continue
        print(f"{sid}: transcoding the 720p browser copy ...", flush=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(MEDIA / f"{sid}_src.mp4"), "-vf", "scale=1280:720",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "50", "-pix_fmt", "yuv420p", "-an",
                        "-movflags", "+faststart", str(play) + ".part.mp4"], check=True)
        os.replace(str(play) + ".part.mp4", play)
    print("media ready:", ", ".join(sorted(p.name for p in MEDIA.glob("*.mp4"))))


if __name__ == "__main__":
    main()
