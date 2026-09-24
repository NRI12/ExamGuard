"""Download a public Google Drive folder with one overall progress bar (bytes, speed, ETA) and resume.

  python gdrive_download.py <folder url> <out dir>

1. gdown lists the folder tree (no download), 2. each file's size is read from the download endpoint,
3. files stream through one tqdm bar over the total size; partial files resume with HTTP Range.
"""
import os
import sys
import time

import gdown
import requests
from tqdm import tqdm

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def remote_size(sess, fid):
    with sess.get(URL.format(fid), stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"{fid}: Drive returned an HTML page (quota exceeded or not public)")
        return int(r.headers.get("Content-Length", 0))


def main():
    url, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    files = gdown.download_folder(url, output=out, skip_download=True, quiet=True)
    sess = requests.Session()
    sizes = {}
    for f in tqdm(files, desc="sizing", unit="file"):
        sizes[f.id] = remote_size(sess, f.id)
    total = sum(sizes.values())
    print(f"{len(files)} files, {total / 1e9:.2f} GB total", flush=True)
    for f in files:
        print(f"  {sizes[f.id] / 1e6:10.1f} MB  {f.path}", flush=True)

    done = sum(min(os.path.getsize(f.local_path), sizes[f.id]) for f in files if os.path.exists(f.local_path))
    bar = tqdm(total=total, initial=done, unit="B", unit_scale=True, unit_divisor=1024, desc="download",
               mininterval=5, smoothing=0.05)
    for f in files:
        os.makedirs(os.path.dirname(f.local_path), exist_ok=True)
        have = os.path.getsize(f.local_path) if os.path.exists(f.local_path) else 0
        if have >= sizes[f.id] > 0:
            continue
        for attempt in range(5):
            try:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with sess.get(URL.format(f.id), headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    if have and r.status_code != 206:  # server ignored Range: start over
                        bar.update(-have)
                        have = 0
                    with open(f.local_path, "ab" if have else "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                            have += len(chunk)
                            bar.update(len(chunk))
                            bar.set_postfix_str(os.path.basename(f.path)[:40], refresh=False)
                break
            except (requests.RequestException, OSError) as e:
                tqdm.write(f"retry {attempt + 1} for {f.path}: {e}")
                time.sleep(5 * (attempt + 1))
    bar.close()
    bad = [f.path for f in files if not os.path.exists(f.local_path) or os.path.getsize(f.local_path) != sizes[f.id]]
    print("DOWNLOAD_DONE" if not bad else f"INCOMPLETE: {bad}", flush=True)


if __name__ == "__main__":
    main()
