"""Download the raw DOAN1 dataset from its public Google Drive folder (data.source_url).

Only what `prepare` needs is fetched: annotations/backup (task backups with the videos) and
annotations/cvat_zip (CVAT exports); the duplicated normalized/ videos and xml/ exports are skipped.
One tqdm bar over the total size, resume of partial files, retries.

  python -m cheating_v1 download           # ~3.7 GB
  python -m cheating_v1 download --list    # show files and sizes only
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ..config import config_parser, load_config

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
NEEDED = ("/annotations/backup/", "/annotations/cvat_zip/")


def remote_size(fid):
    """Ask for 1 byte and read the total from Content-Range (a plain GET starts streaming the whole file)."""
    import requests
    with requests.get(URL.format(fid), headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"{fid}: Google Drive returned a web page (quota exceeded or file not public)")
        if "Content-Range" in r.headers:
            return int(r.headers["Content-Range"].split("/")[-1])
        return int(r.headers.get("Content-Length", 0))


def download(url, out_dir, needed=NEEDED, list_only=False):
    import gdown
    import requests
    os.makedirs(out_dir, exist_ok=True)
    listing = gdown.download_folder(url, output=out_dir, skip_download=True, quiet=True)
    files = [f for f in listing if any(n in "/" + f.path.replace(os.sep, "/") for n in needed)]
    sess = requests.Session()
    # Drive answers consecutive requests slowly (~12 s each); asking in parallel takes seconds
    with ThreadPoolExecutor(8) as pool:
        sizes = dict(zip([f.id for f in files],
                         tqdm(pool.map(remote_size, [f.id for f in files]), total=len(files), desc="listing",
                              unit="file")))
    total = sum(sizes.values())
    have = lambda f: os.path.getsize(f.local_path) if os.path.exists(f.local_path) else 0
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {out_dir}")
    if list_only:
        for f in files:
            print(f"  {sizes[f.id] / 1e6:9.1f} MB  {f.path}")
        return
    bar = tqdm(total=total, initial=sum(min(have(f), sizes[f.id]) for f in files), unit="B", unit_scale=True,
               unit_divisor=1024, desc="download", mininterval=5, smoothing=0.05)
    for f in files:
        os.makedirs(os.path.dirname(f.local_path), exist_ok=True)
        done = have(f)
        if done >= sizes[f.id] > 0:
            continue
        for attempt in range(5):
            try:
                headers = {"Range": f"bytes={done}-"} if done else {}
                with sess.get(URL.format(f.id), headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    if done and r.status_code != 206:  # Range ignored: restart this file
                        bar.update(-done)
                        done = 0
                    with open(f.local_path, "ab" if done else "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                            done += len(chunk)
                            bar.update(len(chunk))
                            bar.set_postfix_str(os.path.basename(f.path)[:40], refresh=False)
                break
            except (requests.RequestException, OSError) as e:
                tqdm.write(f"retry {attempt + 1} for {f.path}: {e}")
                time.sleep(5 * (attempt + 1))
    bar.close()
    bad = [f.path for f in files if have(f) != sizes[f.id]]
    if bad:
        raise SystemExit(f"incomplete files, run the command again to resume: {bad}")
    print("download complete")


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="list the files and sizes, download nothing")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    # gdown writes the folder's contents into the target: Drive folder DOAN1 -> data/raw/DOAN1
    download(cfg["data"]["source_url"], str(Path(cfg["paths"]["raw_data"]).parent), list_only=args.list)


if __name__ == "__main__":
    main()
