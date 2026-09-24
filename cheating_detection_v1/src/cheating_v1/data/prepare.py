"""Prepare the dataset: extract session videos from the CVAT task backups, then cut per-person clips.

Output (paths.clips):
  videos/{split}/{class}/{sess}_{pid}_{kind}{id}_{start}.mp4   clip_size^2, fps, clip_len frames
  train.csv / val.csv   "relative/path.mp4 label"
  meta.csv              provenance of every clip (kind: e event actor, r related actor, h hard negative, n normal)
  classes.txt, clips_info.json

  python -m cheating_v1 prepare
"""
import csv
import glob
import json
import os
import queue
import random
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from tqdm import tqdm

from ..config import config_parser, load_config
from .cvat import box_at, load_tracks, related_actor, session_names

PROGRESS_EVERY = 250


def extract_videos(cfg):
    os.makedirs(cfg["paths"]["videos"], exist_ok=True)
    backups = sorted(glob.glob(os.path.join(cfg["paths"]["raw_data"], "backup", "*_task_backup.zip")))
    for z in tqdm(backups, desc="extract videos", unit="video"):
        sess = os.path.basename(z).split("_task_backup")[0]
        dst = os.path.join(cfg["paths"]["videos"], f"{sess}.mp4")
        if os.path.exists(dst):
            continue
        with zipfile.ZipFile(z) as zf:
            mp4 = next(n for n in zf.namelist() if n.endswith(".mp4"))
            with zf.open(mp4) as src, open(dst + ".part", "wb") as f:
                while chunk := src.read(1 << 24):
                    f.write(chunk)
        os.replace(dst + ".part", dst)


def square_crop_box(boxes, w, h, pad):
    """One fixed square covering the union of boxes over the clip (no jitter)."""
    b = np.stack(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x2 - x1, y2 - y1) * (1 + pad), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return x1, y1, x1 + int(side), y1 + int(side)


def event_windows(start, end, n_frames, clip_len, stride):
    if end - start + 1 <= clip_len:  # short event: one window centred on it
        return [int(np.clip(start + (end - start + 1) // 2 - clip_len // 2, 0, n_frames - clip_len))]
    last = end - clip_len + 1
    starts = list(range(start, last + 1, stride))
    return starts if starts[-1] == last else starts + [last]


def plan_session(cfg, sess):
    d = cfg["data"]
    L = d["clip_len"]
    rng = random.Random(f"{d['seed']}-{sess}")
    n_frames, tracks = load_tracks(cfg["paths"]["raw_data"], sess)
    persons = {t["attrs"]["participant_id"]: t for t in tracks if t["label"] == "person"}
    split = "val" if sess[:3] in d["val_sessions"] else "train"
    specs, busy = [], defaultdict(list)
    for e in (t for t in tracks if t["label"] in d["label_map"]):
        a, label = e["attrs"], d["label_map"][e["label"]]
        rel = related_actor(a)
        busy[a["participant_id"]].append((e["start"], e["end"]))
        if rel:
            busy[rel].append((e["start"], e["end"]))
        # one clip per person, cropped on that person's own track (inference only needs per-person crops);
        # the related actor gets a clip only for labels in data.related_labels (default: every labelled pair)
        with_related = rel and e["label"] in d.get("related_labels", list(d["label_map"]))
        actors = [(a["participant_id"], "e" if label else "h")] + ([(rel, "r")] if with_related else [])
        for pid, kind in actors:
            if pid not in persons:
                continue
            for s in event_windows(e["start"], e["end"], n_frames, L, d["event_stride"]):
                cover = (min(e["end"], s + L - 1) - max(e["start"], s) + 1) / L
                specs.append(dict(sess=sess, split=split, label=label, cls=d["classes"][label], pid=pid,
                                  related=rel or "", orig_label=e["label"], subtype=a.get("subtype") or "",
                                  quality=a.get("quality") or "", kind=kind, src_id=e["id"], start=s,
                                  boxes=[box_at(persons[pid], f) for f in range(s, s + L, 8)],
                                  cover=round(min(cover, 1.0), 3)))
    n_normal = max(1, int(d["normal_per_person_min"] * n_frames / d["fps"] / 60))
    for pid, p in persons.items():
        blocked = [(s - d["normal_margin"] - L, e + d["normal_margin"]) for s, e in busy[pid]]
        cands = [s for s in range(0, n_frames - L, L // 2) if not any(a <= s <= b for a, b in blocked)]
        for s in sorted(rng.sample(cands, min(n_normal, len(cands)))):
            specs.append(dict(sess=sess, split=split, label=0, cls=d["classes"][0], pid=pid, related="",
                              orig_label="normal", subtype="", quality="", kind="n", src_id=p["id"], start=s,
                              boxes=[box_at(p, f) for f in range(s, s + L, 8)], cover=1.0))
    return specs


def render_session(job):
    sess, specs, video, out, clip_len, size, pad, fps, progress = job
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    by_start = defaultdict(list)
    for sp in specs:
        sp["crop"] = square_crop_box(sp.pop("boxes"), w, h, pad)
        sp["path"] = f"videos/{sp['split']}/{sp['cls']}/{sess[:3]}_{sp['pid']}_{sp['kind']}{sp['src_id']}_{sp['start']:06d}.mp4"
        os.makedirs(os.path.dirname(os.path.join(out, sp["path"])), exist_ok=True)
        by_start[sp["start"]].append(sp)
    active, fourcc = [], cv2.VideoWriter_fourcc(*"mp4v")
    last, f = max(sp["start"] for sp in specs) + clip_len, 0
    while f < last:
        ok, img = cap.read()
        if not ok:
            break
        for sp in by_start.get(f, []):
            sp["writer"] = cv2.VideoWriter(os.path.join(out, sp["path"]), fourcc, fps, (size, size))
            sp["written"] = 0
            active.append(sp)
        for sp in active:
            x1, y1, x2, y2 = sp["crop"]
            sp["writer"].write(cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA))
            sp["written"] += 1
        for sp in [sp for sp in active if sp["written"] >= clip_len]:
            sp.pop("writer").release()
            active.remove(sp)
        f += 1
        if f % PROGRESS_EVERY == 0:
            progress.put(PROGRESS_EVERY)
    progress.put(f % PROGRESS_EVERY)
    for sp in active:
        sp.pop("writer").release()
    cap.release()
    return specs


def build_clips(cfg):
    d, out = cfg["data"], cfg["paths"]["clips"]
    os.makedirs(out, exist_ok=True)
    json.dump(dict(clip_len=d["clip_len"], size=d["clip_size"], fps=d["fps"], event_stride=d["event_stride"]),
              open(os.path.join(out, "clips_info.json"), "w"), indent=1)
    sessions = session_names(cfg["paths"]["raw_data"])
    plans = [(s, plan_session(cfg, s)) for s in tqdm(sessions, desc="plan", unit="sess")]
    total = sum(max(sp["start"] for sp in specs) + d["clip_len"] for _, specs in plans)
    progress = Manager().Queue()
    jobs = [(s, specs, os.path.join(cfg["paths"]["videos"], f"{s}.mp4"), out, d["clip_len"], d["clip_size"],
             d["crop_pad"], d["fps"], progress) for s, specs in plans]
    with Pool(len(jobs)) as pool, tqdm(total=total, desc="render", unit="fr", unit_scale=True, mininterval=2) as bar:
        job = pool.map_async(render_session, jobs)
        while not job.ready() or not progress.empty():
            try:
                bar.update(progress.get(timeout=1))
            except queue.Empty:
                pass
        rows = [sp for specs in job.get() for sp in specs]

    with open(os.path.join(out, "classes.txt"), "w") as f:
        f.writelines(f"{i} {c}\n" for i, c in enumerate(d["classes"]))
    for split in ("train", "val"):
        with open(os.path.join(out, f"{split}.csv"), "w") as f:
            f.writelines(f"{r['path']} {r['label']}\n" for r in rows if r["split"] == split)
    cols = ["path", "split", "label", "cls", "orig_label", "sess", "pid", "related", "subtype", "quality", "kind",
            "src_id", "start", "cover", "written", "crop"]
    with open(os.path.join(out, "meta.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows({k: r.get(k) for k in cols} for r in rows)
    print(f"clips: {len(rows)}  incomplete: {sum(r.get('written') != d['clip_len'] for r in rows)}")
    for split in ("train", "val"):
        c = Counter(r["cls"] for r in rows if r["split"] == split)
        print(f"  {split}: " + ", ".join(f"{k} {c[k]}" for k in d["classes"]))


def main(argv=None):
    args = config_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config, args.set)
    extract_videos(cfg)
    build_clips(cfg)


if __name__ == "__main__":
    main()
