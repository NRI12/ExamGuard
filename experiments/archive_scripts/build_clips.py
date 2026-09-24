"""Convert CVAT track annotations + long videos into trimmed per-person clips for X3D / TSM.

Output layout (/workspace/data/clips):
  videos/{split}/{class}/{sess}_{pid}_{kind}{id}_{start}.mp4   OUT_SIZE^2, 25 fps, CLIP_LEN frames
  train.csv / val.csv     "relative/path.mp4 label_id"   (PySlowFast / mmaction2 / pytorchvideo format)
  meta.csv                full provenance of every clip
  classes.txt             label_id -> name
  clips_info.json         clip_len / size, read by the training scripts

  python build_clips.py                                        # 256px, 80 frames -> clips/
  python build_clips.py --out /workspace/data/clips_hr --size 356 --clip-len 100
"""
import argparse
import csv
import glob
import json
import os
import queue
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from tqdm import tqdm

ROOT = "/workspace/data/DOAN1/annotations"
VID_DIR = "/workspace/data/videos"
OUT = "/workspace/data/clips"

CLASSES = ["normal", "looking", "interaction", "phone_cheatsheet"]
# hard_negative is non-cheating -> normal (kept identifiable via kind="h" in meta.csv for oversampling)
LABEL_MAP = {"hard_negative": 0, "suspicious_looking": 1, "communicating": 2, "exchange_object": 2,
             "using_phone/cheat_sheet": 3}
VAL_SESSIONS = ("S07", "S08")

CLIP_LEN = 80          # 3.2 s @ 25 fps: X3D-M 16x5, TSM 8 segments x 10
EVENT_STRIDE = 40      # sliding-window step inside long events
NORMAL_MARGIN = 50     # frames kept away from any event of that person
NORMAL_PER_PERSON_MIN = 1.5  # normal clips sampled per person per minute of video
OUT_SIZE = 256
PAD = 0.15
SEED = 0
PROGRESS_EVERY = 250   # frames between progress reports from each worker


def load_tracks(sess):
    with zipfile.ZipFile(glob.glob(f"{ROOT}/cvat_zip/{sess}_cvat_v1.zip")[0]) as zf:
        root = ET.fromstring(zf.read("annotations.xml"))
    n_frames = int(root.findtext("meta/job/size") or root.findtext("meta/task/size"))
    tracks = []
    for t in root.iter("track"):
        boxes, attrs = {}, {}
        for b in t.iter("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = np.array([float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
            if not attrs:
                attrs = {a.get("name"): a.text for a in b.iter("attribute")}
        if boxes:
            tracks.append(dict(id=int(t.get("id")), label=t.get("label"), boxes=boxes, attrs=attrs,
                               start=min(boxes), end=max(boxes)))
    return n_frames, tracks


def box_at(track, f):
    if f in track["boxes"]:
        return track["boxes"][f]
    k = min(track["boxes"], key=lambda x: abs(x - f))
    return track["boxes"][k]


def square_crop_box(boxes, w, h):
    """One fixed square covering the union of boxes over the clip (no jitter)."""
    b = np.stack(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = max(x2 - x1, y2 - y1) * (1 + PAD)
    side = min(side, w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return x1, y1, x1 + int(side), y1 + int(side)


def event_windows(start, end, n_frames):
    dur = end - start + 1
    if dur <= CLIP_LEN:
        s = start + dur // 2 - CLIP_LEN // 2
        s = int(np.clip(s, 0, n_frames - CLIP_LEN))
        return [s]  # short event: one window centred on it; coverage logged in meta.csv
    last = end - CLIP_LEN + 1
    starts = list(range(start, last + 1, EVENT_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def plan_session(sess):
    rng = random.Random(f"{SEED}-{sess}")
    n_frames, tracks = load_tracks(sess)
    persons = {t["attrs"]["participant_id"]: t for t in tracks if t["label"] == "person"}
    events = [t for t in tracks if t["label"] in LABEL_MAP]
    split = "val" if sess[:3] in VAL_SESSIONS else "train"
    specs = []

    busy = defaultdict(list)  # pid -> [(s, e)] frames where that person takes part in any event
    for e in events:
        a = e["attrs"]
        busy[a["participant_id"]].append((e["start"], e["end"]))
        if a.get("related_actor") not in (None, "__undefined__"):
            busy[a["related_actor"]].append((e["start"], e["end"]))

        # one clip per person: the actor, plus the related actor for pair interactions,
        # always cropped on that person's own track so inference only needs per-person crops
        label = LABEL_MAP[e["label"]]
        actors = [(a["participant_id"], "h" if e["label"] == "hard_negative" else "e")]
        if a.get("related_actor") not in (None, "__undefined__", a["participant_id"]):
            actors.append((a["related_actor"], "r"))
        for pid, kind in actors:
            if pid not in persons:
                continue
            for s in event_windows(e["start"], e["end"], n_frames):
                cover = (min(e["end"], s + CLIP_LEN - 1) - max(e["start"], s) + 1) / CLIP_LEN
                boxes = [box_at(persons[pid], f) for f in range(s, s + CLIP_LEN, 8)]
                specs.append(dict(sess=sess, split=split, label=label, cls=CLASSES[label], pid=pid,
                                  related=a.get("related_actor") or "", orig_label=e["label"],
                                  subtype=a.get("subtype") or "", quality=a.get("quality") or "",
                                  kind=kind, src_id=e["id"], start=s, boxes=boxes, cover=round(min(cover, 1.0), 3)))

    # normal clips: per person, windows far from any event involving that person
    n_normal = max(1, int(NORMAL_PER_PERSON_MIN * n_frames / 25 / 60))
    for pid, p in persons.items():
        blocked = [(s - NORMAL_MARGIN - CLIP_LEN, e + NORMAL_MARGIN) for s, e in busy[pid]]
        cands = [s for s in range(0, n_frames - CLIP_LEN, CLIP_LEN // 2)
                 if not any(a <= s <= b for a, b in blocked)]
        for s in sorted(rng.sample(cands, min(n_normal, len(cands)))):
            boxes = [box_at(p, f) for f in range(s, s + CLIP_LEN, 8)]
            specs.append(dict(sess=sess, split=split, label=0, cls="normal", pid=pid, related="",
                              orig_label="normal", subtype="",
                              quality="", kind="n", src_id=p["id"], start=s, boxes=boxes, cover=1.0))
    return specs


def render_session(args):
    sess, specs, progress = args
    cap = cv2.VideoCapture(f"{VID_DIR}/{sess}.mp4")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for sp in specs:
        sp["crop"] = square_crop_box(sp["boxes"], w, h)
        sp["path"] = f"videos/{sp['split']}/{sp['cls']}/{sess[:3]}_{sp['pid']}_{sp['kind']}{sp['src_id']}_{sp['start']:06d}.mp4"
        os.makedirs(os.path.dirname(f"{OUT}/{sp['path']}"), exist_ok=True)
    by_start = defaultdict(list)
    for sp in specs:
        by_start[sp["start"]].append(sp)
    active, fourcc = [], cv2.VideoWriter_fourcc(*"mp4v")
    last = max(sp["start"] for sp in specs) + CLIP_LEN
    f = 0
    while f < last:
        ok, img = cap.read()
        if not ok:
            break
        for sp in by_start.get(f, []):
            sp["writer"] = cv2.VideoWriter(f"{OUT}/{sp['path']}", fourcc, 25, (OUT_SIZE, OUT_SIZE))
            sp["written"] = 0
            active.append(sp)
        for sp in active:
            x1, y1, x2, y2 = sp["crop"]
            sp["writer"].write(cv2.resize(img[y1:y2, x1:x2], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA))
            sp["written"] += 1
        done = [sp for sp in active if sp["written"] >= CLIP_LEN]
        for sp in done:
            sp["writer"].release()
            del sp["writer"]
            active.remove(sp)
        f += 1
        if f % PROGRESS_EVERY == 0:
            progress.put(PROGRESS_EVERY)
    progress.put(f % PROGRESS_EVERY)
    for sp in active:
        sp["writer"].release()
        del sp["writer"]
    cap.release()
    for sp in specs:
        sp.pop("boxes")
    return specs


def main():
    global OUT, OUT_SIZE, CLIP_LEN, EVENT_STRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    ap.add_argument("--clip-len", type=int, default=CLIP_LEN)
    ap.add_argument("--event-stride", type=int, default=EVENT_STRIDE)
    a = ap.parse_args()
    # module globals are inherited by the forked render workers
    OUT, OUT_SIZE, CLIP_LEN, EVENT_STRIDE = a.out, a.size, a.clip_len, a.event_stride
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(clip_len=CLIP_LEN, size=OUT_SIZE, fps=25, event_stride=EVENT_STRIDE),
              open(f"{OUT}/clips_info.json", "w"), indent=1)
    sessions = sorted(os.path.basename(z).split("_cvat_v1")[0] for z in glob.glob(f"{ROOT}/cvat_zip/*.zip"))
    plans = [(s, plan_session(s)) for s in tqdm(sessions, desc="plan", unit="sess")]
    # workers decode sequentially up to the last clip end; report decoded frames to one shared bar
    total = sum(max(sp["start"] for sp in specs) + CLIP_LEN for _, specs in plans)
    progress = Manager().Queue()
    with Pool(len(plans)) as pool, tqdm(total=total, desc="render", unit="fr", unit_scale=True,
                                        mininterval=2) as bar:
        job = pool.map_async(render_session, [(s, specs, progress) for s, specs in plans])
        while not job.ready() or not progress.empty():
            try:
                bar.update(progress.get(timeout=1))
            except queue.Empty:
                pass
        results = job.get()
    rows = [sp for r in results for sp in r]

    with open(f"{OUT}/classes.txt", "w") as f:
        f.write("\n".join(f"{i} {c}" for i, c in enumerate(CLASSES)) + "\n")
    for split in ("train", "val"):
        with open(f"{OUT}/{split}.csv", "w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(f"{r['path']} {r['label']}\n")
    cols = ["path", "split", "label", "cls", "orig_label", "sess", "pid", "related", "subtype", "quality", "kind",
            "src_id", "start", "cover", "written", "crop"]
    with open(f"{OUT}/meta.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in cols})

    bad = [r for r in rows if r.get("written") != CLIP_LEN]
    print(f"clips: {len(rows)}  incomplete: {len(bad)}")
    for split in ("train", "val"):
        sub = [r for r in rows if r["split"] == split]
        print(split, "total", len(sub))
        for k in CLASSES:
            rk = [r for r in sub if r["cls"] == k]
            kinds = Counter(r["kind"] for r in rk)
            events = {(r["sess"], r["src_id"]) for r in rk if r["kind"] != "n"}
            print(f"  {k:17s} clips={len(rk):4d}  events={len(events):3d}  by_kind={dict(kinds)}")


if __name__ == "__main__":
    main()
