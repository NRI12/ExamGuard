"""Inspect CVAT exam-cheating dataset and render sample visuals for X3D / TSM readiness."""
import collections
import glob
import json
import os
import random
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = "/workspace/data/DOAN1/annotations"
VID_DIR = "/workspace/data/videos"
OUT = "/workspace/data/inspect"
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
random.seed(0)

EVENT_CLASSES = ["suspicious_looking", "communicating", "exchange_object",
                 "using_phone/cheat_sheet", "hard_negative"]
X3D_FRAMES, X3D_STRIDE = 16, 5   # X3D-M default: 16 frames x stride 5 = 80 frames = 3.2 s @25fps
TSM_SEGMENTS = 8


def extract_videos():
    for z in tqdm(sorted(glob.glob(f"{ROOT}/backup/*_task_backup.zip")), desc="extract videos", unit="video"):
        sess = os.path.basename(z).split("_task_backup")[0]
        dst = f"{VID_DIR}/{sess}.mp4"
        if os.path.exists(dst):
            continue
        with zipfile.ZipFile(z) as zf:
            mp4 = [n for n in zf.namelist() if n.endswith(".mp4")][0]
            with zf.open(mp4) as src, open(dst, "wb") as f:
                while chunk := src.read(1 << 24):
                    f.write(chunk)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,codec_name,duration",
                        "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(w=s["width"], h=s["height"], fps=num / den, frames=int(s["nb_read_packets"]),
                codec=s["codec_name"], dur=float(s.get("duration", 0)))


def load_xml(sess):
    z = glob.glob(f"{ROOT}/cvat_zip/{sess}_cvat_v1.zip")[0]
    with zipfile.ZipFile(z) as zf:
        root = ET.fromstring(zf.read("annotations.xml"))
    size = int(root.findtext("meta/job/size") or root.findtext("meta/task/size"))
    tracks = []
    for t in root.iter("track"):
        boxes = {}
        attrs = {}
        for b in t.iter("box"):
            if b.get("outside") == "1":
                continue
            fr = int(b.get("frame"))
            boxes[fr] = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            if not attrs:
                attrs = {a.get("name"): a.text for a in b.iter("attribute")}
        if boxes:
            tracks.append(dict(id=int(t.get("id")), label=t.get("label"), boxes=boxes, attrs=attrs,
                               start=min(boxes), end=max(boxes)))
    return size, tracks


def box_at(track, fr):
    ks = sorted(track["boxes"])
    k = min(ks, key=lambda x: abs(x - fr))
    return track["boxes"][k]


def read_frames(cap, frames):
    out = []
    for fr in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        out.append(img if ok else None)
    return out


def crop(img, box, pad=0.15, size=224):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    c = img[y1:y2, x1:x2]
    return cv2.resize(c, (size, size)) if c.size else np.zeros((size, size, 3), np.uint8)


def main():
    extract_videos()
    sessions = sorted(os.path.basename(z).split("_cvat_v1")[0] for z in glob.glob(f"{ROOT}/cvat_zip/*.zip"))
    report = {"sessions": {}, "issues": []}
    events = []
    for sess in tqdm(sessions, desc="parse + probe", unit="sess"):
        size, tracks = load_xml(sess)
        vpath = f"{VID_DIR}/{sess}.mp4"
        info = probe(vpath) if os.path.exists(vpath) else None
        persons = {t["attrs"].get("participant_id"): t for t in tracks if t["label"] == "person"}
        ev = [t for t in tracks if t["label"] != "person"]
        report["sessions"][sess] = dict(xml_frames=size, video=info, persons=len(persons), events=len(ev))
        if info and info["frames"] != size:
            report["issues"].append(f"{sess}: video has {info['frames']} frames but XML says {size}")
        for t in ev:
            t["sess"] = sess
            t["dur"] = t["end"] - t["start"] + 1
            t["gaps"] = t["dur"] - len(t["boxes"])
            events.append(t)
            pid = t["attrs"].get("participant_id")
            if pid not in persons:
                report["issues"].append(f"{sess} track {t['id']}: participant {pid} has no person track")
            if t["dur"] < X3D_FRAMES * X3D_STRIDE:
                pass
        # overlapping events for same participant
        by_pid = collections.defaultdict(list)
        for t in ev:
            by_pid[t["attrs"].get("participant_id")].append(t)
        for pid, ts in by_pid.items():
            ts.sort(key=lambda t: t["start"])
            for a, b in zip(ts, ts[1:]):
                if b["start"] <= a["end"]:
                    report["issues"].append(f"{sess} {pid}: overlap {a['label']}#{a['id']} [{a['start']}-{a['end']}] "
                                            f"& {b['label']}#{b['id']} [{b['start']}-{b['end']}]")

    # ---- stats
    fps = 25.0
    cls_cnt = collections.Counter(e["label"] for e in events)
    sub_cnt = collections.Counter((e["label"], e["attrs"].get("subtype")) for e in events)
    q_cnt = collections.Counter(e["attrs"].get("quality") for e in events)
    durs = {c: [e["dur"] / fps for e in events if e["label"] == c] for c in EVENT_CLASSES}
    total_frames = sum(s["xml_frames"] for s in report["sessions"].values())
    pos_frames = sum(e["dur"] for e in events if e["label"] != "hard_negative")
    report.update(class_counts=cls_cnt, quality=q_cnt,
                  subtypes={f"{k[0]}::{k[1]}": v for k, v in sub_cnt.items()},
                  duration_s={c: dict(n=len(v), min=round(min(v), 2) if v else None,
                                      median=round(float(np.median(v)), 2) if v else None,
                                      max=round(max(v), 2) if v else None,
                                      shorter_than_x3d_window=sum(d < X3D_FRAMES * X3D_STRIDE / fps for d in v))
                              for c, v in durs.items()},
                  total_hours=round(total_frames / fps / 3600, 2),
                  positive_frame_ratio_per_person=round(pos_frames / (total_frames * 6), 4),
                  gap_tracks=sum(1 for e in events if e["gaps"] > 0))
    for e in events:
        if e["attrs"].get("quality") not in ("good", "acceptable"):
            report["issues"].append(f"{e['sess']} track {e['id']} {e['label']}: quality={e['attrs'].get('quality')}")
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=1, default=str)

    # ---- charts
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].barh([c for c in EVENT_CLASSES], [cls_cnt[c] for c in EVENT_CLASSES], color="#4a7fb0")
    ax[0].set_title("Event tracks per class (all sessions)")
    for i, c in enumerate(EVENT_CLASSES):
        ax[0].text(cls_cnt[c], i, f" {cls_cnt[c]}", va="center")
    ax[1].boxplot([durs[c] for c in EVENT_CLASSES], vert=False, tick_labels=[c.split("/")[0] for c in EVENT_CLASSES])
    ax[1].axvline(X3D_FRAMES * X3D_STRIDE / fps, color="r", ls="--", label="X3D-M window 3.2s (16x5)")
    ax[1].axvline(X3D_FRAMES * 2 / fps, color="orange", ls=":", label="X3D-S window 1.3s (13x6≈)")
    ax[1].set_xlabel("event duration (s)"); ax[1].legend(fontsize=8); ax[1].set_title("Event duration")
    subs = sorted(sub_cnt.items(), key=lambda kv: (kv[0][0], -kv[1]))
    ax[2].barh([f"{k[0].split('/')[0][:12]}:{k[1]}" for k, _ in subs], [v for _, v in subs], color="#7a9e5b")
    ax[2].set_title("Subtype counts"); ax[2].tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/stats.png", dpi=110); plt.close()

    # ---- timelines
    fig, axes = plt.subplots(len(sessions), 1, figsize=(18, 1.6 * len(sessions)), sharex=False)
    colors = dict(zip(EVENT_CLASSES, ["#d63a13", "#2f7ed8", "#8e44ad", "#e6a100", "#7f8c8d"]))
    for axx, sess in zip(np.atleast_1d(axes), sessions):
        for e in [e for e in events if e["sess"] == sess]:
            p = int((e["attrs"].get("participant_id") or "P0")[1:])
            axx.barh(p, e["dur"] / fps, left=e["start"] / fps, color=colors[e["label"]], height=0.8)
        axx.set_ylim(0.3, 6.7); axx.set_yticks(range(1, 7)); axx.set_ylabel(sess[:3]); axx.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, EVENT_CLASSES, loc="upper center", ncol=5)
    axes[-1].set_xlabel("time (s)"); plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"{OUT}/timelines.png", dpi=100); plt.close()

    # ---- overview frame per session (person boxes)
    for sess in tqdm(sessions, desc="overview frames", unit="sess"):
        vpath = f"{VID_DIR}/{sess}.mp4"
        if not os.path.exists(vpath):
            continue
        _, tracks = load_xml(sess)
        cap = cv2.VideoCapture(vpath)
        img = read_frames(cap, [100])[0]
        cap.release()
        for t in tracks:
            if t["label"] != "person" or 100 not in t["boxes"]:
                continue
            x1, y1, x2, y2 = map(int, t["boxes"][100])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{t['attrs'].get('participant_id')}/{t['attrs'].get('seat_id')}", (x1 + 5, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.imwrite(f"{OUT}/overview_{sess[:3]}.jpg", cv2.resize(img, (960, 540)))

    # ---- clip strips: what X3D (16 frames, stride 5) and TSM (8 segments) would see
    samples = []
    for c in EVENT_CLASSES:
        pool = [e for e in events if e["label"] == c]
        samples += random.sample(pool, min(3, len(pool)))
    strips = []
    caps = {}
    for i, e in enumerate(tqdm(samples, desc="clip strips", unit="clip")):
        vpath = f"{VID_DIR}/{e['sess']}.mp4"
        if not os.path.exists(vpath):
            continue
        cap = caps.setdefault(e["sess"], cv2.VideoCapture(vpath))
        mid = (e["start"] + e["end"]) // 2
        x3d = [min(max(0, mid + (k - X3D_FRAMES // 2) * X3D_STRIDE), e["end"] + 200) for k in range(X3D_FRAMES)]
        seg = np.linspace(e["start"], e["end"], TSM_SEGMENTS + 1)
        tsm = [int((seg[k] + seg[k + 1]) / 2) for k in range(TSM_SEGMENTS)]
        rows = []
        for name, frs in (("X3D", x3d[::2]), ("TSM", tsm)):
            imgs = read_frames(cap, frs)
            tiles = []
            for fr, img in zip(frs, imgs):
                if img is None:
                    tiles.append(np.zeros((160, 160, 3), np.uint8)); continue
                tile = crop(img, box_at(e, fr), size=160)
                inside = e["start"] <= fr <= e["end"]
                cv2.rectangle(tile, (0, 0), (159, 159), (0, 200, 0) if inside else (0, 0, 255), 4)
                cv2.putText(tile, str(fr), (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                tiles.append(tile)
            row = np.hstack(tiles)
            cv2.putText(row, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rows.append(row)
        head = np.zeros((34, rows[0].shape[1], 3), np.uint8)
        a = e["attrs"]
        txt = (f"{e['sess'][:3]} #{e['id']} {e['label']} / {a.get('subtype')} / {a.get('participant_id')}"
               f"{'->' + a['related_actor'] if a.get('related_actor') not in (None, '__undefined__') else ''}"
               f" / q={a.get('quality')} / {e['dur'] / fps:.1f}s")
        cv2.putText(head, txt, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        strip = np.vstack([head] + rows)
        cv2.imwrite(f"{OUT}/clip_{i:02d}_{e['label'].replace('/', '-')}.jpg", strip)
        strips.append(strip)
        # short mp4 of the crop (full event, 2x downsample in time) for playback
        vw = cv2.VideoWriter(f"{OUT}/clip_{i:02d}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (224, 224))
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["start"])
        for fr in range(e["start"], min(e["end"], e["start"] + 150) + 1):
            ok, img = cap.read()
            if not ok:
                break
            if (fr - e["start"]) % 2 == 0:
                vw.write(crop(img, box_at(e, fr)))
        vw.release()
    for cap in caps.values():
        cap.release()
    print(json.dumps({k: v for k, v in report.items() if k != "issues"}, indent=1, default=str))
    print("ISSUES:", len(report["issues"]))
    print("\n".join(report["issues"][:60]))


if __name__ == "__main__":
    main()
