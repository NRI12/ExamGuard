"""Build the data of the realtime demo website (demo_web/) from the workspace results.

Reads (never imports) the projects' saved outputs:
  - DOAN1 CVAT exports + task backups         -> ground-truth person boxes and cheating events, source videos
  - eval-videos per-window logs of each model -> the model's decisions on the full S07 / S08 videos
and writes:
  media/<session>.mp4          720p H.264 (fast seeking, faststart)
  data/manifest.json           sessions, models, classes, default thresholds
  data/gt_<session>.json       person boxes every 0.2 s + ground-truth events
  data/<model>_<session>.json  per-track windows: time, box, class probabilities, matched participant

  python demo_web/build_demo.py            # ~5-10 min the first time (video transcoding), seconds afterwards
"""
import collections
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
RAW = os.path.join(WS, "cheating_detection_v1", "data", "raw", "DOAN1", "annotations")
FPS, SAMPLE = 25, 5          # videos are 25 fps; ground-truth boxes are kept every 5 frames (0.2 s)
OUT_W, OUT_H = 1280, 720
SESSIONS = {"S07": "S07_6P_1080p_25fps", "S08": "S08_6P_1080p_25fps"}
CLASSES4 = ["normal", "looking", "interaction", "phone_cheatsheet"]
GT_CLASS = {"hard_negative": "normal", "suspicious_looking": "looking", "communicating": "interaction",
            "exchange_object": "interaction", "using_phone/cheat_sheet": "phone_cheatsheet"}
MODELS = [
    dict(id="v1", name="v1  X3D-L, 4 classes", classes=CLASSES4, recommended=True,
         logs="cheating_detection_v1/outputs/reference/x3d_l/app_eval", thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Recommended model. RGB X3D-L on a 3.2 s per-person window, 4 classes."),
    dict(id="v2", name="v2  X3D-L, normal / abnormal", classes=["normal", "abnormal"],
         logs="cheating_detection_v2/outputs/reference/x3d_l/app_eval", thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Same network trained on 2 classes: flags behaviour without naming its type."),
    dict(id="v3", name="v3  RGB + optical flow, fine-tuned fusion", classes=CLASSES4,
         logs="cheating_detection_v3/outputs/reference/fusion_trained/app_eval", thr_on=0.85, thr_off=0.70, smooth=3,
         min_windows=2, note="Two-stream model: X3D-L on RGB plus X3D-S on RAFT optical flow."),
    dict(id="v3late", name="v3  RGB + optical flow, late fusion", classes=CLASSES4,
         logs="cheating_detection_v3/outputs/reference/late_fusion/app_eval", thr_on=0.70, thr_off=0.55, smooth=3,
         min_windows=3, note="Average of the RGB and flow streams. Only S07 is available (S08 log lost with the old disk)."),
    dict(id="v1a", name="v1 variant A  (interaction labels)", classes=CLASSES4,
         logs="cheating_detection_v1/outputs/variant_a/outputs/app_eval_a", thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Experiment: passive partner of whisper / tap events not labelled. Worse than v1 on interaction."),
]


def cvat_tracks(session):
    with zipfile.ZipFile(os.path.join(RAW, "cvat_zip", f"{session}_cvat_v1.zip")) as zf:
        root = ET.fromstring(zf.read("annotations.xml"))
    n = int(root.findtext("meta/job/size") or root.findtext("meta/task/size"))
    tracks = []
    for t in root.iter("track"):
        boxes, attrs = {}, {}
        for b in t.iter("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            attrs = attrs or {a.get("name"): a.text for a in b.iter("attribute")}
        if boxes:
            tracks.append(dict(label=t.get("label"), boxes=boxes, attrs=attrs, start=min(boxes), end=max(boxes)))
    return n, tracks


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def build_gt(short, session):
    n, tracks = cvat_tracks(session)
    persons = {t["attrs"]["participant_id"]: t for t in tracks if t["label"] == "person"}
    gt_persons = {pid: [[round(f / FPS, 2)] + [round(v, 1) for v in p["boxes"][f]]
                        for f in range(0, n, SAMPLE) if f in p["boxes"]] for pid, p in sorted(persons.items())}
    events = []
    for t in sorted((t for t in tracks if t["label"] in GT_CLASS), key=lambda t: t["start"]):
        a = t["attrs"]
        rel = a.get("related_actor")
        events.append(dict(pid=a["participant_id"], related=None if rel in (None, "__undefined__") else rel,
                           label=t["label"], cls=GT_CLASS[t["label"]], subtype=a.get("subtype") or "",
                           cheating=t["label"] != "hard_negative",
                           start=round(t["start"] / FPS, 2), end=round(t["end"] / FPS, 2)))
    gt = dict(session=short, frames=n, duration=round(n / FPS, 2), fps=FPS, width=1920, height=1080,
              persons=gt_persons, events=events)
    json.dump(gt, open(os.path.join(HERE, "data", f"gt_{short}.json"), "w"), separators=(",", ":"))
    return persons, gt


def build_model(model, short, persons):
    path = os.path.join(WS, model["logs"], f"{short}_log.jsonl")
    if not os.path.exists(path):
        return False
    rows = [json.loads(line) for line in open(path)]
    votes = collections.defaultdict(collections.Counter)
    for r in rows:
        f = int(round(r["t"] * FPS))
        cand = [(iou(r["box"], p["boxes"][f]), pid) for pid, p in persons.items() if f in p["boxes"]]
        if cand and max(cand)[0] >= 0.3:
            votes[r["track"]][max(cand)[1]] += 1
    tracks = collections.defaultdict(lambda: dict(t=[], box=[], p=[]))
    for r in sorted(rows, key=lambda r: (r["track"], r["t"])):
        tr = tracks[r["track"]]
        tr["t"].append(round(r["t"], 2))
        tr["box"].append([round(v, 1) for v in r["box"]])
        tr["p"].append([round(v, 3) for v in r["probs"]])
    out = {str(k): dict(pid=votes[k].most_common(1)[0][0] if votes[k] else None, **v) for k, v in tracks.items()}
    json.dump(dict(model=model["id"], session=short, tracks=out),
              open(os.path.join(HERE, "data", f"{model['id']}_{short}.json"), "w"), separators=(",", ":"))
    return True


def transcode(short, session):
    dst = os.path.join(HERE, "media", f"{short}.mp4")
    if os.path.exists(dst):
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, f"{session}.mp4")
        with zipfile.ZipFile(os.path.join(RAW, "backup", f"{session}_task_backup.zip")) as zf:
            with zf.open(next(n for n in zf.namelist() if n.endswith(".mp4"))) as s, open(src, "wb") as f:
                shutil.copyfileobj(s, f, 1 << 24)
        print(f"transcoding {short} to {OUT_W}x{OUT_H} ...", flush=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-stats", "-y", "-i", src, "-vf", f"scale={OUT_W}:{OUT_H}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "50", "-pix_fmt", "yuv420p",
                        "-an", "-movflags", "+faststart", dst + ".part.mp4"], check=True)
        os.replace(dst + ".part.mp4", dst)


def main():
    for d in ("data", "media"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    manifest = dict(sessions=[], models=[], video_size=[OUT_W, OUT_H])
    available = collections.defaultdict(list)
    for short, session in SESSIONS.items():
        persons, gt = build_gt(short, session)
        for m in MODELS:
            if build_model(m, short, persons):
                available[m["id"]].append(short)
        transcode(short, session)
        manifest["sessions"].append(dict(id=short, video=f"media/{short}.mp4", duration=gt["duration"],
                                         persons=len(persons), events=sum(e["cheating"] for e in gt["events"])))
        print(f"{short}: {len(persons)} persons, {len(gt['events'])} ground-truth events", flush=True)
    for m in MODELS:
        if available[m["id"]]:
            manifest["models"].append({k: v for k, v in m.items() if k != "logs"} | dict(sessions=available[m["id"]]))
    json.dump(manifest, open(os.path.join(HERE, "data", "manifest.json"), "w"), indent=1)
    print("models:", ", ".join(f"{m['id']} ({'/'.join(m['sessions'])})" for m in manifest["models"]))
    print("done. Run: python demo_web/serve.py")


if __name__ == "__main__":
    main()
