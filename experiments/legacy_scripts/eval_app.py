"""End-to-end check of infer_app.py on full, untrimmed val videos against the CVAT ground truth.

Unlike the clip metrics (hand-cut clips, annotated boxes), this runs the real pipeline: YOLO + ByteTrack boxes,
sliding 1 s windows over the whole video, smoothing and alert hysteresis.

Reports
  window level : every classifier window matched to an annotated person (IoU of mapped box vs CVAT box)
  event level  : share of ground-truth cheating events that raised an alert on that person
  false alarms : alert episodes that overlap no cheating event of that person, per person-hour

  python eval_app.py --ckpt /workspace/runs/<x3d run>/best.pt --sessions S07 S08
"""
import argparse
import collections
import glob
import json
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_clips import CLASSES as GT_CLASSES, LABEL_MAP, load_tracks  # noqa: E402
from infer_app import CheatDetector  # noqa: E402

VID_DIR = "/workspace/data/videos"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def gt_for_session(sess):
    n, tracks = load_tracks(sess)
    persons = {t["attrs"]["participant_id"]: t for t in tracks if t["label"] == "person"}
    events = collections.defaultdict(list)  # pid -> [(start, end, label)], interactions also on the related actor
    for e in tracks:
        if e["label"] not in LABEL_MAP:
            continue
        lab = LABEL_MAP[e["label"]]
        a = e["attrs"]
        events[a["participant_id"]].append((e["start"], e["end"], lab))
        if a.get("related_actor") not in (None, "__undefined__", a["participant_id"]):
            events[a["related_actor"]].append((e["start"], e["end"], lab))
    return n, persons, events


def window_label(evs, f0, f1):
    """Label of the window [f0, f1]: event covering >= half of the window (or of the event), else normal."""
    best, lab = 0, 0
    for s, e, l in evs:
        ov = min(e, f1) - max(s, f0) + 1
        if ov > 0 and ov >= 0.5 * min(f1 - f0 + 1, e - s + 1) and ov > best:
            best, lab = ov, l
    return lab


def run_session(det, sess, fps=25.0):
    det.reset()
    cap = cv2.VideoCapture(f"{VID_DIR}/{sess}.mp4")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in tqdm(range(n), desc=sess[:3], unit="fr", mininterval=2):
        ok, frame = cap.read()
        if not ok:
            break
        det.process(frame, f / fps)
    cap.release()
    return det.log, det.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sessions", nargs="+", default=["S07", "S08"])
    ap.add_argument("--det", default="yolo11m.pt")
    ap.add_argument("--thr-on", type=float, default=0.6)
    ap.add_argument("--thr-off", type=float, default=0.45)
    ap.add_argument("--out", default="/workspace/app_eval")
    ap.add_argument("--det-imgsz", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det = CheatDetector(args.ckpt, args.det, det_imgsz=args.det_imgsz, thr_on=args.thr_on, thr_off=args.thr_off)
    classes = GT_CLASSES
    same_classes = det.classes == GT_CLASSES  # per-class window metrics only make sense then
    y_true, y_pred, scores, alerts = [], [], [], []
    ev_total, ev_hit = collections.Counter(), collections.Counter()
    false_alarms, person_seconds, match_ious = 0, 0.0, []
    for short in args.sessions:
        sess = os.path.basename(glob.glob(f"{VID_DIR}/{short}_*.mp4")[0])[:-4]
        n, persons, events = gt_for_session(sess)
        log, found = run_session(det, sess)
        with open(f"{args.out}/{short}_log.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in log)
        json.dump(found, open(f"{args.out}/{short}_events.json", "w"), indent=1)
        person_seconds += len(persons) * n / 25.0

        track_pid = {}  # track id -> annotated participant (majority vote over its windows)
        votes = collections.defaultdict(collections.Counter)
        for r in log:
            f = int(round(r["t"] * 25))
            cand = [(iou(r["box"], p["boxes"][f]), pid) for pid, p in persons.items() if f in p["boxes"]]
            if cand:
                v, pid = max(cand)
                if v >= 0.3:
                    votes[r["track"]][pid] += 1
                    match_ious.append(v)
        track_pid = {k: c.most_common(1)[0][0] for k, c in votes.items()}

        alert_frames = collections.defaultdict(list)  # pid -> frames of alert windows
        for r in log:
            pid = track_pid.get(r["track"])
            if pid is None:
                continue
            f1 = int(round(r["t"] * 25))
            f0 = f1 - 75
            y_true.append(window_label(events[pid], f0, f1))
            p = np.array(r["probs"])
            y_pred.append(int(p.argmax()))
            scores.append(r["cheat_score"])
            alerts.append(r["alert"])
            if r["alert"]:
                alert_frames[pid].append(f1)

        for pid, evs in events.items():
            for s, e, lab in evs:
                if lab == 0:
                    continue
                ev_total[GT_CLASSES[lab]] += 1
                if any(s <= f <= e + 75 for f in alert_frames.get(pid, [])):
                    ev_hit[GT_CLASSES[lab]] += 1
        for e in found:  # alert episodes overlapping no cheating event of that person
            pid = track_pid.get(e["track"])
            f0, f1 = int(e["start"] * 25) - 75, int(e["end"] * 25)
            if pid is None or not any(l > 0 and s <= f1 and e_ >= f0 for s, e_, l in events[pid]):
                false_alarms += 1

    y_true, y_pred, scores, alerts = map(np.array, (y_true, y_pred, scores, alerts))
    if not same_classes:  # other label set (e.g. ver 2 normal / abnormal): collapse to cheating vs normal
        y_pred = (y_pred > 0).astype(int)
        y_true_eval, classes = (y_true > 0).astype(int), ["normal", "cheating"]
    else:
        y_true_eval = y_true
    prec, rec, f1, sup = precision_recall_fscore_support(y_true_eval, y_pred, labels=range(len(classes)),
                                                         zero_division=0)
    cheat = y_true > 0
    res = dict(
        windows=int(len(y_true)), box_match_iou_median=float(np.median(match_ious)),
        window_macro_f1=float(f1.mean()), window_f1=dict(zip(classes, f1.round(3).tolist())),
        window_recall=dict(zip(classes, rec.round(3).tolist())), window_support=dict(zip(classes, sup.tolist())),
        cheat_auroc=float(roc_auc_score(cheat, scores)),
        alert_precision=float((alerts & cheat).sum() / max(1, alerts.sum())),
        alert_recall=float((alerts & cheat).sum() / max(1, cheat.sum())),
        event_recall={c: f"{ev_hit[c]}/{ev_total[c]}" for c in ev_total},
        model_classes=det.classes,
        event_recall_total=float(sum(ev_hit.values()) / max(1, sum(ev_total.values()))),
        false_alarm_episodes=false_alarms,
        false_alarms_per_person_hour=float(false_alarms / (person_seconds / 3600)),
        confusion=confusion_matrix(y_true_eval, y_pred, labels=range(len(classes))).tolist(),
        thr_on=args.thr_on, thr_off=args.thr_off,
    )
    json.dump(res, open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=1))
    print("confusion (rows=true):")
    for c, r in zip(classes, res["confusion"]):
        print(f"  {c:17s} {r}")


if __name__ == "__main__":
    main()
