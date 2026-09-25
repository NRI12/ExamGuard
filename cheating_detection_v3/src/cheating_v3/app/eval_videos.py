"""End-to-end evaluation of the detector on full, untrimmed videos against the CVAT ground truth.

Window level : every classifier window matched to an annotated person (IoU of mapped box vs CVAT box)
Event level  : share of ground-truth cheating events that raised an alert on that person
False alarms : alert episodes overlapping no cheating event of that person, per person-hour
Logs are saved so `tune` can replay other thresholds without decoding video again.

  python -m cheating_v3 eval-videos [--sessions S07 S08]
"""
import collections
import glob
import json
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from ..config import config_parser, load_config
from ..data.cvat import load_tracks, related_actor
from .detector import CheatDetector


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def session_file(cfg, short):
    return os.path.basename(glob.glob(os.path.join(cfg["paths"]["videos"], f"{short}_*.mp4"))[0])[:-4]


def gt_for_session(cfg, sess):
    """(n_frames, persons{pid: track}, events{pid: [(start, end, label)]}); interactions count for both people."""
    n, tracks = load_tracks(cfg["paths"]["raw_data"], sess)
    persons = {t["attrs"]["participant_id"]: t for t in tracks if t["label"] == "person"}
    events = collections.defaultdict(list)
    for e in tracks:
        if e["label"] in cfg["data"]["label_map"]:
            lab = cfg["data"]["label_map"][e["label"]]
            events[e["attrs"]["participant_id"]].append((e["start"], e["end"], lab))
            if related_actor(e["attrs"]):
                events[related_actor(e["attrs"])].append((e["start"], e["end"], lab))
    return n, persons, events


def match_tracks(log, persons, fps):
    """track id -> annotated participant, by majority vote of IoU >= 0.3 matches."""
    votes, ious = collections.defaultdict(collections.Counter), []
    for r in log:
        f = int(round(r["t"] * fps))
        cand = [(iou(r["box"], p["boxes"][f]), pid) for pid, p in persons.items() if f in p["boxes"]]
        if cand and max(cand)[0] >= 0.3:
            votes[r["track"]][max(cand)[1]] += 1
            ious.append(max(cand)[0])
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}, ious


def window_label(evs, f0, f1):
    """Event covering >= half of the window (or of the event) -> its label, else 0."""
    best, lab = 0, 0
    for s, e, l in evs:
        ov = min(e, f1) - max(s, f0) + 1
        if ov > 0 and ov >= 0.5 * min(f1 - f0 + 1, e - s + 1) and ov > best:
            best, lab = ov, l
    return lab


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--checkpoint", help="default: paths.weights")
    p.add_argument("--sessions", nargs="+", help="default: data.val_sessions")
    p.add_argument("--out", help="default: outputs/app_eval")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out = args.out or os.path.join(cfg["paths"]["outputs"], "app_eval")
    os.makedirs(out, exist_ok=True)
    fps = cfg["data"]["fps"]
    det = CheatDetector(cfg, args.checkpoint)
    classes = det.classes
    win = int(round((det.n_frames - 1) * det.sample_dt * fps))  # window span in frames (75 = 3 s)
    y_true, y_pred, scores, alerts = [], [], [], []
    ev_total, ev_hit = collections.Counter(), collections.Counter()
    false_alarms, person_s, match_ious = 0, 0.0, []
    for short in args.sessions or cfg["data"]["val_sessions"]:
        sess = session_file(cfg, short)
        n, persons, events = gt_for_session(cfg, sess)
        det.reset()
        cap = cv2.VideoCapture(os.path.join(cfg["paths"]["videos"], f"{sess}.mp4"))
        for f in tqdm(range(n), desc=short, unit="fr", mininterval=2):
            ok, frame = cap.read()
            if not ok:
                break
            det.process(frame, f / fps)
        cap.release()
        found = det.finish()
        with open(os.path.join(out, f"{short}_log.jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in det.log)
        json.dump(found, open(os.path.join(out, f"{short}_events.json"), "w"), indent=1)
        person_s += len(persons) * n / fps
        track_pid, ious = match_tracks(det.log, persons, fps)
        match_ious += ious
        alert_frames = collections.defaultdict(list)
        for r in det.log:
            pid = track_pid.get(r["track"])
            if pid is None:
                continue
            f1 = int(round(r["t"] * fps))
            y_true.append(window_label(events[pid], f1 - win, f1))
            y_pred.append(int(np.argmax(r["probs"])))
            scores.append(r["cheat_score"])
            alerts.append(r["alert"])
            if r["alert"]:
                alert_frames[pid].append(f1)
        for pid, evs in events.items():
            for s, e, lab in evs:
                if lab:
                    ev_total[classes[lab]] += 1
                    ev_hit[classes[lab]] += any(s <= f <= e + win for f in alert_frames.get(pid, []))
        for e in found:
            pid = track_pid.get(e["track"])
            f0, f1 = int(e["start"] * fps) - win, int(e["end"] * fps)
            if pid is None or not any(l and s <= f1 and e_ >= f0 for s, e_, l in events[pid]):
                false_alarms += 1

    y_true, y_pred, scores, alerts = map(np.array, (y_true, y_pred, scores, alerts))
    prec, rec, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    cheat = y_true > 0
    res = dict(
        windows=int(len(y_true)), box_match_iou_median=float(np.median(match_ious)),
        window_macro_f1=float(f1s.mean()), window_f1=dict(zip(classes, f1s.round(3).tolist())),
        window_recall=dict(zip(classes, rec.round(3).tolist())), window_support=dict(zip(classes, sup.tolist())),
        cheat_auroc=float(roc_auc_score(cheat, scores)),
        alert_precision=float((alerts & cheat).sum() / max(1, alerts.sum())),
        alert_recall=float((alerts & cheat).sum() / max(1, cheat.sum())),
        event_recall={c: f"{ev_hit[c]}/{ev_total[c]}" for c in ev_total},
        event_recall_total=float(sum(ev_hit.values()) / max(1, sum(ev_total.values()))),
        false_alarm_episodes=false_alarms, false_alarms_per_person_hour=float(false_alarms / (person_s / 3600)),
        confusion=confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
        thr_on=cfg["app"]["thr_on"], thr_off=cfg["app"]["thr_off"], min_windows=cfg["app"]["min_windows"])
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=1))
    print("confusion (rows = true):")
    for c, r in zip(classes, res["confusion"]):
        print(f"  {c:17s} {r}")


if __name__ == "__main__":
    main()
