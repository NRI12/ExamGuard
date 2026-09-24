"""Replay the app's alert logic on saved eval_app.py logs to pick smoothing / hysteresis thresholds.

Uses the per-window probabilities in <eval dir>/<session>_log.jsonl, so no video is decoded again.
Prints event recall vs false alarms per person-hour for every setting. Thresholds picked here on the
val sessions are optimistic for those same sessions; confirm on new recordings.

  python tune_thresholds.py --eval-dir /workspace/app_eval
"""
import argparse
import collections
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_app import VID_DIR, gt_for_session, iou  # noqa: E402


def load(eval_dir, short):
    sess = os.path.basename(glob.glob(f"{VID_DIR}/{short}_*.mp4")[0])[:-4]
    n, persons, events = gt_for_session(sess)
    log = [json.loads(line) for line in open(f"{eval_dir}/{short}_log.jsonl")]
    votes = collections.defaultdict(collections.Counter)
    for r in log:
        f = int(round(r["t"] * 25))
        cand = [(iou(r["box"], p["boxes"][f]), pid) for pid, p in persons.items() if f in p["boxes"]]
        if cand and max(cand)[0] >= 0.3:
            votes[r["track"]][max(cand)[1]] += 1
    track_pid = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    by_track = collections.defaultdict(list)
    for r in log:
        by_track[r["track"]].append((r["t"], np.array(r["probs"])))
    return dict(n=n, persons=persons, events=events, track_pid=track_pid, by_track=by_track)


def episodes(seq, smooth, thr_on, thr_off, min_windows):
    """Replay smoothing + hysteresis; returns [(start_t, end_t)] alert episodes of >= min_windows windows."""
    out, buf, cur = [], collections.deque(maxlen=smooth), None
    for t, p in seq:
        buf.append(p)
        score = 1 - np.mean(buf, 0)[0]
        if cur is None and score >= thr_on:
            cur = [t, t, 1]
        elif cur is not None and score < thr_off:
            if cur[2] >= min_windows:
                out.append((cur[0], cur[1]))
            cur = None
        elif cur is not None:
            cur[1], cur[2] = t, cur[2] + 1
    if cur is not None and cur[2] >= min_windows:
        out.append((cur[0], cur[1]))
    return out


def score(data, smooth, thr_on, thr_off, min_windows):
    hit = total = fa = 0
    person_hours = 0.0
    for d in data:
        person_hours += len(d["persons"]) * d["n"] / 25 / 3600
        eps = {tid: episodes(seq, smooth, thr_on, thr_off, min_windows) for tid, seq in d["by_track"].items()}
        per_pid = collections.defaultdict(list)
        for tid, es in eps.items():
            pid = d["track_pid"].get(tid)
            for s, e in es:
                per_pid[pid].append((int(s * 25) - 75, int(e * 25)))
        for pid, evs in d["events"].items():
            for s, e, lab in evs:
                if lab:
                    total += 1
                    hit += any(a <= e + 75 and b >= s for a, b in per_pid.get(pid, []))
        for pid, es in per_pid.items():
            for a, b in es:
                if pid is None or not any(lab and s <= b and e >= a for s, e, lab in d["events"][pid]):
                    fa += 1
    return hit, total, fa, fa / person_hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/workspace/app_eval")
    ap.add_argument("--sessions", nargs="+", default=["S07", "S08"])
    args = ap.parse_args()
    data = [load(args.eval_dir, s) for s in args.sessions]
    rows = []
    for smooth, thr_on, gap, min_w in itertools.product((3, 5, 8), (0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
                                                         (0.15,), (1, 2, 3, 5)):
        hit, total, fa, fa_h = score(data, smooth, thr_on, thr_on - gap, min_w)
        rows.append((smooth, thr_on, round(thr_on - gap, 2), min_w, hit, total, fa, fa_h))
    print(f"{'smooth':>6} {'thr_on':>6} {'thr_off':>7} {'min_win':>7} {'events':>9} {'recall':>6} "
          f"{'false_alarms':>12} {'FA/person-h':>11}")
    for r in sorted(rows, key=lambda r: (r[7], -r[4])):
        print(f"{r[0]:>6} {r[1]:>6} {r[2]:>7} {r[3]:>7} {r[4]:>4}/{r[5]:<4} {r[4] / r[5]:>6.2f} {r[6]:>12} {r[7]:>11.1f}")
    json.dump(rows, open(f"{args.eval_dir}/threshold_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
