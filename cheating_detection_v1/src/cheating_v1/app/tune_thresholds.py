"""Replay the alert logic on saved eval-videos logs to choose smoothing / hysteresis thresholds.

Prints cheating events caught vs false alarms per person-hour for each setting (Pareto front first).
Thresholds picked on the val sessions are optimistic for those sessions; confirm on new recordings.

  python -m cheating_v1 tune [--eval-dir outputs/app_eval]
"""
import collections
import itertools
import json
import os

import numpy as np

from ..config import config_parser, load_config
from ..models.registry import model_info, window_frames
from .eval_videos import gt_for_session, match_tracks, session_file


def episodes(seq, smooth, thr_on, thr_off, min_windows):
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


def score(data, fps, win, smooth, thr_on, thr_off, min_windows):
    hit = total = fa = 0
    person_h = 0.0
    for d in data:
        person_h += len(d["persons"]) * d["n"] / fps / 3600
        per_pid = collections.defaultdict(list)
        for tid, seq in d["by_track"].items():
            for s, e in episodes(seq, smooth, thr_on, thr_off, min_windows):
                per_pid[d["track_pid"].get(tid)].append((int(s * fps) - win, int(e * fps)))
        for pid, evs in d["events"].items():
            for s, e, lab in evs:
                if lab:
                    total += 1
                    hit += any(a <= e + win and b >= s for a, b in per_pid.get(pid, []))
        for pid, es in per_pid.items():
            fa += sum(pid is None or not any(lab and s <= b and e >= a for s, e, lab in d["events"][pid]) for a, b in es)
    return hit, total, fa, fa / person_h


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--eval-dir", help="default: outputs/app_eval")
    p.add_argument("--sessions", nargs="+", help="default: data.val_sessions")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    eval_dir = args.eval_dir or os.path.join(cfg["paths"]["outputs"], "app_eval")
    fps = cfg["data"]["fps"]
    win = window_frames(model_info(cfg["model"], cfg["data"]), fps)
    data = []
    for short in args.sessions or cfg["data"]["val_sessions"]:
        n, persons, events = gt_for_session(cfg, session_file(cfg, short))
        log = [json.loads(line) for line in open(os.path.join(eval_dir, f"{short}_log.jsonl"))]
        by_track = collections.defaultdict(list)
        for r in log:
            by_track[r["track"]].append((r["t"], np.array(r["probs"])))
        data.append(dict(n=n, persons=persons, events=events, track_pid=match_tracks(log, persons, fps)[0],
                         by_track=by_track))
    rows = []
    for smooth, thr_on, min_w in itertools.product((3, 5, 8), (0.6, 0.7, 0.8, 0.85, 0.9, 0.95), (1, 2, 3, 5)):
        hit, total, fa, fa_h = score(data, fps, win, smooth, thr_on, round(thr_on - 0.15, 2), min_w)
        rows.append((smooth, thr_on, round(thr_on - 0.15, 2), min_w, hit, total, fa, fa_h))
    lines = [f"{'smooth':>6} {'thr_on':>6} {'thr_off':>7} {'min_win':>7} {'events':>9} {'recall':>6} "
             f"{'false_alarms':>12} {'FA/person-h':>11}"]
    best = -1
    for r in sorted(rows, key=lambda r: (r[7], -r[4])):
        if r[4] > best:  # Pareto front: more events caught for each extra false-alarm level
            best = r[4]
            lines.append(f"{r[0]:>6} {r[1]:>6} {r[2]:>7} {r[3]:>7} {r[4]:>4}/{r[5]:<4} {r[4] / r[5]:>6.2f} "
                         f"{r[6]:>12} {r[7]:>11.1f}")
    print("\n".join(lines))
    open(os.path.join(eval_dir, "threshold_sweep.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(eval_dir, "threshold_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
