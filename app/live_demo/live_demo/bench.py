"""Throughput and correctness check of the live pipeline, without a browser.

Runs the same Analyzer the page uses on the first N seconds of a video as fast as possible (or paced at --rate x
real time), reports samples/s and GPU time per sample, and compares the probabilities with a reference eval-videos
log (same tracks, same windows) when given.

  python -m live_demo.bench --video S07 --seconds 60
  python -m live_demo.bench --video S07 --seconds 20 --rate 1          # paced like a viewer at 1x
  python -m live_demo.bench --video S07 --seconds 10 --model v1 --ref ../cheating_detection_v1/outputs/reference/x3d_l/app_eval/S07_log.jsonl
"""
import argparse
import json
import logging
import threading
import time

import numpy as np

from .pipeline import Analyzer, ModelBank
from .videos import Library


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="S07")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--rate", type=float, default=0, help="0 = as fast as possible, else viewer playback rate")
    ap.add_argument("--ref", help="eval-videos per-window log (JSONL) to compare with")
    ap.add_argument("--model", default="v1", help="model id compared with --ref")
    ap.add_argument("--models", default="", help="models to run, comma separated (default: all loaded)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    bank = ModelBank()
    video = Library().get(args.video)
    rows, stats, done = [], [], threading.Event()

    def emit(msg):
        if msg["type"] == "win":
            rows.extend(msg["rows"])
        elif msg["type"] == "stat":
            stats.append(msg)
            if msg["proc"] >= args.seconds - 0.05:
                done.set()
        elif msg["type"] == "error":
            print(msg["message"])
            done.set()

    a = Analyzer(bank, video, emit, [m for m in args.models.split(",") if m] or None)
    if not args.rate:
        a.max_lag = float("inf")  # unpaced: never jump ahead
    t0 = time.time()
    if args.rate:
        a.clock(0, True, args.rate)
    else:
        a.clock(args.seconds, False, 1.0)
    done.wait(timeout=args.seconds * 60 + 600)
    wall = time.time() - t0
    a.close()
    lags = [s["lag"] for s in stats if s["proc"] > 4]
    print(f"\n{args.video}: {args.seconds:.0f} s of video in {wall:.1f} s wall "
          f"({args.seconds / wall:.2f}x real time), {len(rows)} windows, models {a.run_models} on {bank.gpu_name}")
    if stats:
        print(f"GPU time per 0.2 s sample (incl. classifiers): mean {np.mean([s['ms'] for s in stats if s['ms']]):.1f} ms")
    if args.rate and lags:
        print(f"lag behind the viewer: mean {np.mean(lags):.2f} s, max {max(lags):.2f} s")
    if args.ref:
        ref = {}
        for line in open(args.ref):
            r = json.loads(line)
            if r["t"] <= args.seconds:
                ref[(r["t"], r["track"])] = r
        diffs, top1, missing = [], 0, 0
        for r in rows:
            q = ref.get((r["t"], r["track"]))
            if q is None:
                missing += 1
                continue
            p = np.array(r["p"][args.model])
            diffs.append(np.abs(p - np.array(q["probs"])).max())
            top1 += int(np.argmax(p) == np.argmax(q["probs"]))
        n = len(diffs)
        print(f"vs reference: {n} windows matched by (t, track), {missing} live windows not in the reference, "
              f"{len(ref) - n} reference windows not produced live")
        if n:
            print(f"max |p_live - p_ref|: median {np.median(diffs):.4f}, max {np.max(diffs):.4f}; "
                  f"same top-1 class {top1}/{n}")


if __name__ == "__main__":
    main()
