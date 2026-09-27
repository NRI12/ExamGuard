from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DatasetBuilder
from .io_utils import load_events, load_seats, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--seats", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    events = load_events(args.events)
    seats = load_seats(args.seats)
    cfg = load_yaml(args.config)
    out = Path(args.out)
    project_root = Path(args.project_root) if args.project_root else None

    builder = DatasetBuilder(events, seats, cfg, out, project_root)
    try:
        builder.build()
    finally:
        builder.close()

    print(f"DONE: {out / 'deployment_roi_manifest.jsonl'}")
    print(f"SUMMARY: {out / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
