from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import load_events, load_seats, read_video_meta, resolve_video_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--seats", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    events = load_events(args.events)
    seats = load_seats(args.seats)
    project_root = Path(args.project_root) if args.project_root else None

    errors: list[str] = []
    warnings: list[str] = []

    event_ids = Counter(e.source_event_id for e in events)
    dup_ids = [k for k, v in event_ids.items() if v > 1]
    if dup_ids:
        errors.append(f"Duplicate source_event_id: {dup_ids[:10]}")

    seat_keys = Counter((s.session_id, s.participant_id) for s in seats)
    dup_seat_people = [k for k, v in seat_keys.items() if v > 1]
    if dup_seat_people:
        errors.append(f"Participant mapped to multiple seats: {dup_seat_people[:10]}")

    seat_id_keys = Counter((s.session_id, s.seat_id) for s in seats)
    dup_seat_ids = [k for k, v in seat_id_keys.items() if v > 1]
    if dup_seat_ids:
        errors.append(f"Duplicate seat_id within session: {dup_seat_ids[:10]}")

    seat_lookup = {(s.session_id, s.participant_id): s for s in seats}
    seat_by_id = {(s.session_id, s.seat_id): s for s in seats}

    video_meta_cache = {}
    for e in events:
        key = (e.session_id, e.participant_id)
        if key not in seat_lookup:
            errors.append(
                f"Missing seat mapping: event={e.source_event_id}, "
                f"session={e.session_id}, participant={e.participant_id}"
            )
            continue
        s = seat_lookup[key]
        if s.global_subject_id != e.global_subject_id:
            errors.append(
                f"Global subject mismatch: event={e.source_event_id}, "
                f"event={e.global_subject_id}, seat_map={s.global_subject_id}"
            )

        p = resolve_video_path(e.video_path, project_root)
        if not p.exists():
            errors.append(f"Missing video: {p}")
            continue
        if p not in video_meta_cache:
            try:
                video_meta_cache[p] = read_video_meta(p)
            except Exception as ex:
                errors.append(f"Cannot read video {p}: {ex}")
                continue
        meta = video_meta_cache[p]
        if e.end_sec > meta.duration_sec + 0.25:
            errors.append(
                f"Event {e.source_event_id} ends at {e.end_sec:.3f}s "
                f"> video {meta.duration_sec:.3f}s"
            )

    for s in seats:
        for n in s.neighbors:
            if (s.session_id, n) not in seat_by_id:
                errors.append(
                    f"{s.session_id}/{s.seat_id} references unknown neighbor {n}"
                )
            elif n == s.seat_id:
                errors.append(f"{s.session_id}/{s.seat_id} references itself as neighbor")

    classes = Counter(e.behavior for e in events)
    splits = Counter(e.split for e in events)
    folds = Counter(e.fold for e in events if e.fold)

    print("=== VALIDATION SUMMARY ===")
    print(f"events: {len(events)}")
    print(f"seats: {len(seats)}")
    print(f"videos: {len(video_meta_cache)}")
    print(f"classes: {dict(classes)}")
    print(f"splits: {dict(splits)}")
    print(f"folds: {dict(folds)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    for x in warnings:
        print(f"[WARN] {x}")
    for x in errors:
        print(f"[ERROR] {x}")

    if errors:
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
