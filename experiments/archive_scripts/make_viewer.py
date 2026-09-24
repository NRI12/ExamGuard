"""Build /workspace/viewer: an index.html with every result video + a metrics table, served by the
'Result Viewer' supervisor service behind the Vast Caddy token auth (see /opt/supervisor-scripts/viewer.sh).

  python make_viewer.py            # re-run whenever new results land; the server picks files up live
"""
import html
import json
import os

ROOT = "/workspace/viewer"
VIDEOS = [  # (title, file on disk, session, seconds) - one video, original val session
    ("Ver 2 (normal / abnormal) - S08, first 150 s", "/workspace/ver2/S08_ver2_150s_720p.mp4", "S08", 150),
]


def gt_note(short, seconds):
    """Ground-truth events inside the excerpt, from the CVAT export."""
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_clips import load_tracks
    sess = os.path.basename(glob.glob(f"/workspace/data/videos/{short}_*.mp4")[0])[:-4]
    _, tracks = load_tracks(sess)
    evs = sorted((t for t in tracks if t["label"] != "person" and t["start"] < seconds * 25), key=lambda t: t["start"])
    parts = []
    for t in evs:
        a = t["attrs"]
        who = a["participant_id"] + (f"->{a['related_actor']}" if a.get("related_actor") not in (None, "__undefined__") else "")
        parts.append(f"{t['start'] / 25:.0f}-{t['end'] / 25:.0f} s {who} {t['label']} ({a.get('subtype')})")
    return "Ground truth: " + ("; ".join(parts) if parts else "no events")


METRICS = [  # (name, clip-level multiview json, app eval summary json)
    ("Ver 1: X3D-L, 4 classes", "/workspace/runs/x3d_l_s5_clips_hr_20260924_105705/final_val_multiview.json",
     "/workspace/app_eval/summary.json"),
    ("Ver 2: X3D-L, normal / abnormal", "/workspace/ver2/final_val_multiview.json", "/workspace/ver2/app_eval/summary.json"),
]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    os.makedirs(ROOT, exist_ok=True)
    rows = []
    for name, clip_p, app_p in METRICS:
        c, a = load(clip_p), load(app_p)
        cell = lambda v, f="{:.3f}": f.format(v) if v is not None else "-"
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{cell(c and c['cheat_auroc'])}</td><td>{cell(c and c['cheat_recall'])}</td>"
            f"<td>{cell(c and c['cheat_precision'])}</td>"
            f"<td>{cell(a and a['cheat_auroc'])}</td>"
            f"<td>{html.escape(str(a['event_recall_total'] if a else '-'))[:5]}</td>"
            f"<td>{cell(a and a['false_alarms_per_person_hour'], '{:.1f}')}</td></tr>")
    for old in os.listdir(ROOT):  # drop links to videos no longer listed
        if old.endswith(".mp4") and os.path.islink(os.path.join(ROOT, old)):
            os.remove(os.path.join(ROOT, old))
    cards = []
    for i, (title, path, short, seconds) in enumerate(VIDEOS):
        if not os.path.exists(path):
            continue
        note = gt_note(short, seconds)
        link = f"v{i}.mp4"
        dst = os.path.join(ROOT, link)
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(path, dst)
        mb = os.path.getsize(path) / 1e6
        cards.append(f"<section><h2>{html.escape(title)}</h2><p class=note>{html.escape(note)} ({mb:.0f} MB)</p>"
                     f"<video controls preload=metadata src='{link}'></video></section>")
    sweep = "/workspace/ver2/app_eval/threshold_sweep.txt"
    sweep_html = f"<h2>Ver 2 threshold sweep (S07 + S08)</h2><pre>{html.escape(open(sweep).read())}</pre>" \
        if os.path.exists(sweep) else ""
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Cheating Detection Results</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--muted:#5d6675;--line:#dde1e7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--card:#1b2029;--muted:#9aa3b2;--line:#2c3340}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
h1{{font-size:22px}} h2{{font-size:17px;margin:0 0 4px}} .note{{color:var(--muted);margin:0 0 8px}}
video{{width:100%;border-radius:6px;background:#000}}
table{{border-collapse:collapse;width:100%;background:var(--card);font-variant-numeric:tabular-nums}}
td,th{{border:1px solid var(--line);padding:6px 8px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.wrap{{overflow-x:auto}} pre{{overflow-x:auto;background:var(--card);padding:10px;border:1px solid var(--line)}}
</style></head><body><main>
<h1>Exam cheating detection: results</h1>
<p class=note>Clip level = 450 held-out clips (S07, S08). App level = full untrimmed S07 + S08 videos
(YOLO + ByteTrack + 1 s windows). Original training dataset only; S07 and S08 were never trained on.</p>
<div class=wrap><table><tr><th>Model</th><th>Clip AUROC</th><th>Clip recall</th><th>Clip precision</th>
<th>App AUROC</th><th>Events caught</th><th>False alarms / person-hour</th></tr>{''.join(rows)}</table></div>
{''.join(cards)}{sweep_html}</main></body></html>"""
    open(os.path.join(ROOT, "index.html"), "w").write(page)
    print(f"viewer: {len(cards)} videos, {len(rows)} metric rows")


if __name__ == "__main__":
    main()
