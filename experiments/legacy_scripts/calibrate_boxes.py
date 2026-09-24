"""Fit BOX_MAP in infer_app.py: match YOLO person boxes to CVAT person boxes on train sessions."""
import numpy as np, cv2, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from build_clips import load_tracks
from ultralytics import YOLO
det = YOLO("yolo11m.pt")
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)
rows = []
for sess in ["S00_6P_1080p_25fps", "S01_6P_1080p_25fps", "S03_6P_1080p_25fps", "S05_6P_1080p_25fps"]:
    n, tracks = load_tracks(sess)
    persons = [t for t in tracks if t["label"] == "person"]
    cap = cv2.VideoCapture(f"/workspace/data/videos/{sess}.mp4")
    for f in np.linspace(100, n - 100, 12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        r = det(img, classes=[0], conf=0.25, verbose=False)[0]
        yb = r.boxes.xyxy.cpu().numpy()
        for p in persons:
            g = p["boxes"][f]
            # best YOLO box by horizontal overlap + top alignment (full-body boxes differ in height)
            cand = [(iou([y[0], g[1], y[2], g[3]], g), y) for y in yb]
            if not cand: continue
            s, y = max(cand, key=lambda c: c[0])
            if s < 0.5: continue
            rows.append(dict(sess=sess, gw=g[2]-g[0], gh=g[3]-g[1], yw=y[2]-y[0], yh=y[3]-y[1],
                             dx1=(g[0]-y[0])/(y[2]-y[0]), dx2=(g[2]-y[2])/(y[2]-y[0]),
                             dy1=(g[1]-y[1])/(y[2]-y[0]), gh_over_yw=(g[3]-g[1])/(y[2]-y[0]),
                             full_iou=iou(y, g)))
    cap.release()
a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "sess"}
print("matched", len(rows))
for k in ("dx1", "dx2", "dy1", "gh_over_yw", "full_iou"):
    print(f"{k:11s} median {np.median(a[k]):+.3f}  IQR [{np.percentile(a[k],25):+.3f}, {np.percentile(a[k],75):+.3f}]")
print("GT h/w median", np.median(a["gh"]/a["gw"]).round(3), " YOLO h/w median", np.median(a["yh"]/a["yw"]).round(3))
m = dict(dx1=np.median(a["dx1"]), dx2=np.median(a["dx2"]), dy1=np.median(a["dy1"]), h=np.median(a["gh_over_yw"]))
def mapped(y):
    w = y[2]-y[0]
    return [y[0]+m["dx1"]*w, y[1]+m["dy1"]*w, y[2]+m["dx2"]*w, y[1]+m["dy1"]*w+m["h"]*w]
print("IoU raw YOLO vs GT      :", np.median(a["full_iou"]).round(3))
print("mapping (fractions of YOLO width):", {k: round(float(v), 3) for k, v in m.items()})
