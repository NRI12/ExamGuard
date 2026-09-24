"""Read the DOAN1 CVAT 1.1 video exports: person tracks and behaviour event tracks."""
import glob
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


def session_names(raw_dir):
    zips = glob.glob(os.path.join(raw_dir, "cvat_zip", "*_cvat_v1.zip"))
    return sorted(os.path.basename(z).split("_cvat_v1")[0] for z in zips)


def load_tracks(raw_dir, session):
    """Returns (n_frames, tracks); a track is dict(id, label, boxes{frame: xyxy}, attrs, start, end)."""
    with zipfile.ZipFile(os.path.join(raw_dir, "cvat_zip", f"{session}_cvat_v1.zip")) as zf:
        root = ET.fromstring(zf.read("annotations.xml"))
    n_frames = int(root.findtext("meta/job/size") or root.findtext("meta/task/size"))
    tracks = []
    for t in root.iter("track"):
        boxes, attrs = {}, {}
        for b in t.iter("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = np.array([float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
            if not attrs:
                attrs = {a.get("name"): a.text for a in b.iter("attribute")}
        if boxes:
            tracks.append(dict(id=int(t.get("id")), label=t.get("label"), boxes=boxes, attrs=attrs,
                               start=min(boxes), end=max(boxes)))
    return n_frames, tracks


def box_at(track, frame):
    if frame in track["boxes"]:
        return track["boxes"][frame]
    return track["boxes"][min(track["boxes"], key=lambda k: abs(k - frame))]


def related_actor(attrs):
    r = attrs.get("related_actor")
    return None if r in (None, "__undefined__", attrs.get("participant_id")) else r
