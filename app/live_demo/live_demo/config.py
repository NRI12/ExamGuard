"""Settings of the live service. Everything can be overridden with environment variables (LIVE_*)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("LIVE_WEIGHTS", ROOT / "weights"))
MEDIA = Path(os.environ.get("LIVE_MEDIA", ROOT / "media"))
DATA = ROOT / "data"
STATIC = ROOT / "static"

# the detection / crop settings shared by every project's app section (configs/default.yaml)
APP = dict(
    detector=str(WEIGHTS / "yolo11m.pt"), det_conf=0.3, det_imgsz=640, infer_every=1.0, min_coverage=0.75,
    clip_size=356, crop_pad=0.15, box_map=dict(dx1=-0.101, dx2=0.094, dy1=-0.131, h=1.021),
)

CLASSES4 = ["normal", "looking", "interaction", "phone_cheatsheet"]
# same ids, names, default decision rules and order as demo_web/data/manifest.json
MODELS = [
    dict(id="v1", name="v1  X3D-L, 4 classes", weights="x3d_l_v1_best.pt", recommended=True,
         thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Recommended model. RGB X3D-L on a 3.2 s per-person window, 4 classes."),
    dict(id="v2", name="v2  X3D-L, normal / abnormal", weights="x3d_l_v2_best.pt",
         thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Same network trained on 2 classes: flags behaviour without naming its type."),
    dict(id="v3", name="v3  RGB + optical flow, fine-tuned fusion", weights="fusion_best.pt",
         thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Two-stream model: X3D-L on RGB plus X3D-S on RAFT optical flow (flow computed live on the GPU)."),
    dict(id="v3late", name="v3  RGB + optical flow, late fusion", weights="late_fusion.pt",
         thr_on=0.70, thr_off=0.55, smooth=3, min_windows=3,
         note="Average of the RGB and flow streams (shares the live RAFT flow with v3)."),
    dict(id="v1a", name="v1 variant A  (interaction labels)", weights="x3d_l_v1a_best.pt",
         thr_on=0.85, thr_off=0.70, smooth=3, min_windows=2,
         note="Experiment: passive partner of whisper / tap events not labelled. Worse than v1 on interaction."),
]
ENABLED = [m.strip() for m in os.environ.get("LIVE_MODELS", "v1,v2,v3,v3late,v1a").split(",") if m.strip()]

# built-in sessions with ground truth (validation sessions, never trained on)
SESSIONS = {"S07": "S07_6P_1080p_25fps", "S08": "S08_6P_1080p_25fps"}

DEVICE = os.environ.get("LIVE_DEVICE", "cuda")
COMPILE = os.environ.get("LIVE_COMPILE", "1") == "1"   # torch.compile the classifiers (GPU only)
BATCH = int(os.environ.get("LIVE_BATCH", "8"))          # clips per classifier call (padded): one compiled shape
# models run for a viewer: "family" (the model on screen + the models of the same kind: RGB v1 / v2 / v1a together,
# real time on an RTX 3060), "selected" (only the model on screen) or "all" (all five: needs more than a 3060)
DEFAULT_RUN = os.environ.get("LIVE_DEFAULT_RUN", "family")
MAX_SESSIONS = int(os.environ.get("LIVE_MAX_SESSIONS", "2"))
PREROLL = 3.2            # s decoded before a seek target so the first decision is available right away
MAX_LAG = float(os.environ.get("LIVE_MAX_LAG", "3.0"))  # s behind the viewer before jumping to the present
MAX_UPLOAD_BYTES = int(os.environ.get("LIVE_MAX_UPLOAD_BYTES", str(4 << 30)))
TOKEN = os.environ.get("LIVE_TOKEN", "")  # optional shared secret (?token=...) for a public instance
