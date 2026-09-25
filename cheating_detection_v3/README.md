# cheating_detection_v3

Two-stream exam cheating detection: the v1 RGB model (X3D-L) plus an **optical-flow stream**, fused.
Same 4 classes as v1: `0 normal`, `1 looking`, `2 interaction`, `3 phone_cheatsheet`; alert on `1 - P(normal)`.

- **Optical flow**: RAFT-small (torchvision), flow between frames 0.2 s apart (the sampling step of the models),
  pre-computed for every clip (`flow` command, uint8 at 160 px) and computed live by the app on the half-resolution
  frame, then cropped per person.
- **Flow stream**: X3D-S with a 2-channel stem (Kinetics-400 RGB filters averaged) and an adaptive head pool,
  16 flow frames at 160 px.
- **Fusion**: pooled RGB (2048) + flow (2048) features -> dropout -> linear. The fusion layer starts as the average
  of the two single-stream heads (**late fusion**, `weights/late_fusion.pt`) and can be fine-tuned
  (`weights/fusion_best.pt`).

Self-contained: its own code (`cheating_v3`), configs, data, weights and results.

## Layout

```
cheating_detection_v3/
├── configs/default.yaml       fusion (default model: weights/late_fusion.pt)
├── configs/flow.yaml          flow stream alone
├── src/cheating_v3/
│   ├── data/flow.py           RAFT pre-computation of the clips' optical flow
│   ├── data/dataset.py        RGB / flow / both, GPU augmentation (flow flips negate x, resizes rescale vectors)
│   ├── models/raft_flow.py    RAFT-small wrapper, flow quantisation
│   ├── models/two_stream.py   flow X3D-S, FusionNet (late-fusion initialisation)
│   ├── models/registry.py     rgb / flow / fusion: build, forward, save / load
│   ├── train.py, evaluate.py  training (epoch 0 of the fusion = late fusion, evaluated first) and evaluation
│   └── app/                   real-time detector with live RAFT flow, eval-videos, tune, calibrate
├── scripts/                   upload_to_instance.sh, setup_instance.sh (install, download, prepare, flow)
├── data/raw/DOAN1/            original dataset
├── weights/                   rgb_x3d_l_best.pt (v1), flow_x3d_s_best.pt, late_fusion.pt, fusion_best.pt
└── outputs/
    ├── runs/                  flow and fusion training runs (best.pt, last.pt, history, TensorBoard)
    └── reference/             cleaned logs and results of every model below
```

## Run

```bash
bash scripts/upload_to_instance.sh <ssh_port> root@<ip> /workspace        # on your machine
cd /workspace/cheating_detection_v3 && bash scripts/setup_instance.sh      # install, data, clips, flow (~20 min)
python -m cheating_v3 train --config configs/flow.yaml                      # flow stream (~11 min)
cp outputs/runs/<flow run>/best.pt weights/flow_x3d_s_best.pt
python -m cheating_v3 train                                                 # fusion fine-tuning (~17 min, optional)
python -m cheating_v3 evaluate                                              # default: late fusion
python -m cheating_v3 evaluate --checkpoint weights/rgb_x3d_l_best.pt       # RGB only (= v1)
python -m cheating_v3 infer --source exam.mp4 --out annotated.mp4          # ~35 fps on 1080p, 6 people, RTX 3090
python -m cheating_v3 eval-videos && python -m cheating_v3 tune
```

## Results (val = S07 + S08, never trained on)

Clip level, 450 clips, 3-view test:

| model | acc | macro-F1 | cheat AUROC | cheat recall | cheat precision | F1 normal / looking / interaction / phone |
|---|---|---|---|---|---|---|
| RGB X3D-L (= v1) | 0.829 | 0.748 | 0.946 | **0.823** | 0.936 | 0.884 / 0.512 / 0.763 / **0.832** |
| flow X3D-S alone | 0.751 | 0.653 | 0.908 | 0.753 | 0.902 | 0.840 / 0.419 / 0.660 / 0.694 |
| **late fusion** | **0.831** | **0.765** | **0.948** | 0.797 | **0.953** | 0.882 / **0.595** / **0.784** / 0.798 |
| fusion fine-tuned (best epoch 4 / 10) | 0.822 | 0.733 | 0.945 | 0.797 | 0.948 | 0.880 / 0.476 / 0.747 / 0.829 |

Full S07 + S08 videos (24 min, 52 cheating events), window AUROC: v1 0.921, late fusion **0.929**,
fine-tuned fusion 0.920. At a comparable false-alarm rate the event recall is about the same:

| false alarms / person-hour | v1 RGB | late fusion (v3) |
|---|---|---|
| ~3.5 | 27 / 52 | 28 / 52 |
| ~13-15 | 41 / 52 (14.7) | 41 / 52 (13.4, app default: thr 0.70 / 0.55, min 3) |
| ~23-24 | 47 / 52 (22.9) | 47 / 52 (24.2) |

**Conclusion.** Optical flow alone is informative (AUROC 0.908) but weaker than RGB. Late fusion gives a small
gain on clips (macro-F1 +0.017, precision +0.017, better looking / interaction, lower recall and phone F1) and a
slightly higher window AUROC, but the same events-vs-false-alarms trade-off on full videos. Fine-tuning the fusion
over-fits the 1285 training clips. The differences are within the noise of a 2-session val set (14 looking clips),
while v3 needs a second network and live RAFT (~35 fps instead of ~40 fps). v1 remains the simpler choice. v3 is worth
re-testing once data from other rooms / cameras exists: motion cues may depend less on the background than
appearance does (not verified here).

The epoch and thresholds were chosen on S07 + S08, so all numbers are optimistic.
