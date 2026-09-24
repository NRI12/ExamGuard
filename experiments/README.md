# Exam cheating detection (per-person video classification)

Detects cheating behaviour of each examinee in a classroom camera feed:
YOLO person detection + ByteTrack, then an X3D video classifier on a 3.2 s crop of every person, once per second.

Classes: `0 normal` (incl. hard negatives), `1 looking`, `2 interaction` (communicating / exchanging objects),
`3 phone_cheatsheet`. `cheat_score = 1 - P(normal)` drives the alert; the argmax of classes 1-3 gives the type.

## Layout

| Path | Content |
|---|---|
| `scripts/` | all code (below) |
| `data/DOAN1/` | original dataset from Google Drive: CVAT task backups (videos inside), CVAT XML exports |
| `data/clips/` | 256 px, 80-frame per-person clips (4 classes) |
| `data/clips_hr/` | 356 px, 100-frame clips used by the final models; `train.csv`, `val.csv`, `meta.csv` |
| `data/inspect/` | dataset visual checks (stats, timelines, clip strips, sample sheets) |
| `runs/<run>/` | `best.pt`, `last.pt`, `history.json`, `final_val_multiview.json`, TensorBoard events |
| `app_eval/` | ver 1: end-to-end results of the app on the full val videos (S07, S08) |
| `ver2/` | **ver 2** (normal / abnormal): `x3d_l_ver2_best.pt`, app results, threshold sweep, S07/S08 demos |
| `ver1/` | ver 1 demo on S07 |
| `data/clips_hr_v2/` | ver 2 labels (0 normal, 1 abnormal); the clips themselves are `data/clips_hr/videos` |
| `demo/` | ver 1 demo on S08 (annotated, 720p) |
| `external/` | tests outside the training data: D01 10 min excerpt, 3 Pexels videos, ver 1 results |
| `pretrained/` | TSM-R50 Kinetics-400 checkpoint (OpenMMLab) |
| `logs/` | training / evaluation logs |

`data/videos/*.mp4` (the full session videos) are not in the archive because they are byte-identical to the mp4
inside `data/DOAN1/annotations/backup/*_task_backup.zip`. Recreate them with:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); import inspect_dataset as d; d.extract_videos()"
```
(edit `ROOT` / `VID_DIR` at the top of `scripts/inspect_dataset.py` if the data is not under `/workspace/data`).

## Scripts

| Script | Purpose |
|---|---|
| `inspect_dataset.py` | extract videos, dataset stats and visual checks |
| `build_clips.py` | CVAT tracks -> per-person clips + `train.csv` / `val.csv` (val = S07, S08) |
| `video_data.py` | shared dataset, balanced sampler, GPU augmentation |
| `train_x3d.py` | X3D-M / X3D-L fine-tuning (pytorchvideo, Kinetics-400 init) |
| `train_tsm.py` | TSM-R50 fine-tuning (OpenMMLab Kinetics-400 init) |
| `train_vjepa2.py` | V-JEPA 2 ViT-L teacher (frozen encoder + attentive probe) |
| `train_distill.py` | online distillation teacher -> X3D |
| `infer_app.py` | **real-time app**: video file, webcam or RTSP |
| `eval_app.py` | end-to-end evaluation of the app on untrimmed val videos |
| `tune_thresholds.py` | replay alert logic on `eval_app.py` logs, recall vs false-alarm sweep |
| `calibrate_boxes.py` | re-fit `BOX_MAP` (YOLO box -> annotated crop geometry) |
| `make_v2_binary.py` | build the ver 2 label set (normal / abnormal) from `clips_hr` |
| `make_viewer.py` | results web page served from the Vast instance (token-protected) |
| `gdrive_download.py` | download a public Google Drive folder with one overall progress bar and resume |

## Results (val = sessions S07 + S08, 450 clips, 3-view test)

| Model | macro-F1 | cheat AUROC | cheat recall | cheat precision | GFLOPs / clip |
|---|---|---|---|---|---|
| X3D-M 16x4 | 0.688 | 0.946 | 0.788 | 0.938 | ~6 |
| X3D-M 16x6 | 0.698 | 0.939 | 0.810 | 0.921 | ~6 |
| **X3D-L 16x5, 312 px (deployed)** | **0.748** | **0.946** | **0.823** | **0.936** | ~25 |
| TSM-R50, 8 segments | 0.695 | 0.902 | 0.758 | 0.897 | ~33 |
| V-JEPA 2 ViT-L teacher (not real-time) | 0.818 | 0.968 | 0.866 | 0.966 | ~1200 |

## Ver 2: normal / abnormal only

Same clips and X3D-L recipe, labels collapsed to 2 classes (`scripts/make_v2_binary.py`), 15 epochs.

| | clip AUROC (3-view) | clip recall | clip precision | app: events caught / false alarms per person-hour at thr 0.85/0.70 |
|---|---|---|---|---|
| Ver 1 (4 classes) | 0.946 | 0.823 | 0.936 | 43/52 (83 %) / 15.6 |
| Ver 2 (2 classes) | 0.949 | 0.848 | 0.912 | 42/52 (81 %) / 22.5 |

Equal on clips, worse on full videos (about 45 % more false alarms at the same recall). Ver 1 stays the
recommended model; to show only "normal / abnormal" use ver 1 and hide the type in the overlay.

## Unseen classroom (D01)

Neither version is reliable on the unseen D01 classroom (25+ students, wide lens):
more labelled data from the target cameras is required.

## App on full untrimmed val videos (S07 + S08, ~24 min, 8766 windows)

`eval_app.py` runs the real pipeline (YOLO + ByteTrack boxes, 1 s windows over the whole video) against the CVAT
ground truth; `tune_thresholds.py` replays the alert logic from the saved logs. Mapped YOLO boxes match the
annotated boxes with median IoU 0.73. Window-level cheat AUROC is 0.921.

| smooth | thr_on / thr_off | min windows | cheating events caught | false alarms / person-hour |
|---|---|---|---|---|
| 3 | 0.60 / 0.45 | 1 | 50/52 (96 %) | 43.6 |
| 3 | 0.70 / 0.55 | 3 | 47/52 (90 %) | 22.9 |
| **3** | **0.85 / 0.70** | **2** | **43/52 (83 %)** | **15.6** (app default) |
| 3 | 0.90 / 0.75 | 1 | 38/52 (73 %) | 9.5 |
| 3 | 0.95 / 0.80 | 3 | 27/52 (52 %) | 3.5 |

These thresholds were picked on the same val sessions, so they are optimistic. Most false alarms are normal
movement (writing, turning, arranging papers) read as `interaction`. The app suits flagging clips for human review,
not fully automatic decisions. Throughput on an RTX 3090: ~40 fps for 1080p with 6 people, including drawing and
writing the output video. `demo/` contains an annotated 150 s excerpt of S08.

## Run the app

```bash
pip install torch torchvision "git+https://github.com/facebookresearch/pytorchvideo.git" ultralytics lap opencv-python tqdm
cd scripts
python infer_app.py --source exam.mp4 --ckpt ../runs/x3d_l_s5_clips_hr_20260924_105705/best.pt \
       --out annotated.mp4 --events events.csv --log windows.jsonl
python infer_app.py --source 0 --ckpt ../runs/x3d_l_s5_clips_hr_20260924_105705/best.pt --show          # webcam
python infer_app.py --source rtsp://cam/stream --ckpt ../runs/x3d_l_s5_clips_hr_20260924_105705/best.pt --show
```

Tuning: `--thr-on` / `--thr-off` (alert hysteresis on `cheat_score`), `--smooth` (windows averaged),
`--infer-every` (seconds between classifier runs). The YOLO box is remapped to the annotated "upper body + desk"
geometry with `BOX_MAP` in `infer_app.py` (fitted on 288 matched boxes); re-fit it for a very different camera angle.
