#!/bin/bash
# Experiment 3: more pixels / longer temporal context for small objects (phone, cheat sheet).
set -e
source /venv/main/bin/activate
cd /workspace
HR=/workspace/data/clips_hr
[ -f $HR/clips_info.json ] && [ -f $HR/val.csv ] || python build_clips.py --out $HR --size 356 --clip-len 100
python train_x3d.py --data $HR --model x3d_m --stride 4                    # baseline on the same val set
python train_x3d.py --data $HR --model x3d_m --stride 6                    # 16x6 = 3.6 s context
python train_x3d.py --data $HR --model x3d_l --stride 5 --bs 12 --workers 11  # 312 px
