#!/bin/bash
# Wait for the V-JEPA 2 teacher, then distil it into X3D-L and X3D-M students.
set -e
source /venv/main/bin/activate
cd /workspace
while pgrep -f "python train_vjepa2.py" >/dev/null; do sleep 20; done
TEACHER=$(ls -d /workspace/runs/vjepa2_vitl-fpc16-256-ssv2_u0_s5_clips_hr_* | tail -1)/best.pt
echo "teacher: $TEACHER"
python train_distill.py --teacher $TEACHER --student x3d_l --bs 12 --epochs 1 --max-steps 3 --test-views 1 \
    --out /workspace/runs_debug && rm -rf /workspace/runs_debug
python train_distill.py --teacher $TEACHER --student x3d_l --bs 12
python train_distill.py --teacher $TEACHER --student x3d_m --bs 24
