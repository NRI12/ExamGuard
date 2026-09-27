#!/bin/bash
# Run on the GPU instance, inside live_demo/: system packages, Python packages, videos. Idempotent.
# Tested on the Vast.ai "PyTorch (Vast)" image (vastai/pytorch:cuda-12.8.1-auto, torch in /venv/main).
#
#   bash scripts/setup_vast.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
if ! command -v ffmpeg > /dev/null || ! ldconfig -p | grep -q libGL.so.1; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 > /dev/null
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "| %.0f GB VRAM" % (torch.cuda.get_device_properties(0).total_memory / 2**30), "| bf16", torch.cuda.is_bf16_supported())
PY
pip install -q -r requirements.txt
python -m live_demo.fetch_media
# RAFT-small weights (v3 models) are downloaded once here instead of at the first start
python -c "from torchvision.models.optical_flow import Raft_Small_Weights, raft_small; raft_small(weights=Raft_Small_Weights.DEFAULT)"
ls -la weights media
echo "setup done. start: bash scripts/start.sh"
