#!/bin/bash
# Run on the GPU machine, inside the project folder: install the package, download DOAN1, cut the clips.
# Tested on the Vast.ai "PyTorch (Vast)" template (CUDA 12.8, Python 3.12, torch preinstalled in /venv/main).
#
#   bash scripts/setup_instance.sh            # install + download (~3.7 GB) + prepare (~6 min)
#   bash scripts/setup_instance.sh --no-data  # install only
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available: use a GPU machine / install a CUDA build of torch first"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "| %.0f GB VRAM" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
EOF
pip install -q -e .
python -m cheating_v2 --help > /dev/null
if [ "${1:-}" != "--no-data" ]; then
    python -m cheating_v2 download
    python -m cheating_v2 prepare
fi
echo "ready: python -m cheating_v2 <command>   (outputs under $(pwd)/outputs)"
