#!/bin/bash
# One-time system setup of a Vast.ai GPU instance for ExamGuard without Docker:
# PostgreSQL 16, nginx, Node (frontend build), Python 3.11 venv (uv, torch cu126 from uv.lock) + X3D deps.
# Idempotent. Run from the repository root on the instance:
#
#   bash scripts/vast/setup_instance.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
NODE_VERSION=$(tr -dc '0-9.' < .nvmrc)

echo "== system packages"
export DEBIAN_FRONTEND=noninteractive
if ! command -v psql > /dev/null || ! command -v nginx > /dev/null || ! command -v ffprobe > /dev/null; then
    apt-get update -qq
    apt-get install -y -qq postgresql nginx ffmpeg libgl1 libglib2.0-0 curl > /dev/null
fi

echo "== node $NODE_VERSION"
if ! command -v node > /dev/null || [ "$(node --version)" != "v$NODE_VERSION" ]; then
    curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" | tar -xJ -C /opt
    ln -sf "/opt/node-v$NODE_VERSION-linux-x64/bin/"{node,npm,npx} /usr/local/bin/
fi

echo "== python venv (backend/.venv)"
cd "$ROOT/backend"
export UV_PROJECT_ENVIRONMENT="$ROOT/backend/.venv" UV_LINK_MODE=copy
uv sync --frozen --no-dev --extra cu126 --python 3.11
# the X3D-L classifier (pytorchvideo) is not in uv.lock: installed next to the locked environment
uv pip install --python .venv/bin/python -q \
    "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git" fvcore iopath av
.venv/bin/python - <<'PY'
import torch, pytorchvideo, ultralytics
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, torch.cuda.get_device_name(0), "| ultralytics", ultralytics.__version__)
PY
echo "setup done"
