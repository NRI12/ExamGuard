#!/bin/bash
# ExamGuard on a Linux server with an NVIDIA GPU (Ubuntu 22.04 / 24.04, Debian 12), no Docker.
#   - PostgreSQL + ffmpeg from apt, Python 3.11 venv built by uv from backend/uv.lock (torch 2.7.1 cu126)
#   - database created and, on first install, restored from data/examguard.dump (sessions, events, reviews)
#   - one process: uvicorn serves the API, the WebSocket and the built frontend on port 8000
#
#   sudo bash deploy/linux/install.sh              # install / update, then: bash deploy/linux/start.sh
#   sudo bash deploy/linux/install.sh --service    # also install and start the systemd service "examguard"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
SERVICE=0; [ "${1:-}" = "--service" ] && SERVICE=1
RUN_USER=${SUDO_USER:-$(id -un)}
[ "$(id -u)" = 0 ] || { echo "run with sudo (apt, postgres)"; exit 1; }

echo "== GPU"
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Install the NVIDIA driver (>= 560 for CUDA 12.6) first."
fi

echo "== system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq postgresql ffmpeg libgl1 libglib2.0-0 curl ca-certificates openssl > /dev/null

echo "== uv + Python 3.11 environment"
if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
cd "$ROOT/backend"
export UV_PROJECT_ENVIRONMENT="$ROOT/backend/.venv" UV_LINK_MODE=copy
uv sync --frozen --no-dev --extra cu126 --python 3.11
uv pip install --python .venv/bin/python -q \
    "pytorchvideo @ https://github.com/facebookresearch/pytorchvideo/archive/f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0.zip" \
    fvcore==0.1.5.post20221221 iopath==0.1.10 av==18.1.0
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
cd "$ROOT"

echo "== configuration (.env)"
if [ ! -f .env ]; then
    DB_PASSWORD=$(openssl rand -hex 16)
    cat > .env <<EOF
APP_ENV=production
APP_PROFILE=rtx3060
POSTGRES_DB=examguard
POSTGRES_USER=examguard
POSTGRES_PASSWORD=$DB_PASSWORD
DATABASE_URL=postgresql+psycopg://examguard@127.0.0.1:5432/examguard
DATABASE_PASSWORD=$DB_PASSWORD
JWT_SECRET=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
UPLOAD_ROOT=$ROOT/data/uploads
EVIDENCE_ROOT=$ROOT/data/evidence
MODEL_ROOT=$ROOT/model_artifacts
CONFIG_ROOT=$ROOT/backend/configs
FRONTEND_DIST=$ROOT/frontend/dist
MAX_UPLOAD_BYTES=4294967296
FFPROBE_TIMEOUT_SECONDS=30
# HTTP on a LAN needs COOKIE_SECURE=false; set true behind HTTPS (deploy/linux/tunnel.sh or a reverse proxy)
COOKIE_SECURE=false
PORT=8000
ADMIN_USERNAME=admin
ADMIN_PASSWORD=$(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | head -c 14)Aa1
ADMIN_FULL_NAME="Quản trị viên"
EOF
    chmod 600 .env
fi
set -a; source .env; set +a
mkdir -p "$UPLOAD_ROOT" "$EVIDENCE_ROOT" logs

echo "== PostgreSQL"
if command -v systemctl > /dev/null && systemctl list-unit-files postgresql.service > /dev/null 2>&1 \
    && [ -d /run/systemd/system ]; then
    systemctl enable --now postgresql > /dev/null
else
    pg_ctlcluster "$(pg_lsclusters -h | awk 'NR==1{print $1}')" main start 2> /dev/null || true
fi
for _ in $(seq 1 30); do pg_isready -q -h 127.0.0.1 && break; sleep 1; done
su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='$POSTGRES_USER'\"" | grep -q 1 \
    || su postgres -c "psql -qc \"CREATE ROLE $POSTGRES_USER LOGIN PASSWORD '$POSTGRES_PASSWORD'\""
su postgres -c "psql -qc \"ALTER ROLE $POSTGRES_USER PASSWORD '$POSTGRES_PASSWORD'\""
if ! su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'\"" | grep -q 1; then
    su postgres -c "createdb -O $POSTGRES_USER $POSTGRES_DB"
    if [ -f data/examguard.dump ]; then
        echo "restoring data/examguard.dump"
        PGPASSWORD=$POSTGRES_PASSWORD pg_restore -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
            --no-owner --no-privileges data/examguard.dump
    fi
fi

echo "== migrations + admin"
cd backend
.venv/bin/python -m alembic upgrade head
.venv/bin/python -m app.cli.bootstrap_admin
cd "$ROOT"

echo "== models / frontend"
for f in model_artifacts/detection/yolo11n.pt model_artifacts/cheating/x3d_l_v1_best.pt frontend/dist/index.html; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
chown -R "$RUN_USER" "$ROOT"

if [ "$SERVICE" = 1 ]; then
    cat > /etc/systemd/system/examguard.service <<EOF
[Unit]
Description=ExamGuard exam monitoring (FastAPI + GPU AI)
After=network-online.target postgresql.service
Wants=postgresql.service

[Service]
User=$RUN_USER
WorkingDirectory=$ROOT/backend
EnvironmentFile=$ROOT/.env
ExecStart=$ROOT/backend/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port \${PORT} --proxy-headers
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now examguard
    echo "service started: systemctl status examguard / journalctl -u examguard -f"
fi
echo
echo "Installed. Admin login: $ADMIN_USERNAME / $ADMIN_PASSWORD  (see .env; a restored database keeps its own accounts)"
[ "$SERVICE" = 1 ] || echo "Start: bash deploy/linux/start.sh   then open http://<server-ip>:$PORT"
