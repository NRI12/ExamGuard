#!/bin/bash
# Configure and (re)start ExamGuard on a Vast.ai instance without Docker:
#   PostgreSQL 16 (localhost) -> Alembic migrations -> admin bootstrap
#   uvicorn (backend, 127.0.0.1:8001) behind nginx (public port, default 8000):
#   frontend build (static), /api and /ws proxied to the backend, HTTP Range for videos.
# Secrets are generated once into .env (kept on the instance, never committed).
#
#   bash scripts/vast/deploy.sh                 # after scripts/vast/setup_instance.sh
#   PUBLIC_PORT=8000 bash scripts/vast/deploy.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PUBLIC_PORT=${PUBLIC_PORT:-8000}
BACKEND_PORT=${BACKEND_PORT:-8001}
PY="$ROOT/backend/.venv/bin/python"

echo "== .env"
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
MAX_UPLOAD_BYTES=4294967296
FFPROBE_TIMEOUT_SECONDS=30
ADMIN_USERNAME=admin
ADMIN_PASSWORD=$(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | head -c 14)Aa1
ADMIN_FULL_NAME="Quản trị viên"
EOF
    chmod 600 .env
fi
set -a; source .env; set +a
mkdir -p "$UPLOAD_ROOT" "$EVIDENCE_ROOT" "$MODEL_ROOT/detection" "$MODEL_ROOT/cheating" logs

echo "== postgres"
if ! pg_lsclusters -h | grep -q online; then pg_ctlcluster 16 main start; fi
for i in $(seq 1 30); do pg_isready -q -h 127.0.0.1 && break; sleep 1; done
su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='$POSTGRES_USER'\"" | grep -q 1 \
    || su postgres -c "psql -c \"CREATE ROLE $POSTGRES_USER LOGIN PASSWORD '$POSTGRES_PASSWORD'\""
su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'\"" | grep -q 1 \
    || su postgres -c "createdb -O $POSTGRES_USER $POSTGRES_DB"

echo "== models"
"$PY" backend/scripts/download_yolo11n.py "$MODEL_ROOT/detection/yolo11n.pt"
[ -f "$MODEL_ROOT/cheating/x3d_l_v1_best.pt" ] || { echo "missing $MODEL_ROOT/cheating/x3d_l_v1_best.pt"; exit 1; }

echo "== database migrations + admin"
cd backend
"$PY" -m alembic upgrade head
"$PY" -m app.cli.bootstrap_admin
cd "$ROOT"

echo "== frontend build"
cd frontend
[ -d node_modules ] || npm ci --no-audit --no-fund
npm run build
cd "$ROOT"

echo "== nginx"
sed -e "s#listen 80;#listen $PUBLIC_PORT;#" \
    -e "s#root /usr/share/nginx/html;#root $ROOT/frontend/dist;#" \
    -e "s#http://backend:8000#http://127.0.0.1:$BACKEND_PORT#g" \
    frontend/nginx.conf > /etc/nginx/sites-available/examguard
ln -sf /etc/nginx/sites-available/examguard /etc/nginx/sites-enabled/examguard
rm -f /etc/nginx/sites-enabled/default
nginx -t
if pgrep -x nginx > /dev/null; then nginx -s reload; else nginx; fi

echo "== backend"
pkill -f "[u]vicorn app.main:app" && sleep 2 || true
cd backend
nohup "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --proxy-headers \
    > "$ROOT/logs/backend.log" 2>&1 &
cd "$ROOT"
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$BACKEND_PORT/api/v1/health" > /dev/null && break
    sleep 2
done
curl -sf "http://127.0.0.1:$PUBLIC_PORT/api/v1/health" && echo
echo "ExamGuard is up on port $PUBLIC_PORT (backend log: logs/backend.log)"
echo "admin login: $ADMIN_USERNAME / $ADMIN_PASSWORD"
