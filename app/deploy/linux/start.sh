#!/bin/bash
# Start ExamGuard in the background (without the systemd service): http://<server-ip>:$PORT
# The first start after an install compiles the X3D model on the GPU (about a minute).
#
#   bash deploy/linux/start.sh        bash deploy/linux/stop.sh        tail -f logs/backend.log
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
set -a; source .env; set +a
mkdir -p logs
pg_isready -q -h 127.0.0.1 || sudo pg_ctlcluster "$(pg_lsclusters -h | awk 'NR==1{print $1}')" main start
if [ -f logs/backend.pid ] && kill -0 "$(cat logs/backend.pid)" 2> /dev/null; then
    echo "already running (pid $(cat logs/backend.pid))"; exit 0
fi
cd backend
nohup .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers \
    > "$ROOT/logs/backend.log" 2>&1 &
echo $! > "$ROOT/logs/backend.pid"
cd "$ROOT"
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT:-8000}/api/v1/health" > /dev/null; then
        echo "ExamGuard running: http://$(hostname -I 2> /dev/null | awk '{print $1}'):${PORT:-8000}"
        exit 0
    fi
    kill -0 "$(cat logs/backend.pid)" 2> /dev/null || { tail -30 logs/backend.log; exit 1; }
    sleep 2
done
tail -30 logs/backend.log; exit 1
