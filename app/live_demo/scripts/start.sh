#!/bin/bash
# (Re)start the server in the background on port 8000 (logs/server.log). Environment overrides: LIVE_MODELS,
# LIVE_TOKEN, LIVE_MAX_SESSIONS, LIVE_MAX_LAG (see live_demo/config.py).
#
#   bash scripts/start.sh              # all models
#   LIVE_MODELS=v1,v2 bash scripts/start.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
mkdir -p logs
# the access token is remembered in .token, so a plain `bash scripts/start.sh` keeps it
if [ -n "${LIVE_TOKEN:-}" ]; then echo "$LIVE_TOKEN" > .token; chmod 600 .token
elif [ -f .token ]; then export LIVE_TOKEN=$(cat .token); fi
pkill -f "live_demo.server" 2> /dev/null && sleep 2 || true
nohup python -m live_demo.server --host 0.0.0.0 --port "${PORT:-8000}" > logs/server.log 2>&1 &
for i in $(seq 1 600); do  # first start: torch.compile of every model (minutes), cached afterwards
    if curl -sf "http://127.0.0.1:${PORT:-8000}/api/manifest${LIVE_TOKEN:+?token=$LIVE_TOKEN}" > /dev/null; then
        echo "server up (pid $!), log: $(pwd)/logs/server.log"
        exit 0
    fi
    if ! kill -0 $! 2> /dev/null; then tail -30 logs/server.log; echo "server exited"; exit 1; fi
    sleep 2
done
tail -30 logs/server.log; echo "server not answering after 20 min"; exit 1
