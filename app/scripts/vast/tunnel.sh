#!/bin/bash
# HTTPS for ExamGuard on Vast.ai: a Cloudflare quick tunnel to the local nginx. Secure cookies (media, WebSocket)
# need HTTPS in production. Prints the public https://*.trycloudflare.com URL (it changes on every restart).
#
#   bash scripts/vast/tunnel.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PUBLIC_PORT=${PUBLIC_PORT:-8000}
mkdir -p "$ROOT/logs"
if [ -f "$ROOT/logs/tunnel.pid" ] && kill -0 "$(cat "$ROOT/logs/tunnel.pid")" 2> /dev/null; then
    kill "$(cat "$ROOT/logs/tunnel.pid")"; sleep 1
fi
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PUBLIC_PORT" > "$ROOT/logs/tunnel.log" 2>&1 &
echo $! > "$ROOT/logs/tunnel.pid"
for _ in $(seq 1 40); do
    url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/logs/tunnel.log" | head -1 || true)
    if [ -n "$url" ]; then echo "$url"; exit 0; fi
    sleep 1
done
tail -20 "$ROOT/logs/tunnel.log"; exit 1
