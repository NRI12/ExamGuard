#!/bin/bash
# Optional HTTPS without a domain: Cloudflare quick tunnel to the local ExamGuard. Prints the public
# https://*.trycloudflare.com URL (new URL on every start). With HTTPS, set COOKIE_SECURE=true in .env.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
set -a; source "$ROOT/.env"; set +a
if ! command -v cloudflared > /dev/null; then
    sudo curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    sudo chmod +x /usr/local/bin/cloudflared
fi
mkdir -p "$ROOT/logs"
[ -f "$ROOT/logs/tunnel.pid" ] && kill "$(cat "$ROOT/logs/tunnel.pid")" 2> /dev/null && sleep 1
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT:-8000}" > "$ROOT/logs/tunnel.log" 2>&1 &
echo $! > "$ROOT/logs/tunnel.pid"
for _ in $(seq 1 40); do
    url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/logs/tunnel.log" | head -1 || true)
    [ -n "$url" ] && { echo "$url"; exit 0; }
    sleep 1
done
tail -20 "$ROOT/logs/tunnel.log"; exit 1
