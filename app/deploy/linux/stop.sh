#!/bin/bash
# Stop the background ExamGuard started by start.sh (the systemd service: sudo systemctl stop examguard).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$ROOT/logs/backend.pid" ] && kill "$(cat "$ROOT/logs/backend.pid")" 2> /dev/null; then
    echo "stopped"
else
    echo "not running"
fi
rm -f "$ROOT/logs/backend.pid"
