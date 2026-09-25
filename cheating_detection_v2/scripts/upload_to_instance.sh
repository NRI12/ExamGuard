#!/bin/bash
# Run on your own machine (Linux / macOS / Git Bash on Windows): copy this project's code, configs, weights and
# reference results to a remote GPU machine. The dataset is not sent; `download` fetches it there much faster.
#
#   bash scripts/upload_to_instance.sh <ssh_port> <user@host> [remote_dir]
#   bash scripts/upload_to_instance.sh 21935 root@<GPU_HOST_IP> /workspace
set -euo pipefail
PORT=${1:?ssh port}; HOST=${2:?user@host}; DEST=${3:-/workspace}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
NAME=$(basename "$PROJECT_DIR")
tar -C "$(dirname "$PROJECT_DIR")" -czf - --anchored     --exclude="$NAME/data" --exclude="$NAME/outputs/runs" --exclude="$NAME/.venv" --exclude="*/__pycache__" "$NAME"   | ssh -o StrictHostKeyChecking=accept-new -p "$PORT" "$HOST" "mkdir -p '$DEST' && tar -xzf - -C '$DEST'"
echo "uploaded to $HOST:$DEST/$NAME"
echo "next: ssh -p $PORT $HOST 'cd $DEST/$NAME && bash scripts/setup_instance.sh'"
