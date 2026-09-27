#!/bin/bash
# Run on your machine (Git Bash on Windows / Linux / macOS): send live_demo (code, weights, 720p browser videos)
# to a GPU instance, install, fetch the 1080p sources there and start the server.
# The weights (~175 MB) must be in live_demo/weights (see README). The 1080p videos are not sent: the instance
# downloads them from Google Drive, much faster than an upload.
#
#   bash scripts/deploy.sh <ssh_port> <user@host> [remote_dir]
#   bash scripts/deploy.sh 41234 root@<GPU_HOST_IP> /workspace
set -euo pipefail
PORT=${1:?ssh port}; HOST=${2:?user@host}; DEST=${3:-/workspace}
DIR=$(cd "$(dirname "$0")/.." && pwd)
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p $PORT $HOST"
for w in yolo11m.pt x3d_l_v1_best.pt; do
    [ -f "$DIR/weights/$w" ] || { echo "missing weights/$w"; exit 1; }
done
echo "uploading code + weights + 720p videos ..."
tar -C "$DIR/.." -czf - --exclude="live_demo/media/*_src.mp4" --exclude="live_demo/media/uploads" \
    --exclude="live_demo/media/_download" --exclude="live_demo/logs" --exclude="*/__pycache__" live_demo \
    | $SSH "mkdir -p '$DEST' && tar -xzf - -C '$DEST'"
echo "installing on the instance ..."
$SSH "cd '$DEST/live_demo' && bash scripts/setup_vast.sh && ${LIVE_TOKEN:+LIVE_TOKEN=$LIVE_TOKEN} bash scripts/start.sh"
