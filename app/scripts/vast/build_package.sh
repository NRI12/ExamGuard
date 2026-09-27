#!/bin/bash
# Build the portable deployment package from the running Vast.ai deployment:
#   source tree + built frontend + models + uploaded videos + database dump (pg_dump -Fc)
#
#   bash scripts/vast/build_package.sh /workspace/pkg     -> /workspace/pkg/ExamGuard (+ ExamGuard.tar)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/workspace/pkg}
PKG="$OUT/ExamGuard"
set -a; source "$ROOT/.env"; set +a
rm -rf "$PKG"; mkdir -p "$PKG"
tar -C "$ROOT" -cf - \
    --exclude=./backend/.venv --exclude=./frontend/node_modules --exclude=./logs --exclude=./.env \
    --exclude=./data --exclude='*/__pycache__' --exclude='*/.pytest_cache' --exclude='*/.ruff_cache' \
    --exclude=./model_artifacts/action . | tar -C "$PKG" -xf -
mkdir -p "$PKG/model_artifacts/action" "$PKG/data/evidence"
cp -r "$ROOT/model_artifacts/action/." "$PKG/model_artifacts/action/" 2> /dev/null || true
cp -r "$ROOT/data/uploads" "$PKG/data/uploads"
touch "$PKG/data/evidence/.gitkeep"
PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc \
    -f "$PKG/data/examguard.dump"
find "$PKG" -name '*.sh' -exec chmod +x {} +
(cd "$PKG" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
du -sh "$PKG"
echo "package: $PKG"
