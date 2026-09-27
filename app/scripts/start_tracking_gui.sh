#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

compose=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml)
model_path="model_artifacts/detection/yolo11n.pt"

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" .env | tail -n 1
}

require_non_placeholder() {
  local key="$1"
  local value
  value="$(env_value "$key")"
  if [[ -z "$value" || "${value,,}" == *change-me* || "${value,,}" == *changeme* || "${value,,}" == *replace-with* ]]; then
    echo "ERROR: $key must be set to a non-placeholder value in .env" >&2
    exit 2
  fi
}

if [[ ! -f .env ]]; then
  echo "ERROR: .env was not found" >&2
  exit 2
fi
if [[ ! -s "$model_path" ]]; then
  echo "ERROR: YOLO11n was not found at $project_dir/$model_path" >&2
  exit 2
fi

require_non_placeholder POSTGRES_PASSWORD
require_non_placeholder JWT_SECRET
require_non_placeholder ADMIN_USERNAME
require_non_placeholder ADMIN_PASSWORD

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${compose[@]}" build backend frontend
fi

"${compose[@]}" up -d postgres
"${compose[@]}" run --rm --no-deps backend alembic upgrade head
"${compose[@]}" run --rm --no-deps backend python -m app.cli.bootstrap_admin
"${compose[@]}" up -d --no-build backend frontend

"${compose[@]}" exec -T backend python -c \
  "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))"

frontend_port="$(env_value FRONTEND_PORT)"
frontend_port="${frontend_port:-8080}"
health_url="http://localhost:${frontend_port}/api/v1/health"
for _ in $(seq 1 30); do
  if curl --fail --silent --show-error "$health_url" >/dev/null; then
    echo "ExamGuard tracking GUI: http://localhost:${frontend_port}"
    echo "YOLO model: $project_dir/$model_path"
    exit 0
  fi
  sleep 2
done

echo "ERROR: backend did not become healthy; inspect logs with:" >&2
echo "  docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=200 backend postgres frontend" >&2
exit 1
