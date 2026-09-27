FROM python:3.11.15-slim-bookworm

ARG UV_VERSION=0.12.5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock ./
ARG TORCH_FLAVOR=cpu
ARG UV_HTTP_TIMEOUT=300
ARG UV_HTTP_CONNECT_TIMEOUT=30
ARG UV_HTTP_RETRIES=8
ARG UV_CONCURRENT_DOWNLOADS=2
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    case "$TORCH_FLAVOR" in \
        cpu|cu126) ;; \
        *) echo "TORCH_FLAVOR không hợp lệ: $TORCH_FLAVOR"; exit 2 ;; \
    esac \
    && UV_HTTP_TIMEOUT="$UV_HTTP_TIMEOUT" \
        UV_HTTP_CONNECT_TIMEOUT="$UV_HTTP_CONNECT_TIMEOUT" \
        UV_HTTP_RETRIES="$UV_HTTP_RETRIES" \
        UV_CONCURRENT_DOWNLOADS="$UV_CONCURRENT_DOWNLOADS" \
        uv sync \
            --frozen \
            --no-dev \
            --no-install-project \
            --extra "$TORCH_FLAVOR"

COPY . .

RUN python -c "import fastapi, pydantic, sqlalchemy, cv2, numpy, torch, torchvision; print('runtime imports OK')"

RUN chmod +x /app/scripts/start_backend.sh

EXPOSE 8000

CMD ["/app/scripts/start_backend.sh"]
