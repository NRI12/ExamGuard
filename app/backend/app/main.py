import logging
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.ai.event_aggregation.types import AggregatedEvent
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import ApiError, error_payload
from app.db.session import get_session_factory
from app.features.events.service import persist_ai_event
from app.features.monitoring.router import websocket_router
from app.features.monitoring.service import (
    reset_interrupted_sessions,
    terminal_database_update,
)
from app.monitoring.config import load_runtime_profile
from app.monitoring.manager import MonitoringRuntimeManager, RuntimeState

logger = logging.getLogger(__name__)


def _persist_terminal_state(
    session_id: uuid.UUID,
    state: RuntimeState,
    _: str | None,
) -> None:
    with get_session_factory()() as db:
        terminal_database_update(db, session_id, state)


def _persist_ai_event(event: AggregatedEvent) -> None:
    with get_session_factory()() as db:
        persist_ai_event(db, event)


def _reset_interrupted_sessions() -> None:
    try:
        with get_session_factory()() as db:
            count = reset_interrupted_sessions(db)
        if count:
            logger.warning("%s interrupted monitoring session(s) set back to READY", count)
    except Exception:  # no database yet (tests, first start): nothing to reset
        logger.debug("Interrupted session reset skipped", exc_info=True)


def _preload_models(manager: MonitoringRuntimeManager) -> None:
    """Warm the GPU models of the default profile so the first monitoring start is immediate."""
    try:
        manager.preload(load_runtime_profile(get_settings()))
    except Exception:  # the runtime loads them on the first start instead
        logger.warning("Model preload skipped", exc_info=True)


def _mount_frontend(application: FastAPI) -> None:
    """Serve the built single-page frontend when FRONTEND_DIST is set (no nginx needed)."""
    try:
        dist = get_settings().frontend_dist
    except Exception:  # settings are validated again, with a clear error, by the first request
        return
    if dist is None or not (Path(dist) / "index.html").is_file():
        return
    root = Path(dist).resolve()

    @application.get("/{path:path}", include_in_schema=False, response_model=None)
    def frontend(path: str) -> FileResponse:
        if path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404)
        target = (root / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            immutable = path.startswith("assets/")
            cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
            return FileResponse(target, headers={"Cache-Control": cache})
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    _reset_interrupted_sessions()
    manager = MonitoringRuntimeManager(
        terminal_callback=_persist_terminal_state,
        event_callback=_persist_ai_event,
    )
    application.state.monitoring_runtime = manager
    threading.Thread(target=_preload_models, args=(manager,), daemon=True).start()
    try:
        yield
    finally:
        manager.shutdown()


def create_app() -> FastAPI:
    application = FastAPI(
        title="ExamGuard API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.include_router(api_router)
    application.include_router(websocket_router)
    _mount_frontend(application)

    @application.exception_handler(ApiError)
    async def api_error_handler(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content=error_payload(error))

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        api_error = ApiError(
            status_code=422,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": details},
            field_name=(
                str(error.errors()[0]["loc"][-1])
                if error.errors() and error.errors()[0]["loc"]
                else None
            ),
        )
        return JSONResponse(status_code=422, content=error_payload(api_error))

    return application


app = create_app()
