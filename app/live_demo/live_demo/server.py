"""HTTP + WebSocket server of the live demo.

  GET  /                 the demo page (static/)
  GET  /api/manifest     models (with their default decision rules) and videos
  POST /api/upload       raw video bytes (?name=file.mp4), returns the new video (prepared in the background)
  GET  /api/videos/{id}  video status;  DELETE removes an upload
  GET  /media/..., /data/...   videos (HTTP Range, needed for seeking) and ground truth
  WS   /ws               {"type": "open", "video": id, "models": [...]} then clock / seek / models messages;
                         receives det / win / stat / reset

  python -m live_demo.server --host 0.0.0.0 --port 8000
"""
import argparse
import asyncio
import contextlib
import json
import logging
import secrets

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA, DEFAULT_RUN, MAX_SESSIONS, MAX_UPLOAD_BYTES, MEDIA, STATIC, TOKEN
from .pipeline import Analyzer, ModelBank
from .videos import Library

log = logging.getLogger("live_demo")
STATE = {}


@contextlib.asynccontextmanager
async def lifespan(app):
    STATE["bank"] = await asyncio.to_thread(ModelBank)
    STATE["library"] = Library()
    STATE["active"] = 0
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


def _authorised(token, cookie):
    return not TOKEN or secrets.compare_digest(token or cookie or "", TOKEN)


@app.middleware("http")
async def token_gate(request: Request, call_next):
    """With LIVE_TOKEN set, open http://host:port/?token=... once; a cookie keeps the browser authorised."""
    q = request.query_params.get("token")
    if not _authorised(q, request.cookies.get("live_token")):
        return JSONResponse({"detail": "add ?token=... to the URL"}, status_code=401)
    resp = await call_next(request)
    if TOKEN and q:
        resp.set_cookie("live_token", TOKEN, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.get("/api/manifest")
def manifest():
    bank = STATE["bank"]
    return dict(models=bank.specs, sessions=STATE["library"].list(), gpu=bank.gpu_name, bf16=bank.bf16, default_run=DEFAULT_RUN,
                sample_dt=bank.sample_dt, window=round((bank.n_frames - 1) * bank.sample_dt, 2))


@app.get("/api/videos/{vid}")
def video(vid: str):
    v = STATE["library"].get(vid)
    if not v:
        raise HTTPException(404, "unknown video")
    return v.public()


@app.delete("/api/videos/{vid}")
def delete_video(vid: str):
    if not STATE["library"].delete(vid):
        raise HTTPException(404, "not an uploaded video")
    return {"ok": True}


@app.post("/api/upload")
async def upload(request: Request, name: str = "video.mp4"):
    size = int(request.headers.get("content-length") or 0)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"video larger than {MAX_UPLOAD_BYTES >> 20} MB")
    chunks = request.stream()
    loop = asyncio.get_running_loop()
    got = 0

    def write_body(path):
        nonlocal got
        with open(path, "wb") as fh:
            while True:
                try:
                    chunk = asyncio.run_coroutine_threadsafe(chunks.__anext__(), loop).result()
                except StopAsyncIteration:
                    break
                got += len(chunk)
                if got > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "video too large")
                fh.write(chunk)
        if got == 0:
            raise HTTPException(400, "empty upload")

    v = await asyncio.to_thread(STATE["library"].add_upload, name, write_body)
    return v.public()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    if not _authorised(sock.query_params.get("token"), sock.cookies.get("live_token")):
        await sock.close(code=4401)
        return
    await sock.accept()
    loop, queue = asyncio.get_running_loop(), asyncio.Queue()
    emit = lambda msg: loop.call_soon_threadsafe(queue.put_nowait, msg)  # noqa: E731 - called from threads
    analyzer, counted = None, False

    async def sender():
        while True:
            msg = await queue.get()
            await sock.send_text(json.dumps(msg, separators=(",", ":")))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            kind = msg.get("type")
            if kind == "open":
                if analyzer:
                    await asyncio.to_thread(analyzer.close)
                    analyzer = None
                v = STATE["library"].get(msg.get("video"))
                if not v or v.status != "ready":
                    emit(dict(type="error", message="video not ready"))
                    continue
                if not counted and STATE["active"] >= MAX_SESSIONS:
                    emit(dict(type="error", message=f"GPU busy: {MAX_SESSIONS} viewers already analysing"))
                    continue
                if not counted:
                    STATE["active"] += 1
                    counted = True
                analyzer = await asyncio.to_thread(Analyzer, STATE["bank"], v, emit, msg.get("models"))
                emit(dict(type="ready", video=v.id))
            elif analyzer and kind == "clock":
                analyzer.clock(msg["t"], msg.get("playing", False), msg.get("rate", 1.0))
            elif analyzer and kind == "models":
                analyzer.set_models(msg.get("ids") or [])
            elif analyzer and kind == "seek":
                analyzer.seek(msg["t"])
                analyzer.clock(msg["t"], msg.get("playing", False), msg.get("rate", 1.0))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        send_task.cancel()
        if analyzer:
            await asyncio.to_thread(analyzer.close)
        if counted:
            STATE["active"] -= 1


MEDIA.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA), name="media")
app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", ws_ping_interval=20, ws_ping_timeout=60)


if __name__ == "__main__":
    main()
