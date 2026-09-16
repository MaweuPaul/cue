"""Runs the listener on THIS machine (mic + system audio, like main.py) and
broadcasts live transcript/question/answer events to any phone(s) that open
the viewer page over WebSocket. The phone does no recording at all — it just
watches.

Run: ./.venv/Scripts/python server.py
Then on your phone (same WiFi): https://<this-pc-lan-ip>:8443
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("listener")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # allow running this script from any cwd

from listener_core import CaptureToggle, ListenerLoop  # noqa: E402
from qa_engine import QAEngine  # noqa: E402
from transcriber import Transcriber  # noqa: E402

load_dotenv(BASE_DIR / ".env")

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

clients: set[WebSocket] = set()
main_loop: asyncio.AbstractEventLoop | None = None
listener: ListenerLoop | object | None = None
qa_engine: QAEngine | None = None
capture_toggle: CaptureToggle | None = None


def _has_key(env_var: str) -> bool:
    key = os.environ.get(env_var, "")
    return bool(key) and not key.startswith("your-")


def _on_event(event: dict):
    """Called from the capture thread — hop onto the asyncio loop to broadcast."""
    if main_loop is not None:
        main_loop.call_soon_threadsafe(asyncio.create_task, _broadcast(event))


async def _broadcast(event: dict):
    if not clients:
        return
    msg = json.dumps(event)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


@app.on_event("startup")
async def startup():
    global main_loop, listener, qa_engine, capture_toggle
    main_loop = asyncio.get_running_loop()
    qa = QAEngine() if _has_key("GEMINI_API_KEY") else None
    qa_engine = qa
    if qa is None:
        log.warning("No GEMINI_API_KEY set — broadcasting transcript only, no Q&A answers.")

    capture_system = os.environ.get("CAPTURE_SYSTEM_AUDIO", "true").lower() not in ("0", "false", "no")
    capture_mic = os.environ.get("CAPTURE_MIC", "true").lower() not in ("0", "false", "no")
    sources = " + ".join(filter(None, ["mic" if capture_mic else "", "system audio" if capture_system else ""])) or "nothing (!)"
    capture_toggle = CaptureToggle(mic=capture_mic, system=capture_system)

    if _has_key("DEEPGRAM_API_KEY"):
        from deepgram_stream import DeepgramListenerLoop

        listener = DeepgramListenerLoop(
            os.environ["DEEPGRAM_API_KEY"], qa, _on_event,
            capture_mic=capture_mic, capture_system=capture_system, toggle=capture_toggle,
        )
        log.info("listener started (%s, Deepgram streaming)", sources)
    else:
        transcriber = Transcriber()
        listener = ListenerLoop(
            transcriber, qa, _on_event,
            capture_mic=capture_mic, capture_system=capture_system, toggle=capture_toggle,
        )
        log.info("listener started (%s, local Whisper)", sources)
    listener.start()


@app.on_event("shutdown")
def shutdown():
    if listener is not None:
        listener.stop()


@app.get("/")
def index():
    return FileResponse(
        str(BASE_DIR / "static" / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    expected_pin = os.environ.get("ACCESS_PIN", "")
    submitted_pin = websocket.query_params.get("pin", "")
    if not expected_pin or submitted_pin != expected_pin:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    clients.add(websocket)
    log.info("viewer connected: %s (total: %d)", websocket.client, len(clients))
    if qa_engine is not None and qa_engine.topic_hint:
        await websocket.send_text(json.dumps({"type": "topic", "text": qa_engine.topic_hint}))
    if capture_toggle is not None:
        await websocket.send_text(json.dumps({"type": "capture_state", **capture_toggle.state()}))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "set_topic" and qa_engine is not None:
                topic = str(msg.get("text", ""))[:8000]  # room for a pasted resume
                qa_engine.set_topic(topic)
                log.info("topic hint set: %r", topic)
                await _broadcast({"type": "topic", "text": topic})
            elif msg.get("type") == "set_capture" and capture_toggle is not None:
                source = msg.get("source")
                enabled = bool(msg.get("enabled"))
                if source in ("mic", "system"):
                    capture_toggle.set_enabled(source, enabled)
                    log.info("capture toggle: %s -> %s", source, enabled)
                    await _broadcast({"type": "capture_state", **capture_toggle.state()})
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        log.info("viewer disconnected: %s (total: %d)", websocket.client, len(clients))


if __name__ == "__main__":
    # No mic access happens in the browser anymore (the PC captures audio
    # directly via soundcard) so there's no secure-context requirement —
    # plain HTTP avoids all the self-signed-cert/WSS handshake headaches.
    uvicorn.run(app, host="0.0.0.0", port=8000)
