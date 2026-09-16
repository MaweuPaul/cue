"""Runs the mic+system audio capture -> transcribe -> Q&A pipeline on this
machine (the PC) and reports events (transcript lines, questions, answer
deltas) through a callback. Shared by main.py (CLI) and server.py (phone
viewer, which broadcasts these same events over a WebSocket)."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from audio_capture import AudioCapture
from qa_engine import QAEngine
from transcriber import Transcriber

EventCallback = Callable[[dict], None]
log = logging.getLogger("listener")


class CaptureToggle:
    """Lets the UI mute/unmute a source live without restarting the server or
    reconnecting to Deepgram. The recorder threads always run; this just gates
    whether their audio gets processed/answered."""

    def __init__(self, mic: bool = True, system: bool = True):
        self._lock = threading.Lock()
        self._enabled = {"mic": mic, "system": system}

    def is_enabled(self, source: str) -> bool:
        with self._lock:
            return self._enabled.get(source, True)

    def set_enabled(self, source: str, enabled: bool):
        with self._lock:
            self._enabled[source] = enabled

    def state(self) -> dict:
        with self._lock:
            return dict(self._enabled)


class ListenerLoop:
    def __init__(
        self,
        transcriber: Transcriber,
        qa: QAEngine | None,
        on_event: EventCallback,
        capture_mic: bool = True,
        capture_system: bool = True,
        toggle: CaptureToggle | None = None,
    ):
        self.transcriber = transcriber
        self.qa = qa
        self.on_event = on_event
        self.toggle = toggle or CaptureToggle(capture_mic, capture_system)
        self.capture = AudioCapture(capture_mic=capture_mic, capture_system=capture_system)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.capture.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.capture.stop()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                chunk = self.capture.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.toggle.is_enabled(chunk.source):
                continue
            text = self.transcriber.transcribe(chunk.audio)
            if not text:
                continue
            log.info("transcribed [%s]: %r", chunk.source, text)
            self.on_event({"type": "transcript", "source": chunk.source, "text": text})

            if self.qa is None:
                continue
            self.qa.add_transcript_line(chunk.source, text)
            answer_question(self.qa, self.on_event, text)


def answer_question(qa: QAEngine, on_event: EventCallback, text: str):
    """Shared by ListenerLoop (local Whisper) and DeepgramListenerLoop
    (streaming) once either has a finalized line of transcript text."""
    stream = qa.maybe_answer(text)
    if stream is None:
        return
    try:
        it = iter(stream)
        first_delta = next(it, None)
    except Exception as e:  # noqa: BLE001
        log.exception("error answering question")
        on_event({"type": "error", "text": str(e)})
        return
    if first_delta is None:
        log.info("question skipped by model: %r", text)
        return
    log.info("question detected: %r", text)
    on_event({"type": "question", "text": text})
    on_event({"type": "answer_delta", "text": first_delta})
    try:
        answer = first_delta
        for delta in it:
            answer += delta
            on_event({"type": "answer_delta", "text": delta})
        log.info("answer: %r", answer)
        on_event({"type": "answer_done"})
    except Exception as e:  # noqa: BLE001
        log.exception("error answering question")
        on_event({"type": "error", "text": str(e)})
