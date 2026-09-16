"""Real-time streaming transcription via Deepgram, as an alternative to the
local-Whisper batch pipeline in listener_core.py. Same event-callback
interface (start/stop, on_event dict messages) so server.py can use either.

Audio is fed to Deepgram in small (~250ms), non-overlapping frames as it's
captured — Deepgram does its own buffering/VAD/endpointing, so there's no
fixed "wait N seconds before transcribing" latency like the local pipeline
has."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
from deepgram import DeepgramClient
from deepgram.listen.v1.socket_client import EventType, ListenV1Results, V1SocketClient

from audio_capture import AudioCapture
from listener_core import CaptureToggle, answer_question
from qa_engine import QAEngine

EventCallback = Callable[[dict], None]
log = logging.getLogger("listener")

# Deepgram fires `is_final` for each stabilized chunk within a sentence (e.g.
# on a brief pause) well before the speaker is actually done. Rather than
# trying to detect "end of utterance" ourselves (speech_final, max-duration
# caps, etc. — all of which kept failing in different ways), just buffer
# per-source and flush whatever's there once this much silence passes since
# the last chunk. Simple, and Gemini sees the full rolling transcript as
# context anyway, so it can piece together multi-part questions on its own.
IDLE_FLUSH_SECONDS = 1.5


class DeepgramListenerLoop:
    def __init__(
        self,
        api_key: str,
        qa: QAEngine | None,
        on_event: EventCallback,
        capture_mic: bool = True,
        capture_system: bool = True,
        toggle: CaptureToggle | None = None,
    ):
        self.client = DeepgramClient(api_key=api_key)
        self.qa = qa
        self.on_event = on_event
        self.toggle = toggle or CaptureToggle(capture_mic, capture_system)
        # small, non-overlapping frames — Deepgram handles buffering/VAD itself
        self.capture = AudioCapture(
            capture_mic=capture_mic, capture_system=capture_system,
            block_seconds=0.25, overlap_seconds=0.0,
        )
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: dict[str, V1SocketClient] = {}
        self._connect_cms: dict[str, object] = {}
        self._buffer_lock = threading.Lock()
        self._utterance_buffers: dict[str, str] = {}
        self._utterance_last_update: dict[str, float] = {}

    def start(self):
        self.capture.start()
        if self.capture.capture_mic:
            self._open_source("mic")
        if self.capture.capture_system:
            self._open_source("system")
        t = threading.Thread(target=self._feed_loop, daemon=True)
        t.start()
        self._threads.append(t)
        flush_t = threading.Thread(target=self._idle_flush_loop, daemon=True)
        flush_t.start()
        self._threads.append(flush_t)

    def _open_source(self, source: str):
        cm = self.client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            endpointing=300,
        )
        socket = cm.__enter__()
        self._connect_cms[source] = cm
        self._sockets[source] = socket

        def on_message(result):
            if not isinstance(result, ListenV1Results) or not result.is_final:
                return
            alts = result.channel.alternatives if result.channel else []
            text = alts[0].transcript.strip() if alts else ""
            if not text:
                return
            self._handle_transcript(source, text)

        socket.on(EventType.MESSAGE, on_message)
        socket.on(EventType.ERROR, lambda exc: log.error("deepgram %s error: %s", source, exc))

        listener_thread = threading.Thread(target=socket.start_listening, daemon=True)
        listener_thread.start()
        self._threads.append(listener_thread)

    def _feed_loop(self):
        while not self._stop_event.is_set():
            try:
                chunk = self.capture.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            socket = self._sockets.get(chunk.source)
            if socket is None:
                continue
            pcm16 = (np.clip(chunk.audio, -1, 1) * 32767).astype(np.int16).tobytes()
            try:
                socket.send_media(pcm16)
            except Exception:  # noqa: BLE001
                log.exception("failed to send audio to deepgram (%s)", chunk.source)

    def _handle_transcript(self, source: str, text: str):
        if not self.toggle.is_enabled(source):
            with self._buffer_lock:
                self._utterance_buffers.pop(source, None)
                self._utterance_last_update.pop(source, None)
            return
        log.info("transcribed [%s]: %r", source, text)
        self.on_event({"type": "transcript", "source": source, "text": text})

        with self._buffer_lock:
            buf = self._utterance_buffers.get(source, "")
            self._utterance_buffers[source] = f"{buf} {text}".strip() if buf else text
            self._utterance_last_update[source] = time.time()

    def _flush_utterance(self, source: str):
        with self._buffer_lock:
            full_utterance = self._utterance_buffers.pop(source, "").strip()
            self._utterance_last_update.pop(source, None)
        if not full_utterance or self.qa is None:
            return
        self.qa.add_transcript_line(source, full_utterance)
        answer_question(self.qa, self.on_event, full_utterance)

    def _idle_flush_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.3)
            now = time.time()
            with self._buffer_lock:
                idle_sources = [
                    src for src, ts in self._utterance_last_update.items()
                    if now - ts >= IDLE_FLUSH_SECONDS
                ]
            for src in idle_sources:
                self._flush_utterance(src)

    def stop(self):
        self._stop_event.set()
        self.capture.stop()
        for socket in self._sockets.values():
            try:
                socket.send_close_stream()
            except Exception:  # noqa: BLE001
                pass
        for cm in self._connect_cms.values():
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
