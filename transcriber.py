"""Thin wrapper around faster-whisper for fast local transcription of short
audio blocks (mono float32 @ 16kHz)."""
from __future__ import annotations

import os

import numpy as np
from faster_whisper import WhisperModel

MIN_RMS = 0.004  # skip near-silent blocks so we don't feed whisper (and Claude) noise


class Transcriber:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        model_size = model_size or os.environ.get("WHISPER_MODEL", "base.en")
        device = device or os.environ.get("WHISPER_DEVICE", "cpu")
        compute_type = compute_type or os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < MIN_RMS:
            return ""
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
