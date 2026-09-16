"""Captures audio from the microphone and/or the device's own output (system audio
loopback) and pushes fixed-length chunks onto a queue for transcription.

Uses `soundcard`, which supports WASAPI loopback recording on Windows so we can
capture "what the computer is playing" without any extra virtual-cable driver.
"""
from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass

import numpy as np
import soundcard as sc
from scipy.signal import resample_poly

TARGET_SR = 16_000  # sample rate whisper expects


@dataclass
class AudioChunk:
    source: str  # "mic" or "system"
    audio: np.ndarray  # float32 mono, 16kHz
    timestamp: float


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


# Windows' per-device mic volume/boost keeps getting reset (e.g. by switching
# default devices for VB-Cable), which silently makes speech too quiet to
# transcribe. Auto-gain the mic in software instead of depending on OS
# settings staying put: scale each block's peak up towards a target level,
# capped so we don't amplify silence/noise into garbage.
AUTO_GAIN_TARGET_PEAK = 0.3
AUTO_GAIN_MAX = 40.0


def _auto_gain(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    gain = min(AUTO_GAIN_TARGET_PEAK / peak, AUTO_GAIN_MAX)
    if gain <= 1.0:
        return audio
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


class SourceRecorder(threading.Thread):
    """Records one audio source in fixed-length, overlapping blocks."""

    def __init__(
        self,
        source_name: str,
        recorder,  # soundcard Microphone/loopback recorder (context manager)
        sample_rate: int,
        out_queue: "queue.Queue[AudioChunk]",
        block_seconds: float = 4.0,
        overlap_seconds: float = 1.0,
    ):
        super().__init__(daemon=True, name=f"recorder-{source_name}")
        self.source_name = source_name
        self.recorder_factory = recorder
        self.sample_rate = sample_rate
        self.out_queue = out_queue
        self.block_seconds = block_seconds
        self.overlap_seconds = overlap_seconds
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        import time

        block_frames = int(self.block_seconds * self.sample_rate)
        overlap_frames = int(self.overlap_seconds * self.sample_rate)
        step_frames = block_frames - overlap_frames

        # WASAPI loopback/mic devices can throw (e.g. on sleep/device change,
        # HRESULT errors like 0x88890004) — reopen the device instead of
        # letting the whole recorder thread die silently.
        while not self._stop_event.is_set():
            try:
                with self.recorder_factory.recorder(samplerate=self.sample_rate, channels=1) as rec:
                    buf = np.zeros(0, dtype=np.float32)
                    while not self._stop_event.is_set():
                        data = rec.record(numframes=step_frames)
                        mono = data[:, 0].astype(np.float32)
                        buf = np.concatenate([buf, mono])
                        if len(buf) >= block_frames:
                            block = buf[-block_frames:]
                            buf = buf[-overlap_frames:] if overlap_frames else np.zeros(0, dtype=np.float32)
                            if self.source_name == "mic":
                                block = _auto_gain(block)
                            resampled = _resample(block, self.sample_rate)
                            self.out_queue.put(
                                AudioChunk(source=self.source_name, audio=resampled, timestamp=time.time())
                            )
            except Exception:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                import logging

                logging.getLogger("listener").exception(
                    "%s recorder crashed, reopening in 1s", self.source_name
                )
                self._stop_event.wait(1)


_VIRTUAL_DEVICE_MARKERS = ("cable", "vb-audio", "virtual")


def _pick_real_microphone():
    """Windows (or VB-Cable itself) can silently reclaim the default recording
    device as the VB-Cable virtual device — which then "records" as pure
    silence since nothing feeds directly into it. Actively avoid picking a
    virtual device as the mic, regardless of what Windows currently reports
    as default."""
    import logging

    log = logging.getLogger("listener")
    default = sc.default_microphone()
    if not any(marker in default.name.lower() for marker in _VIRTUAL_DEVICE_MARKERS):
        return default

    log.warning(
        "default mic %r looks like a virtual/loopback device, not a real "
        "microphone — searching for a real one instead", default.name,
    )
    for candidate in sc.all_microphones(include_loopback=False):
        if not any(marker in candidate.name.lower() for marker in _VIRTUAL_DEVICE_MARKERS):
            log.warning("using %r as the mic instead", candidate.name)
            return candidate

    log.warning("no non-virtual microphone found — falling back to the virtual device anyway")
    return default


class AudioCapture:
    """Owns the mic + system-loopback recorder threads and a shared output queue."""

    def __init__(
        self,
        capture_mic: bool = True,
        capture_system: bool = True,
        block_seconds: float = 1.5,
        overlap_seconds: float = 0.3,
    ):
        self.queue: "queue.Queue[AudioChunk]" = queue.Queue()
        self._recorders: list[SourceRecorder] = []
        self.capture_mic = capture_mic
        self.capture_system = capture_system
        self.block_seconds = block_seconds
        self.overlap_seconds = overlap_seconds

    def start(self):
        if self.capture_mic:
            mic = _pick_real_microphone()
            self._recorders.append(
                SourceRecorder(
                    "mic", mic, 48_000, self.queue,
                    self.block_seconds, self.overlap_seconds,
                )
            )
        if self.capture_system:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
            self._recorders.append(
                SourceRecorder(
                    "system", loopback, 48_000, self.queue,
                    self.block_seconds, self.overlap_seconds,
                )
            )
        for r in self._recorders:
            r.start()

    def stop(self):
        for r in self._recorders:
            r.stop()
        for r in self._recorders:
            r.join(timeout=2)
