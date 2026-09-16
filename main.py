"""Live listener: captures mic + system audio, transcribes it in near
real-time, and asks Claude to answer questions as they're detected.

Usage:
    python main.py [--mic-only | --system-only]
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.text import Text

from audio_capture import AudioCapture
from qa_engine import QAEngine
from transcriber import Transcriber

load_dotenv()

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Real-time audio listener + Q&A assistant")
    p.add_argument("--mic-only", action="store_true", help="Only capture the microphone")
    p.add_argument("--system-only", action="store_true", help="Only capture system/loopback audio")
    p.add_argument("--no-qa", action="store_true", help="Transcription only, skip Claude Q&A")
    p.add_argument(
        "--plain", action="store_true",
        help="Plain line-by-line output instead of the live-redraw panel (for non-interactive terminals)",
    )
    return p.parse_args()


def _has_api_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key) and key != "your-api-key-here"


def main():
    args = parse_args()
    capture_mic = not args.system_only
    capture_system = not args.mic_only

    qa_enabled = not args.no_qa and _has_api_key()
    if not args.no_qa and not qa_enabled:
        console.print(
            "[bold yellow]No GEMINI_API_KEY found in .env — running transcription only, "
            "no Q&A answers.[/bold yellow]"
        )

    console.print("[bold cyan]Loading Whisper model...[/bold cyan]")
    transcriber = Transcriber()
    qa = QAEngine() if qa_enabled else None

    capture = AudioCapture(capture_mic=capture_mic, capture_system=capture_system)

    transcript_lines: list[str] = []
    answer_lines: list[str] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def render() -> Text:
        body = Text()
        body.append("Listening", style="bold green")
        body.append(f"  (mic={'on' if capture_mic else 'off'}, system={'on' if capture_system else 'off'})\n\n")
        body.append("Transcript\n", style="bold underline")
        for line in transcript_lines[-15:]:
            body.append(line + "\n")
        body.append("\nAnswers\n", style="bold underline")
        for line in answer_lines[-10:]:
            body.append(line + "\n", style="yellow")
        return body

    def worker(live: Live | None):
        while not stop_event.is_set():
            try:
                chunk = capture.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            text = transcriber.transcribe(chunk.audio)
            if not text:
                continue
            with lock:
                transcript_lines.append(f"[{chunk.source}] {text}")
                if qa is not None:
                    qa.add_transcript_line(chunk.source, text)
                if live is not None:
                    live.update(render())
                else:
                    console.print(f"[{chunk.source}] {text}")

            if qa is None:
                continue
            stream = qa.maybe_answer(text)
            if stream is None:
                continue
            with lock:
                answer_lines.append(f"Q: {text}")
                answer_lines.append("A: ")
                if live is not None:
                    live.update(render())
                else:
                    console.print(f"Q: {text}", style="yellow")
                    console.print("A: ", style="yellow", end="")
            try:
                for delta in stream:
                    with lock:
                        answer_lines[-1] += delta
                        if live is not None:
                            live.update(render())
                        else:
                            console.print(delta, style="yellow", end="")
                if live is None:
                    console.print()
            except Exception as e:  # noqa: BLE001
                with lock:
                    answer_lines.append(f"[error getting answer: {e}]")
                    if live is not None:
                        live.update(render())
                    else:
                        console.print(f"[error getting answer: {e}]", style="red")

    try:
        capture.start()
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Failed to start audio capture:[/bold red] {e}")
        sys.exit(1)

    console.print(
        f"[bold green]Listening[/bold green] (mic={'on' if capture_mic else 'off'}, "
        f"system={'on' if capture_system else 'off'}) — Ctrl+C to stop"
    )

    if args.plain:
        t = threading.Thread(target=worker, args=(None,), daemon=True)
        t.start()
        try:
            while True:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            capture.stop()
    else:
        with Live(render(), console=console, refresh_per_second=8) as live:
            t = threading.Thread(target=worker, args=(live,), daemon=True)
            t.start()
            try:
                while True:
                    threading.Event().wait(1)
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()
                capture.stop()


if __name__ == "__main__":
    main()
