# cue

Real-time listening assistant: captures audio on your PC — your microphone
**and** whatever's playing through the speakers (a call, a video, anything) —
transcribes it live, and answers questions as they come up using Gemini. A
lightweight web page lets you (or your phone, over WiFi) watch the transcript
and answers stream in live; no recording ever happens on the viewing device.

## How it works

```
 mic ──┐
       ├─► audio_capture.py ─► transcription ─► qa_engine.py ─► server.py ─► browser (PC / phone)
system ┘         │                  │                              │
                soundcard      Deepgram (live)              WebSocket broadcast
                (WASAPI)       or faster-whisper (batch)
```

- [audio_capture.py](audio_capture.py) — records mic + system-output audio via
  `soundcard` (WASAPI loopback on Windows, no virtual cable needed for wired
  output — see the Bluetooth note below), in small chunks. Auto-recovers if a
  device throws (sleep/wake, device changes, etc).
- **Transcription** — two interchangeable backends, picked automatically:
  - [deepgram_stream.py](deepgram_stream.py) — real streaming transcription
    (sub-second latency) if `DEEPGRAM_API_KEY` is set. This is the
    recommended path. Deepgram fires a stabilized `is_final` chunk every time
    the speaker briefly pauses mid-sentence, well before they're actually
    done — so this buffers those per source and only treats it as one
    complete utterance once Deepgram's `speech_final` says the speaker is
    done, or (fallback) once ~6s pass with no new speech for that source, so
    a paused video or a stuck buffer doesn't wait forever.
  - [transcriber.py](transcriber.py) — local, offline `faster-whisper` if no
    Deepgram key is configured. Slower (batch, fixed-size chunks) but fully
    free and private.
- [qa_engine.py](qa_engine.py) — keeps a rolling transcript, dedupes
  near-identical questions that show up on both the mic and system-audio
  channels (acoustic leakage), optionally applies a live "topic hint" (and/or
  a pasted resume/background) you set from the page, and streams a short,
  conversational answer from Gemini. There's no local keyword filter —
  *every* transcribed chunk goes to Gemini, which decides for itself whether
  it's a real question worth answering (replying with `SKIP` otherwise, cheap
  since it's checked on the first chunk of the response). This handles
  garbled ASR output, filler narration, and questions buried mid-sentence far
  better than a regex ever could — the tradeoff is more API calls, well
  within Gemini's free daily quota for personal use.
- [listener_core.py](listener_core.py) — shared local-Whisper capture loop,
  the question/answer handling logic re-used by both backends, and
  `CaptureToggle`, which lets the web page mute/unmute mic or system audio
  live without restarting anything.
- [server.py](server.py) — runs the listener on your PC and broadcasts
  transcript/question/answer events over WebSocket to any connected browser
  (PC or phone, same WiFi). Serves [static/index.html](static/index.html), a
  small live-updating viewer page. PIN-protected.
- [main.py](main.py) — CLI alternative to the web viewer: same pipeline, but
  prints to your terminal instead (local-Whisper only). Useful for quick
  local testing.

## Installation

**Prerequisites**: Windows 10/11, [Python 3.10+](https://www.python.org/downloads/)
(check with `python --version`; make sure "Add Python to PATH" was checked
during install).

**1. Get the code** onto your machine — copy/clone the `listener` folder
anywhere, e.g. `D:\listener`.

**2. Open a terminal in that folder** and create an isolated environment so
this doesn't touch your other Python projects:

```bash
cd D:/listener
python -m venv .venv
```

**3. Install dependencies** into that environment:

```bash
./.venv/Scripts/pip install -r requirements.txt
```

This pulls in `faster-whisper`, `soundcard`, `fastapi`, `google-genai`,
`deepgram-sdk`, and a few smaller packages — first run takes a minute or two.

**4. Create your config file:**

```bash
cp .env.example .env
```

Then open `.env` in any text editor and fill in:
- `GEMINI_API_KEY` — **required** for Q&A answers. Get a free key at
  https://aistudio.google.com/apikey (no card needed).
- `DEEPGRAM_API_KEY` — optional but strongly recommended, for real-time
  streaming transcription instead of the slower local fallback. Free $200
  credit, no card required: https://console.deepgram.com/signup
- `ACCESS_PIN` — pick any PIN; required to connect to the viewer page.
- `CAPTURE_MIC` / `CAPTURE_SYSTEM_AUDIO` — starting state for each source
  (`true`/`false`); both are also toggleable live from the page.

**5. (Optional) If your PC's default audio output is Bluetooth headphones**,
see the Bluetooth note under Notes below before running — system-audio
capture won't work until that's set up.

**6. Run it** — see the next section.

## Run (phone/web viewer — recommended)

```bash
D:/listener/.venv/Scripts/python D:/listener/server.py
```

Then open `http://<this-pc's-LAN-IP>:8000` in a browser — on the PC itself,
or on your phone if it's on the same WiFi network. Enter the PIN, and you'll
see:
- **Mic / System audio toggles** — mute either source instantly, synced
  across every connected device.
- **Guide the AI** — a text box (paste as much as you want, including a full
  resume) telling the model what to expect; it collapses to a short summary
  after you hit Set (click it, or the show/hide button, to expand again).
- A live **Q&A** feed and a collapsible raw **Transcript**.

No microphone access happens in the browser — the PC does all the capturing,
so this works over plain HTTP with no certificate hassle.

## Run (terminal, local-only)

```bash
D:/listener/.venv/Scripts/python D:/listener/main.py
```

Flags:
- `--mic-only` / `--system-only` — capture just one source.
- `--no-qa` — transcription only, skip Gemini.
- `--plain` — plain line-by-line output instead of the live-redraw panel
  (useful when piping output or running non-interactively).

Press `Ctrl+C` to stop. This path always uses local `faster-whisper`, even if
`DEEPGRAM_API_KEY` is set.

## Notes

- **Bluetooth headphones**: WASAPI loopback capture returns silence on
  Bluetooth output devices (A2DP) — it's a Windows limitation, not fixable in
  code. If your default output is Bluetooth, install the free
  [VB-Audio Virtual Cable](https://vb-audio.com/Cable/), set **CABLE Input**
  as your default playback device, then in the classic Sound control panel
  (`mmsys.cpl`) → Recording tab → **CABLE Output** → Properties → Listen tab
  → enable "Listen to this device" → route it to your Bluetooth headphones.
  System-audio capture then works normally (it just targets whatever the
  Windows default output device is) while you still hear everything.
- **Privacy**: Gemini's free tier says content "used to improve our
  products" (per Google's pricing page) — the paid tier does not. Since this
  streams live conversation context to Gemini, keep that in mind for
  sensitive use.
- **Deepgram cost**: with system + mic both on, two streaming connections run
  for as long as the server is up (billed by connected-minutes, not just
  detected speech) — a $200 free credit comfortably covers hundreds of hours
  of personal use.
- First local-Whisper run downloads the model (`small.en` by default) from
  Hugging Face and caches it. Tune via `.env`: `WHISPER_MODEL`,
  `WHISPER_DEVICE` (`cpu`/`cuda`), `WHISPER_COMPUTE_TYPE`.
- If mic and system audio pick up the same speech twice (mic hearing your own
  speakers), the fuzzy dedup in `qa_engine.py` usually collapses it into one
  answer; toggle one source off from the page if it's still noisy.
