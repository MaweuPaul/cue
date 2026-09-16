"""Watches the rolling transcript for questions and gets fast streamed answers
from Gemini."""
from __future__ import annotations

import os
import re
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Iterator

from google import genai
from google.genai import types

# Same question can arrive twice in quick succession — once via mic, once via
# system-audio loopback — worded slightly differently by the ASR each time
# (mic picking up acoustic leakage of the speakers). Treat anything similar
# enough within this window as a duplicate rather than a new question.
DEDUP_WINDOW_SECONDS = 12.0
DEDUP_SIMILARITY_THRESHOLD = 0.6


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

SYSTEM_PROMPT = (
    "You are a fast, real-time meeting/interview assistant. You are shown a rolling "
    "transcript of live audio (from the user's microphone and from audio playing on "
    "their device) and the most recently transcribed chunk of it. Every kind of chunk "
    "gets sent to you — real questions, interview-style imperative prompts with no "
    "question mark ('Tell me about yourself'), but also filler narration, ads, "
    "mid-sentence fragments, and incomplete or garbled ASR output. Use your judgment: "
    "if it's a genuine question or prompt directed at the listener, answer it using the "
    "transcript for context. If it's anything else — narration, an incomplete fragment, "
    "background noise transcribed as words, an ad, etc. — reply with exactly: SKIP. "
    "When in doubt, SKIP rather than guess. "
    "Respond the way a knowledgeable person would say it out loud, not like an encyclopedia "
    "or search result: natural, direct, conversational phrasing, first person where it fits "
    "(e.g. \"I'd say it's...\", \"That'd be...\"), no headers, bullet points, or bold text, and no "
    "throat-clearing like \"Great question\" or restating the question back. "
    "Be brief: 1-2 short sentences by default, 3 at the absolute most for something that "
    "genuinely needs more. Say the key point only, skip preamble and caveats."
)


class QAEngine:
    def __init__(self, model: str | None = None, context_turns: int = 40):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.transcript: deque[str] = deque(maxlen=context_turns)
        self._recent_answered: list[tuple[float, str]] = []  # (timestamp, normalized text)
        self.topic_hint = ""

    def set_topic(self, topic: str):
        self.topic_hint = topic.strip()

    def add_transcript_line(self, source: str, text: str):
        if not text:
            return
        self.transcript.append(f"[{source}] {text}")

    def context_text(self) -> str:
        return "\n".join(self.transcript)

    def _is_recent_duplicate(self, normalized: str) -> bool:
        now = time.time()
        self._recent_answered = [
            (ts, t) for ts, t in self._recent_answered if now - ts < DEDUP_WINDOW_SECONDS
        ]
        for _, prev in self._recent_answered:
            if SequenceMatcher(None, normalized, prev).ratio() >= DEDUP_SIMILARITY_THRESHOLD:
                return True
        return False

    def maybe_answer(self, candidate_question: str) -> Iterator[str] | None:
        """Returns a streaming iterator of answer text deltas, or None if this
        isn't worth answering. No local keyword pre-filter — everything gets
        sent to Gemini, which decides via SKIP using its own judgment (handles
        buried/garbled/imperative questions far better than regex ever could)."""
        q = candidate_question.strip()
        if not q:
            return None
        normalized = _normalize(q)
        if self._is_recent_duplicate(normalized):
            return None
        self._recent_answered.append((time.time(), normalized))
        return self._stream_answer(q)

    def _stream_answer(self, question: str) -> Iterator[str]:
        user_msg = (
            f"Transcript so far:\n{self.context_text()}\n\n"
            f"Most recent question to answer: {question}"
        )
        system_prompt = SYSTEM_PROMPT
        if self.topic_hint:
            system_prompt += (
                f"\n\nThe user has told you in advance what this session is about: "
                f'"{self.topic_hint}". Use this to judge whether a detected question is '
                "relevant and worth answering, and to interpret ambiguous or mis-transcribed "
                "questions in that light — but still reply SKIP for anything that isn't "
                "actually a question, even if it's on-topic."
            )
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=120,
            ),
        )
        # Gemini is told to reply with exactly "SKIP" for a non-question — check
        # just the first non-empty chunk so we don't buffer (and delay) real answers.
        buffered = ""
        checked_skip = False
        for chunk in stream:
            if not chunk.text:
                continue
            if not checked_skip:
                buffered += chunk.text
                if len(buffered) < 5 and "SKIP".startswith(buffered.strip().upper()):
                    continue  # need more text to be sure it's not "SKIP"
                checked_skip = True
                if buffered.strip().upper() == "SKIP":
                    return
                yield buffered
            else:
                yield chunk.text
        if not checked_skip and buffered and buffered.strip().upper() != "SKIP":
            yield buffered
