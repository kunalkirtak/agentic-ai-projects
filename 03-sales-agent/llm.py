"""
llm.py — Thin wrapper around the Gemini API.

Keeps all Gemini-specific code in one place, tracks call counts (so the
agent can report how much of the free tier it used), and does its best to
extract clean JSON from model responses without ever throwing away a run
because of a formatting hiccup.

Free-tier friendliness:
- Every call is paced at least GEMINI_MIN_SECONDS_BETWEEN_CALLS apart, so
  the agent doesn't burst past a per-minute rate limit.
- A hard MAX_GEMINI_CALLS_PER_SESSION budget is enforced client-side. Once
  reached, calls fail fast (no network round trip, no wasted quota) with a
  GeminiError that callers already know how to catch and fall back from.
- Rate-limit-shaped errors (429 / quota / resource exhausted) get a much
  longer backoff than ordinary transient errors, since free-tier limits
  reset on the order of a minute, not a couple of seconds.
"""

import json
import re
import time

from config import (
    GEMINI_BACKOFF_SECONDS,
    GEMINI_MAX_RETRIES,
    GEMINI_MIN_SECONDS_BETWEEN_CALLS,
    GEMINI_MODEL,
    MAX_GEMINI_CALLS_PER_SESSION,
)


class GeminiError(RuntimeError):
    """Raised when the Gemini API cannot be reached, errors, or the
    client-side session budget has been used up."""


_RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "quota",
)


class GeminiLLM:
    """
    Minimal wrapper around google-genai's client.

    Usage:
        llm = GeminiLLM(api_key)
        text = llm.generate("Write a haiku about SaaS")
        data = llm.generate_json("Return JSON describing ...")
    """

    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        if not api_key:
            raise GeminiError(
                "No Gemini API key was provided. In Colab, add a secret named "
                "GEMINI_API_KEY (Tools > Secrets) and grant this notebook access."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiError(
                "google-genai is not installed. Run: pip install -q google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.call_count = 0
        self.total_chars_sent = 0
        self.total_chars_received = 0
        self._last_call_time = 0.0

    def _wait_for_pacing(self):
        """Sleep just long enough to keep calls spaced out and avoid RPM limits."""
        elapsed = time.time() - self._last_call_time
        remaining = GEMINI_MIN_SECONDS_BETWEEN_CALLS - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, retries: int = GEMINI_MAX_RETRIES, backoff_seconds: float = GEMINI_BACKOFF_SECONDS) -> str:
        """Send a single prompt to Gemini and return the text response."""
        if self.call_count >= MAX_GEMINI_CALLS_PER_SESSION:
            raise GeminiError(
                f"Session Gemini call budget reached ({MAX_GEMINI_CALLS_PER_SESSION} calls). "
                "Skipping this call and using deterministic fallback logic instead, so the "
                "free-tier key never gets rate-limited or over-quota."
            )

        last_error = None
        for attempt in range(retries + 1):
            self._wait_for_pacing()
            try:
                self.call_count += 1
                self.total_chars_sent += len(prompt)
                self._last_call_time = time.time()
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = (getattr(response, "text", None) or "").strip()
                self.total_chars_received += len(text)
                if not text:
                    raise GeminiError("Gemini returned an empty response.")
                return text
            except Exception as exc:  # noqa: BLE001 - surface any SDK/network error
                last_error = exc
                self._last_call_time = time.time()
                message = str(exc).lower()
                # Don't burn retries on clearly non-transient errors.
                if "api key" in message or "permission" in message or "not found" in message:
                    break
                if self.call_count >= MAX_GEMINI_CALLS_PER_SESSION:
                    break
                if attempt < retries:
                    is_rate_limit = any(marker in message for marker in _RATE_LIMIT_MARKERS)
                    wait = backoff_seconds * (attempt + 1) if is_rate_limit else 2.0 * (attempt + 1)
                    if is_rate_limit:
                        print(f"  ⏳ Gemini rate limit hit; waiting {wait:.0f}s before retrying...")
                    time.sleep(wait)
        raise GeminiError(f"Gemini call failed after retries: {last_error}")

    def generate_json(self, prompt: str, retries: int = GEMINI_MAX_RETRIES) -> dict:
        """
        Ask Gemini for JSON and parse it defensively. Gemini sometimes wraps
        JSON in ```json fences or adds a short preamble — this strips both.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with ONLY valid JSON. No markdown fences, no preamble, "
            "no trailing commentary — just the JSON object."
        )
        raw = self.generate(json_prompt, retries=retries)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fall back to grabbing the first {...} block in the text.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise GeminiError(f"Could not parse JSON from Gemini response: {raw[:300]}")

    def stats(self) -> dict:
        return {
            "gemini_calls": self.call_count,
            "gemini_call_budget": MAX_GEMINI_CALLS_PER_SESSION,
            "gemini_calls_remaining": max(0, MAX_GEMINI_CALLS_PER_SESSION - self.call_count),
            "chars_sent": self.total_chars_sent,
            "chars_received": self.total_chars_received,
        }
