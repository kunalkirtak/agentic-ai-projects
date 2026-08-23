"""
llm.py
------
Thin wrapper around the Gemini API. Everything that talks to the model
goes through GeminiLLM so the rest of the codebase doesn't need to know
or care which SDK/model is behind it.

Deliberately NOT doing any scoring/ranking/arithmetic here - that's all
in scoring.py, in plain Python, so it's deterministic and reproducible.
"""

import json
import re
import time

from google import genai

import config


class GeminiError(Exception):
    """Raised when Gemini can't be reached or returns something unusable."""
    pass


class GeminiLLM:
    def __init__(self, api_key: str, model: str = None):
        if not api_key:
            raise GeminiError(
                "No Gemini API key was provided. In Colab, set a secret named "
                "GEMINI_API_KEY (Tools -> Secrets) and grant this notebook access."
            )
        self.model = model or config.GEMINI_MODEL
        self._client = genai.Client(api_key=api_key)
        self.call_count = 0  # cheap way to track free-tier usage during a run

    def generate(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the raw text response.

        Retries a handful of times with backoff since free-tier keys get
        rate limited fairly easily. If everything fails, raises GeminiError
        rather than letting the whole agent run crash silently.
        """
        last_error = None
        for attempt in range(1, config.GEMINI_MAX_RETRIES + 1):
            try:
                self.call_count += 1
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise GeminiError("Gemini returned an empty response.")
                return text
            except Exception as exc:  # noqa: BLE001 - we want to catch/retry broadly here
                last_error = exc
                if attempt < config.GEMINI_MAX_RETRIES:
                    delay = config.GEMINI_RETRY_BASE_DELAY_SECONDS * attempt
                    print(f"  ⚠ Gemini call failed (attempt {attempt}), retrying in {delay}s... ({exc})")
                    time.sleep(delay)
        raise GeminiError(f"Gemini call failed after {config.GEMINI_MAX_RETRIES} attempts: {last_error}")

    def generate_json(self, prompt: str, retries: int = 1) -> dict:
        """Ask Gemini for JSON and parse it, handling the usual markdown-fence mess.

        If parsing fails, we retry once with a stricter follow-up prompt
        before giving up gracefully.
        """
        raw = self.generate(prompt)
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed

        if retries > 0:
            stricter_prompt = (
                prompt
                + "\n\nIMPORTANT: Your last response could not be parsed as JSON. "
                "Reply with ONLY valid JSON. No markdown fences, no commentary, no preamble."
            )
            raw2 = self.generate(stricter_prompt)
            parsed2 = _extract_json(raw2)
            if parsed2 is not None:
                return parsed2

        raise GeminiError(
            "Could not parse JSON from Gemini's response after retrying. "
            f"Raw response started with: {raw[:200]!r}"
        )


def _extract_json(text: str):
    """Try hard to pull a JSON object out of an LLM response.

    Handles: plain JSON, ```json fenced blocks, ``` fenced blocks with no
    language tag, and JSON with trailing chatter before/after it.
    """
    text = text.strip()

    # Case 1: fenced code block
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = []
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # Case 2: whole string as-is
    candidates.append(text)

    # Case 3: first {...} or [...] block found anywhere in the text
    brace_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None
