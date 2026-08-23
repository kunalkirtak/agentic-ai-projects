"""
llm.py
------
Thin, reusable wrapper around the Gemini API (google-genai SDK).

This is intentionally the ONLY file that talks to Gemini directly. Every
other module calls GeminiLLM.generate(...) so that:
    - the model name is configurable in one place (config.py)
    - error handling / retries live in one place
    - later portfolio projects (coding agent, sales agent, recruiting
      agent) can reuse this exact class unchanged
"""

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger("research_agent.llm")


class GeminiError(RuntimeError):
    """Raised for any Gemini call that fails after retries."""


class GeminiLLM:
    """
    Minimal wrapper around google-genai's client.

    Usage:
        llm = GeminiLLM(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
        text = llm.generate("Write a haiku about oceans")
    """

    def __init__(self, api_key: str, model: str, max_retries: int = 2, retry_delay: float = 2.0):
        if not api_key:
            raise GeminiError("Missing Gemini API key.")

        try:
            from google import genai
        except ImportError as exc:
            raise GeminiError(
                "The 'google-genai' package is not installed. "
                "Run: pip install -q -U google-genai"
            ) from exc

        self._genai = genai
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.call_count = 0  # simple usage counter for free-tier awareness

        # Client reads GEMINI_API_KEY itself, but we pass it explicitly so
        # the key never needs to live in os.environ.
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.4, max_output_tokens: int = 2048) -> str:
        """
        Send a single prompt to Gemini and return the text response.
        Raises GeminiError on unrecoverable failure (caller decides how to
        degrade gracefully).
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):  # e.g. 1 try + N retries
            try:
                self.call_count += 1
                logger.info("Gemini call #%d (attempt %d) -> model=%s", self.call_count, attempt, self.model)

                from google.genai import types

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )

                text = getattr(response, "text", None)
                if text is None or text.strip() == "":
                    raise GeminiError("Gemini returned an empty response.")
                return text

            except Exception as exc:  # broad: SDK raises several exception types
                last_error = exc
                message = str(exc).lower()

                # Don't retry on things a retry can't fix.
                non_retryable = ("api key" in message or "permission" in message or "not found" in message)
                if non_retryable or attempt == self.max_retries + 1:
                    break

                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt, self.max_retries + 1, exc, self.retry_delay,
                )
                time.sleep(self.retry_delay)

        raise GeminiError(f"Gemini call failed after retries: {last_error}")


def safe_json_parse(raw_text: str) -> Optional[dict]:
    """
    Best-effort extraction of a JSON object from an LLM response.

    LLMs frequently wrap JSON in markdown fences or add stray prose before
    or after the object. This function tries several increasingly lenient
    strategies and returns None (never raises) if all of them fail, so a
    single malformed response can never crash the agent.
    """
    if not raw_text:
        return None

    candidates = []

    # Strategy 1: the whole string, as-is.
    candidates.append(raw_text.strip())

    # Strategy 2: strip ```json ... ``` or ``` ... ``` fences.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # Strategy 3: grab the substring between the first '{' and the last '}'.
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    logger.warning("safe_json_parse: could not parse JSON from model output.")
    return None
