"""
llm.py
======
A thin, reusable wrapper around Google's `google-genai` SDK.

This module deliberately knows NOTHING about "agents", "tools", or
"coding tasks" — it is a small, dependency-light interface for calling
Gemini and getting text back. Keeping it separate makes it trivial to
swap the model, tune generation config, or mock it out for local unit
tests that shouldn't burn free-tier quota.

FREE-TIER QUOTA PROTECTION
---------------------------
Three things are layered on top of the raw API call, in this order:

    1. Disk cache   -> identical prompts never hit the network twice.
    2. Pacing       -> a minimum gap is enforced between live calls so
                        we don't blow through a requests/minute limit.
    3. Session cap  -> once GEMINI_SESSION_CALL_BUDGET live calls have
                        been made (across the whole notebook run), any
                        further call raises GeminiQuotaError so the
                        agent can stop *gracefully* instead of crashing
                        on a raw 429 partway through a demo.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import (
    GEMINI_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
    MIN_SECONDS_BETWEEN_CALLS,
    GEMINI_SESSION_CALL_BUDGET,
    GEMINI_CACHE_ENABLED,
    GEMINI_CACHE_DIR,
)


class GeminiError(RuntimeError):
    """Raised when a Gemini call fails after retries."""


class GeminiQuotaError(GeminiError):
    """Raised when the local session call budget has been reached.

    This is a *local* safety cap (see GEMINI_SESSION_CALL_BUDGET in
    config.py), not necessarily the provider's own error — the goal is
    to stop before we hit a real 429, so callers can catch this
    specifically and end the run cleanly.
    """


@dataclass
class GeminiConfig:
    model: str = GEMINI_MODEL
    temperature: float = 0.2
    max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS


class GeminiLLM:
    """
    Reusable Gemini client.

    Usage:
        llm = GeminiLLM(api_key=GEMINI_API_KEY)
        text = llm.generate("Write a haiku about recursion.")
        data = llm.generate_json("Return {'a': 1} as JSON.")

    The API key is passed in explicitly (never hard-coded) — in the
    Colab notebook it is loaded from `google.colab.userdata`.

    Pacing and the session call budget are tracked at the *class*
    level, so they apply across every GeminiLLM instance created in a
    notebook run (e.g. one per CodingAgent, one per demo task) — not
    just within a single instance.
    """

    _last_call_ts: float = 0.0
    _session_calls_used: int = 0

    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_MODEL,
        max_retries: int = 3,
        retry_delay_seconds: float = 3.0,
        min_seconds_between_calls: float = MIN_SECONDS_BETWEEN_CALLS,
        session_call_budget: int = GEMINI_SESSION_CALL_BUDGET,
        cache_enabled: bool = GEMINI_CACHE_ENABLED,
        cache_dir: str = GEMINI_CACHE_DIR,
    ) -> None:
        if not api_key:
            raise GeminiError(
                "No Gemini API key was provided. Add GEMINI_API_KEY to "
                "Colab Secrets (key icon in the left sidebar) and grant "
                "this notebook access to it."
            )
        try:
            from google import genai  # imported lazily so llm.py can be
        except ImportError as exc:  # unit-tested without the package too
            raise GeminiError(
                "The 'google-genai' package is not installed. "
                "Run: pip install -q google-genai"
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.min_seconds_between_calls = min_seconds_between_calls
        self.session_call_budget = session_call_budget

        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session accounting (class-level, shared across instances)
    # ------------------------------------------------------------------
    @classmethod
    def session_calls_used(cls) -> int:
        """Live (non-cached) Gemini calls made so far in this notebook run."""
        return cls._session_calls_used

    @classmethod
    def reset_session_budget(cls) -> None:
        """Reset the shared call counter — mainly useful for local testing."""
        cls._session_calls_used = 0
        cls._last_call_ts = 0.0

    # ------------------------------------------------------------------
    # Disk cache — avoids re-spending quota on a prompt already run
    # ------------------------------------------------------------------
    def _cache_key(
        self, prompt: str, temperature: float, max_output_tokens: int,
        system_instruction: Optional[str],
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "system_instruction": system_instruction or "",
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[str]:
        if not self.cache_enabled:
            return None
        path = self.cache_dir / f"{key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.cache_enabled:
            return
        try:
            (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        except OSError:
            pass  # cache is best-effort only — never fail the call over it

    # ------------------------------------------------------------------
    # Pacing + budget guards for live calls
    # ------------------------------------------------------------------
    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - GeminiLLM._last_call_ts
        wait = self.min_seconds_between_calls - elapsed
        if wait > 0:
            time.sleep(wait)
        GeminiLLM._last_call_ts = time.time()

    def _check_budget(self) -> None:
        if GeminiLLM._session_calls_used >= self.session_call_budget:
            raise GeminiQuotaError(
                f"Reached the local session budget of {self.session_call_budget} "
                "live Gemini calls for this notebook run. This is a safety cap "
                "(GEMINI_SESSION_CALL_BUDGET in config.py) meant to stop the "
                "agent gracefully before your free-tier key returns a raw 429. "
                "Increase the budget via the GEMINI_SESSION_CALL_BUDGET env "
                "var, wait for your quota window to reset, or re-run — cached "
                "prompts from earlier in this run cost nothing."
            )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in ("429", "resource_exhausted", "rate limit", "quota")
        )

    def _backoff_delay(self, exc: Exception, attempt: int) -> float:
        base = self.retry_delay_seconds * (2 ** attempt)
        if self._is_rate_limit_error(exc):
            # Rate-limit errors need noticeably longer waits than a plain
            # transient network hiccup.
            base = max(base, 20.0 * (attempt + 1))
        return base

    # ------------------------------------------------------------------
    # Core text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Send a single prompt to Gemini and return the text response.

        Cache hits skip the network entirely (no pacing delay, no budget
        cost). Live calls are paced and budget-checked, then retried a
        small number of times on transient errors — with longer backoff
        specifically for rate-limit errors — before raising GeminiError.
        """
        cache_key = self._cache_key(prompt, temperature, max_output_tokens, system_instruction)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        self._check_budget()

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()

                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                if system_instruction:
                    config["system_instruction"] = system_instruction

                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                GeminiLLM._session_calls_used += 1

                text = getattr(response, "text", None)
                if not text:
                    raise GeminiError("Gemini returned an empty response.")

                self._write_cache(cache_key, text)
                return text
            except Exception as exc:  # noqa: BLE001 - surface as GeminiError
                last_error = exc
                is_last_attempt = attempt == self.max_retries
                if is_last_attempt:
                    break
                time.sleep(self._backoff_delay(exc, attempt))

        raise GeminiError(
            f"Gemini call failed after {self.max_retries + 1} attempt(s) "
            f"using model '{self.model}': {last_error}"
        )

    # ------------------------------------------------------------------
    # JSON-structured generation
    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
    ) -> dict:
        """
        Ask Gemini for a JSON object and parse it robustly.

        LLMs frequently wrap JSON in Markdown fences (```json ... ```)
        or add stray prose — this method strips that before parsing,
        and raises a clear GeminiError if parsing still fails.
        """
        strict_prompt = (
            f"{prompt}\n\n"
            "Respond with ONLY a valid JSON object. "
            "Do not include Markdown code fences, explanations, or any "
            "text before or after the JSON."
        )
        raw = self.generate(
            strict_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        cleaned = strip_markdown_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiError(
                f"Gemini did not return valid JSON.\nRaw response:\n{raw}"
            ) from exc


def strip_markdown_fence(text: str) -> str:
    """
    Remove surrounding Markdown code fences (```python, ```json, ``` etc.)
    from an LLM response and return the inner content, trimmed.

    Handles:
      - ```python\\n<code>\\n```
      - ```json\\n{...}\\n```
      - ```\\n<code>\\n```
      - plain text with no fences at all (returned unchanged, trimmed)
    """
    if text is None:
        return ""

    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()

    # Drop the opening fence line (``` or ```python / ```json / etc.)
    if lines and lines[0].startswith("```"):
        lines = lines[1:]

    # Drop a trailing fence line, if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()
