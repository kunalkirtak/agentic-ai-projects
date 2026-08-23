"""
config.py
---------
Central configuration for the Agentic Research Agent.

Everything that controls cost, behaviour, or model choice lives here so the
rest of the codebase never hard-codes a magic number or a model name.

Notes on the Gemini model:
    Gemini model availability changes over time and can differ by account
    and region. GEMINI_MODEL is deliberately NOT hard-coded into the logic
    anywhere else in this project -- it is read once, here, from an
    environment variable (with a sensible fallback). If your account does
    not have access to the default model below, either:
        1. Set the environment variable GEMINI_MODEL to a model you do
           have access to, or
        2. Edit DEFAULT_GEMINI_MODEL in this file.
    A good way to check which models your key can use is to run:
        for m in client.models.list():
            print(m.name)
"""

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# This default is current as of project creation but Google regularly ships
# new Gemini model names and retires old ones. Change the env var, not this
# file, if you need a different model long-term.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------------------------
# Free-tier / cost safety limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_AGENT_STEPS = 5      # hard ceiling on the agent loop
DEFAULT_MAX_SEARCH_RESULTS = 5   # results returned per web search call
DEFAULT_MAX_SOURCES = 8          # total sources kept across the whole run
DEFAULT_MAX_PAGE_CHARS = 12000   # characters kept per fetched webpage
DEFAULT_REQUEST_TIMEOUT = 10     # seconds, for webpage fetches


@dataclass
class Config:
    """
    Runtime configuration for the Research Agent.

    GEMINI_API_KEY is intentionally required with no default -- the agent
    should fail fast and clearly if it isn't configured, rather than
    silently doing nothing.
    """

    gemini_api_key: str
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    )
    max_agent_steps: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_AGENT_STEPS", DEFAULT_MAX_AGENT_STEPS)
        )
    )
    max_search_results: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_SEARCH_RESULTS", DEFAULT_MAX_SEARCH_RESULTS)
        )
    )
    max_sources: int = field(
        default_factory=lambda: int(os.environ.get("MAX_SOURCES", DEFAULT_MAX_SOURCES))
    )
    max_page_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_PAGE_CHARS", DEFAULT_MAX_PAGE_CHARS)
        )
    )
    request_timeout: int = field(
        default_factory=lambda: int(
            os.environ.get("REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
        )
    )

    def __post_init__(self):
        if not self.gemini_api_key or not str(self.gemini_api_key).strip():
            raise ValueError(
                "GEMINI_API_KEY is missing or empty. In Google Colab, add it "
                "under the Secrets (key) panel with the name 'GEMINI_API_KEY' "
                "and enable notebook access for it. Never paste the key "
                "directly into a cell."
            )

    def summary(self) -> str:
        """Human readable, secret-free summary for logging."""
        return (
            f"model={self.gemini_model} | "
            f"max_agent_steps={self.max_agent_steps} | "
            f"max_search_results={self.max_search_results} | "
            f"max_sources={self.max_sources} | "
            f"max_page_chars={self.max_page_chars}"
        )


def load_config_from_env(gemini_api_key: str) -> Config:
    """
    Build a Config using an explicitly-passed API key (e.g. loaded from
    Colab Secrets by the caller) plus whatever overrides are present in
    the environment.
    """
    return Config(gemini_api_key=gemini_api_key)
