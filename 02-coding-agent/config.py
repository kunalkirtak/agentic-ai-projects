"""
config.py
=========
Central configuration for the Agentic Coding Agent.

All tunables live here so the rest of the codebase never hard-codes
"magic numbers" or model names. This keeps the project easy to adapt
to whatever Gemini model the user's free-tier account has access to.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Gemini model configuration
# ---------------------------------------------------------------------------
# The model name is intentionally kept configurable via an environment
# variable. Free-tier Gemini access can vary by account/region, so the
# architecture must not assume any single model name is always available.
# Change this (or set the GEMINI_MODEL env var / Colab form field) to
# whatever model your account currently has free-tier access to, e.g.
# "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", etc.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Agent loop bounds — mandatory to protect the free-tier quota
# ---------------------------------------------------------------------------
# 6 iterations is the minimum needed for the agent to actually use its
# "self-fix" loop: PLAN -> GENERATE_CODE -> GENERATE_TESTS -> RUN_TESTS ->
# ANALYZE_FAILURE+FIX -> RUN_TESTS. With fewer iterations the agent can
# reach the failure state but never gets a turn to fix it — the diagram
# in the README (FAILURE ANALYZER -> CODE FIXER) would never actually run.
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "6"))

# ---------------------------------------------------------------------------
# Safe execution settings
# ---------------------------------------------------------------------------
EXECUTION_TIMEOUT: int = int(os.getenv("EXECUTION_TIMEOUT", "10"))  # seconds

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
GEMINI_TEMPERATURE_PLANNING: float = 0.2   # deterministic, structured output
GEMINI_TEMPERATURE_CODEGEN: float = 0.3    # a little room for good code
GEMINI_TEMPERATURE_FIX: float = 0.2        # focused, deterministic fixes

# Max output tokens per call — keeps responses (and quota use) bounded.
# Lowered from 2048 -> 1536: plenty for the small utility-style modules
# this agent generates, while leaving more headroom under free-tier
# tokens-per-minute limits.
GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1536"))

# ---------------------------------------------------------------------------
# Free-tier quota protection
# ---------------------------------------------------------------------------
# Free-tier Gemini keys are rate-limited (requests/minute) AND capped on
# total requests (per day, or in your case as few as ~20 total). The three
# settings below exist purely to keep this notebook well-behaved on a
# constrained free key, so it produces clean, presentable output instead
# of a raw 429 traceback halfway through a demo.

# Minimum gap (seconds) enforced between consecutive *live* Gemini calls.
# 6s -> at most 10 calls/minute, safely under typical free-tier RPM caps.
# Raise this if your key's RPM limit is lower than 10.
MIN_SECONDS_BETWEEN_CALLS: float = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "6"))

# Hard cap on how many *live* (non-cached) Gemini calls this notebook will
# make in one run, across every cell. Once reached, the agent stops
# gracefully (partial results are still shown) instead of burning through
# the rest of a small quota and crashing on a 429. Tune this to match
# whatever your key's actual limit is.
GEMINI_SESSION_CALL_BUDGET: int = int(os.getenv("GEMINI_SESSION_CALL_BUDGET", "20"))

# On-disk response cache so re-running a cell (e.g. while polishing this
# notebook for GitHub) never re-spends quota on a prompt you already ran.
# Identical (model, prompt, temperature, max_tokens) -> served from disk.
GEMINI_CACHE_ENABLED: bool = os.getenv("GEMINI_CACHE_ENABLED", "1") != "0"
GEMINI_CACHE_DIR: str = os.getenv("GEMINI_CACHE_DIR", ".gemini_cache")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "Agentic Coding Agent"
SUPPORTED_TASK_DOMAINS = (
    "CLI applications",
    "Data processing scripts",
    "Algorithms",
    "Utility functions",
    "File processing",
    "API client examples",
    "Small automation scripts",
    "Basic ML preprocessing",
)
