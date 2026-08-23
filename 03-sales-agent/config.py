"""
config.py — Central configuration for the Agentic Sales Agent.

All limits here exist to keep the agent inside the Gemini free tier
and to avoid hammering public websites. Change them with care.
"""

import os

# ---------------------------------------------------------------------------
# Gemini model
# ---------------------------------------------------------------------------
# Configurable via environment variable so the same code works if the user's
# account has access to a different Gemini model name. If you see a
# "model not found" error, change this value (or set GEMINI_MODEL in your
# environment / Colab Secrets) to a model available to your API key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Free-tier / politeness limits
# ---------------------------------------------------------------------------
# Kept deliberately small so a full run (ICP analysis + company discovery +
# qualification) uses only a handful of Gemini calls, and so the notebook
# can run several demos back to back without tripping a free-tier API key's
# request-per-minute, request-per-day, or token-per-minute caps.
MAX_ITERATIONS = 2           # bounded agent loop — never allow infinite execution
MAX_SEARCH_QUERIES = 3        # number of distinct web searches per run
MAX_SEARCH_RESULTS = 5        # results kept per search query
MAX_COMPANIES = 6             # candidate companies carried into research
MAX_COMPANY_PAGES = 2         # public pages fetched per company
MAX_PAGE_CHARS = 6_000        # characters kept per fetched page (truncated)
MAX_COMPANY_EXCERPT_CHARS = 2_500  # per-company text sent to Gemini for qualification

REQUEST_TIMEOUT_SECONDS = 8
USER_AGENT = "Mozilla/5.0 (compatible; SalesResearchAgent/1.0; educational-portfolio-project)"

# ---------------------------------------------------------------------------
# Gemini free-tier request budget & pacing
# ---------------------------------------------------------------------------
# Free-tier Gemini API keys are commonly capped at a low number of requests
# per minute AND a low number of requests per day (some experimental/limited
# models allow as few as ~20 requests/day). These constants exist so the
# *whole notebook* — including multiple demo runs — always finishes inside a
# conservative shared budget, and so calls are paced far enough apart to
# avoid a per-minute rate limit even when the daily cap isn't the issue.
#
# Once the session budget is reached, the agent automatically switches to
# fast, deterministic fallback logic (see sales_agent.py) instead of calling
# Gemini again, so a run always finishes with a real report instead of
# crashing or hanging on a 429.
MAX_GEMINI_CALLS_PER_SESSION = 18   # hard budget; leaves headroom under a 20-request free-tier cap
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 6.5  # paces calls to stay comfortably under ~9 requests/minute
GEMINI_MAX_RETRIES = 3
GEMINI_BACKOFF_SECONDS = 15  # base wait on a rate-limit error; grows with each retry

# ---------------------------------------------------------------------------
# Lead scoring
# ---------------------------------------------------------------------------
SCORING_CRITERIA = [
    "industry_fit",
    "geography_fit",
    "company_size_fit",
    "problem_fit",
    "technology_fit",
    "evidence_quality",
]
MAX_CRITERION_SCORE = 3
MAX_LEAD_SCORE = MAX_CRITERION_SCORE * len(SCORING_CRITERIA)  # 18

PRIORITY_THRESHOLDS = {
    "HIGH PRIORITY": 15,
    "MEDIUM PRIORITY": 11,
    "LOW PRIORITY": 7,
    # anything below LOW PRIORITY's threshold -> INSUFFICIENT DATA
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
LEADS_JSON_PATH = os.path.join(OUTPUT_DIR, "leads.json")
REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "sales_report.md")
