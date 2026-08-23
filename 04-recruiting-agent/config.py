"""
config.py
---------
All the knobs for the recruiting agent live here so nothing is buried
inside logic files. Change these instead of hunting through the code.
"""

import os

# ---------------------------------------------------------------------
# Gemini model
# ---------------------------------------------------------------------
# Keep this overridable via env var so swapping models (e.g. if your
# account only has access to gemini-2.0-flash, or a newer flash model
# shows up) doesn't require touching any other file.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Free tier accounts are rate limited pretty aggressively (this project
# was built against a key limited to ~20 requests/minute). The agent is
# deliberately designed to use only 3-4 calls per run, but we still add
# a small delay + retry wrapper around every call so a transient 429
# doesn't blow up the whole run.
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_DELAY_SECONDS = 4
GEMINI_REQUEST_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------
MAX_ITERATIONS = 6  # hard ceiling, agent must never loop past this

# ---------------------------------------------------------------------
# Free tier protection
# ---------------------------------------------------------------------
MAX_CANDIDATES = 10          # don't process more than this in one run
MAX_RESUME_CHARS = 12000     # truncate long resumes before sending to Gemini

# ---------------------------------------------------------------------
# Scoring weights (must sum to 1.0 - validated in scoring.py)
# ---------------------------------------------------------------------
SCORING_WEIGHTS = {
    "mandatory_skills": 0.50,
    "preferred_skills": 0.15,
    "experience": 0.20,
    "projects": 0.10,
    "evidence": 0.05,
}

# ---------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------
SHORTLIST_SIZE = 3
MIN_MANDATORY_COVERAGE = 0.70  # candidate needs >=70% mandatory skills matched

# ---------------------------------------------------------------------
# Fairness / privacy
# ---------------------------------------------------------------------
# These fields must never be used for scoring, even if they somehow show
# up in an extracted candidate profile. tools.py strips them before the
# profile is passed anywhere near the scoring engine.
PROHIBITED_FIELDS = {
    "age",
    "gender",
    "sex",
    "religion",
    "race",
    "ethnicity",
    "health",
    "disability",
    "marital_status",
    "political_affiliation",
    "sexual_orientation",
    "nationality",
    "photo",
    "photograph",
}

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = "data"
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
OUTPUT_DIR = "outputs"
JOB_DESCRIPTION_PATH = os.path.join(DATA_DIR, "job_description.txt")

DISCLAIMER = (
    "This system provides decision support only.\n"
    "Final hiring decisions must be made by qualified human reviewers."
)
