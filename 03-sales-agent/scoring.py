"""
scoring.py — Deterministic lead scoring.

This is intentionally plain Python, not an LLM call: business scoring
should be transparent, reproducible, and auditable. Gemini is used to
*propose* per-criterion scores with evidence (see sales_agent.py's
qualification step), but the arithmetic and priority classification
that turns those scores into a decision live here, where they can be
unit tested and never hallucinated.
"""

from config import MAX_CRITERION_SCORE, MAX_LEAD_SCORE, PRIORITY_THRESHOLDS, SCORING_CRITERIA


class InvalidScoreError(ValueError):
    """Raised when a criterion score is outside the allowed range."""


def validate_criterion_score(value) -> int:
    """
    A criterion score must be an int in [0, MAX_CRITERION_SCORE], or the
    string "unknown" (treated as 0 for scoring, but preserved as evidence
    that the agent found no information rather than guessing).
    """
    if value == "unknown":
        return 0
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidScoreError(f"Score must be an int or 'unknown', got: {value!r}")
    if not (0 <= value <= MAX_CRITERION_SCORE):
        raise InvalidScoreError(
            f"Score {value} out of range 0-{MAX_CRITERION_SCORE}"
        )
    return value


def calculate_lead_score(criteria: dict) -> int:
    """
    criteria: dict mapping each of SCORING_CRITERIA -> {"score": int|"unknown", "evidence": str}
    Returns the total lead score (int, 0-MAX_LEAD_SCORE).
    Missing criteria are treated as score 0 ("unknown").
    """
    total = 0
    for name in SCORING_CRITERIA:
        entry = criteria.get(name, {})
        raw_score = entry.get("score", "unknown") if isinstance(entry, dict) else "unknown"
        total += validate_criterion_score(raw_score)
    return total


def classify_priority(score: int) -> str:
    """Map a total score to a priority bucket using PRIORITY_THRESHOLDS."""
    if not isinstance(score, int) or isinstance(score, bool):
        raise InvalidScoreError(f"Total score must be an int, got: {score!r}")
    if not (0 <= score <= MAX_LEAD_SCORE):
        raise InvalidScoreError(f"Total score {score} out of range 0-{MAX_LEAD_SCORE}")

    if score >= PRIORITY_THRESHOLDS["HIGH PRIORITY"]:
        return "HIGH PRIORITY"
    if score >= PRIORITY_THRESHOLDS["MEDIUM PRIORITY"]:
        return "MEDIUM PRIORITY"
    if score >= PRIORITY_THRESHOLDS["LOW PRIORITY"]:
        return "LOW PRIORITY"
    return "INSUFFICIENT DATA"


def score_lead(company: str, criteria: dict, sources: list = None) -> dict:
    """
    Build the full scored-lead record for one company.
    Raises InvalidScoreError if any criterion score is malformed — the
    caller should catch this and mark the lead INSUFFICIENT DATA rather
    than crash the whole run.
    """
    total = calculate_lead_score(criteria)
    return {
        "company": company,
        "score": total,
        "max_score": MAX_LEAD_SCORE,
        "priority": classify_priority(total),
        "criteria": criteria,
        "sources": sources or [],
    }
