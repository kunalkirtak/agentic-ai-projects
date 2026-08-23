"""
scoring.py
----------
Deterministic, reproducible scoring. No LLM calls anywhere in this file
- that's the whole point. Given the same matches twice, you get the same
score twice. Gemini is good at reading resumes; it's not the thing that
should decide 78 vs 81.
"""

import config


def _validate_weights():
    total = sum(config.SCORING_WEIGHTS.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"SCORING_WEIGHTS must sum to 1.0, got {total}")


_validate_weights()


def calculate_skill_score(matches: list, weight: float) -> float:
    """matches: list of {'status': 'MATCH'|'PARTIAL_MATCH'|'NO_EVIDENCE'}.

    MATCH counts full, PARTIAL_MATCH counts half, NO_EVIDENCE counts zero.
    Returns points out of (weight * 100), capped so rounding can never
    push it over the max.
    """
    if not matches:
        return 0.0

    total_points = 0.0
    for m in matches:
        status = m.get("status", "NO_EVIDENCE")
        if status == "MATCH":
            total_points += 1.0
        elif status == "PARTIAL_MATCH":
            total_points += 0.5
        # NO_EVIDENCE contributes 0

    ratio = total_points / len(matches)
    max_points = weight * 100
    return round(min(ratio * max_points, max_points), 2)


def calculate_experience_score(candidate_years, required_years, weight: float) -> float:
    """Full credit if candidate meets/exceeds requirement, partial credit
    scaled linearly if below it, zero if years unknown.
    """
    max_points = weight * 100
    if candidate_years in (None, "unknown"):
        return 0.0
    if required_years in (None, "unknown") or required_years == 0:
        return max_points  # no explicit requirement -> don't penalize

    try:
        candidate_years = float(candidate_years)
        required_years = float(required_years)
    except (TypeError, ValueError):
        return 0.0

    if candidate_years >= required_years:
        return round(max_points, 2)

    ratio = max(candidate_years / required_years, 0.0)
    return round(min(ratio * max_points, max_points), 2)


def calculate_project_score(projects: list, relevant_keywords: list, weight: float) -> float:
    """Rough but transparent: how many listed projects mention at least
    one relevant keyword, out of total projects considered (capped at 3
    so a candidate can't just pad a giant project list).
    """
    max_points = weight * 100
    if not projects:
        return 0.0

    keywords_lower = [k.lower() for k in relevant_keywords]
    considered = projects[:3]
    relevant_count = 0
    for project in considered:
        project_text = str(project).lower()
        if any(kw in project_text for kw in keywords_lower):
            relevant_count += 1

    ratio = relevant_count / len(considered)
    return round(min(ratio * max_points, max_points), 2)


def calculate_evidence_score(matches: list, weight: float) -> float:
    """Rewards matches that come with actual evidence text attached,
    not just a bare MATCH label.
    """
    max_points = weight * 100
    matched = [m for m in matches if m.get("status") in ("MATCH", "PARTIAL_MATCH")]
    if not matched:
        return 0.0

    with_evidence = sum(1 for m in matched if m.get("evidence", "").strip())
    ratio = with_evidence / len(matched)
    return round(min(ratio * max_points, max_points), 2)


def calculate_total_score(components: dict) -> dict:
    """components: dict with keys mandatory_skills, preferred_skills,
    experience, projects, evidence - each already a point value (not a
    ratio) that respects its own weight's max.

    Returns a breakdown dict plus the total, clamped to [0, 100].
    """
    weights = config.SCORING_WEIGHTS
    breakdown = {}
    total = 0.0
    for key, weight in weights.items():
        max_points = round(weight * 100, 2)
        value = components.get(key, 0.0)
        value = max(0.0, min(value, max_points))  # never exceed max for that component
        breakdown[key] = {"score": value, "max": max_points}
        total += value

    total = round(max(0.0, min(total, 100.0)), 2)
    breakdown["total"] = total
    return breakdown


def mandatory_coverage_ratio(matches: list) -> float:
    """Fraction of mandatory requirements that are at least MATCH or
    PARTIAL_MATCH. Used for the shortlist threshold, separate from score.
    """
    if not matches:
        return 0.0
    covered = sum(1 for m in matches if m.get("status") in ("MATCH", "PARTIAL_MATCH"))
    return covered / len(matches)


def rank_candidates(evaluations: list) -> list:
    """Sort candidates by total score, descending. Ties broken by
    mandatory coverage, then candidate_id for stability.
    """
    def sort_key(ev):
        return (
            -ev["score_breakdown"]["total"],
            -ev.get("mandatory_coverage", 0.0),
            ev["candidate_id"],
        )

    ranked = sorted(evaluations, key=sort_key)
    for i, ev in enumerate(ranked, start=1):
        ev["rank"] = i
    return ranked


def create_shortlist(ranked_evaluations: list, size: int = None, min_coverage: float = None) -> list:
    """Take the top N candidates that also clear the mandatory-coverage bar.

    Score alone isn't enough to shortlist someone - they also need to
    genuinely cover the mandatory requirements, otherwise a candidate
    with great "nice to have" skills but no mandatory-skill evidence
    could sneak in on preferred/experience/project points.
    """
    size = config.SHORTLIST_SIZE if size is None else size
    min_coverage = config.MIN_MANDATORY_COVERAGE if min_coverage is None else min_coverage

    eligible = [
        ev for ev in ranked_evaluations
        if ev.get("mandatory_coverage", 0.0) >= min_coverage
    ]
    return eligible[:size]
