"""
recruiting_agent.py
--------------------
The agent itself: state, a dynamically-generated plan, a bounded loop
over a fixed set of actions, and a self-check before it's willing to
call the run finished.

This is deliberately NOT "resume -> Gemini -> score". Gemini is used for
three narrow, batched extraction/writing jobs; everything about how a
score is computed, who gets ranked where, and who makes the shortlist
is plain deterministic Python in scoring.py.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
import parser
import scoring
import tools
from llm import GeminiLLM, GeminiError


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

@dataclass
class RecruitingState:
    job_description: str
    job_requirements: dict = field(default_factory=dict)
    plan: list = field(default_factory=list)
    resumes: list = field(default_factory=list)              # raw loaded text
    load_errors: list = field(default_factory=list)          # files that failed to load
    candidate_profiles: list = field(default_factory=list)   # structured, privacy-filtered
    match_results: dict = field(default_factory=dict)        # candidate_id -> matches
    evaluations: list = field(default_factory=list)          # scored + ranked
    shortlisted: list = field(default_factory=list)
    quality_issues: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    gemini_calls: int = 0
    iterations: int = 0
    final_report_md: str = ""
    final_report_json: dict = field(default_factory=dict)


ACTIONS = [
    "ANALYZE_JOB",
    "CREATE_PLAN",
    "PARSE_RESUMES",
    "EXTRACT_PROFILES",
    "MATCH_REQUIREMENTS",
    "SCORE_AND_RANK",
    "VALIDATE_EVIDENCE",
    "GENERATE_REPORT",
    "FINISH",
]


class RecruitingAgent:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    # -------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------
    def run(self, job_description: str, resume_dir: str) -> RecruitingState:
        state = RecruitingState(job_description=job_description)
        action_index = 0

        while state.iterations < config.MAX_ITERATIONS and action_index < len(ACTIONS):
            action = ACTIONS[action_index]
            self._log(state, f"iteration {state.iterations + 1} -> action: {action}")

            if action == "ANALYZE_JOB":
                self._analyze_job(state)
            elif action == "CREATE_PLAN":
                self._create_plan(state)
            elif action == "PARSE_RESUMES":
                self._parse_resumes(state, resume_dir)
            elif action == "EXTRACT_PROFILES":
                self._extract_profiles(state)
            elif action == "MATCH_REQUIREMENTS":
                self._match_requirements(state)
            elif action == "SCORE_AND_RANK":
                self._score_and_rank(state)
            elif action == "VALIDATE_EVIDENCE":
                self._validate_evidence(state)
            elif action == "GENERATE_REPORT":
                self._generate_report(state)
            elif action == "FINISH":
                self._log(state, "run complete")
                break

            action_index += 1
            state.iterations += 1

        state.gemini_calls = self.llm.call_count
        return state

    # -------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------
    def _analyze_job(self, state: RecruitingState):
        print("\n🧠 Step 1 — Analyzing job requirements")
        state.job_requirements = tools.analyze_job_description(self.llm, state.job_description)
        n_mandatory = len(state.job_requirements.get("mandatory_skills", []))
        n_preferred = len(state.job_requirements.get("preferred_skills", []))
        print(f"✓ {n_mandatory} mandatory requirements")
        print(f"✓ {n_preferred} preferred requirements")
        self._log(state, f"extracted {n_mandatory} mandatory / {n_preferred} preferred requirements")

    def _create_plan(self, state: RecruitingState):
        print("\n📋 Step 2 — Creating evaluation plan")
        role = state.job_requirements.get("role", "the role")
        state.plan = [
            f"Analyze requirements for {role}.",
            "Extract mandatory and preferred criteria.",
            "Parse all candidate resumes.",
            "Build structured candidate profiles (skills, experience, projects only).",
            "Match each profile against every requirement with evidence.",
            "Calculate deterministic, weighted scores in Python.",
            "Run a quality/evidence check before ranking.",
            "Rank candidates and build a shortlist for human review.",
            "Produce an explainable report.",
        ]
        print("✓ Evaluation plan created")
        self._log(state, "plan created with 9 steps")

    def _parse_resumes(self, state: RecruitingState, resume_dir: str):
        print("\n📄 Step 3 — Parsing resumes")
        loaded, errors = parser.load_resumes_from_dir(resume_dir)
        state.resumes = loaded
        state.load_errors = errors
        print(f"✓ {len(loaded)} resumes loaded")
        if errors:
            for e in errors:
                print(f"  ⚠ Skipped {e['filename']}: {e['error']}")
        self._log(state, f"loaded {len(loaded)} resumes, {len(errors)} errors")

    def _extract_profiles(self, state: RecruitingState):
        print("\n🧩 Step 4 — Extracting candidate profiles")
        if not state.resumes:
            print("  ⚠ No resumes to extract profiles from.")
            return
        state.candidate_profiles = tools.extract_candidate_profiles(self.llm, state.resumes)
        print(f"✓ {len(state.candidate_profiles)} candidate profiles extracted")
        self._log(state, f"extracted {len(state.candidate_profiles)} profiles")

    def _match_requirements(self, state: RecruitingState):
        print("\n🔍 Step 5 — Matching candidates against requirements")
        if not state.candidate_profiles:
            print("  ⚠ No profiles to match.")
            return
        state.match_results = tools.evaluate_candidates(
            self.llm, state.job_requirements, state.candidate_profiles
        )
        print(f"✓ Candidate evaluations created for {len(state.match_results)} candidates")
        self._log(state, f"matched {len(state.match_results)} candidates against requirements")

    def _score_and_rank(self, state: RecruitingState):
        print("\n📊 Step 6 — Calculating deterministic scores")
        weights = config.SCORING_WEIGHTS
        job_keywords = (
            state.job_requirements.get("mandatory_skills", [])
            + state.job_requirements.get("preferred_skills", [])
        )
        min_experience = state.job_requirements.get("minimum_experience_years")

        evaluations = []
        profiles_by_id = {p["candidate_id"]: p for p in state.candidate_profiles}

        for candidate_id, match in state.match_results.items():
            profile = profiles_by_id.get(candidate_id, {})
            mandatory_matches = match.get("mandatory_matches", [])
            preferred_matches = match.get("preferred_matches", [])
            all_matches = mandatory_matches + preferred_matches

            mandatory_score = scoring.calculate_skill_score(mandatory_matches, weights["mandatory_skills"])
            preferred_score = scoring.calculate_skill_score(preferred_matches, weights["preferred_skills"])
            experience_score = scoring.calculate_experience_score(
                profile.get("experience_years"), min_experience, weights["experience"]
            )
            project_score = scoring.calculate_project_score(
                profile.get("projects", []), job_keywords, weights["projects"]
            )
            evidence_score = scoring.calculate_evidence_score(all_matches, weights["evidence"])

            breakdown = scoring.calculate_total_score({
                "mandatory_skills": mandatory_score,
                "preferred_skills": preferred_score,
                "experience": experience_score,
                "projects": project_score,
                "evidence": evidence_score,
            })

            evaluations.append({
                "candidate_id": candidate_id,
                "profile": profile,
                "mandatory_matches": mandatory_matches,
                "preferred_matches": preferred_matches,
                "strengths": match.get("strengths", []),
                "gaps": match.get("gaps", []),
                "mandatory_coverage": round(scoring.mandatory_coverage_ratio(mandatory_matches), 2),
                "score_breakdown": breakdown,
            })

        evaluations = scoring.rank_candidates(evaluations)
        state.evaluations = evaluations
        state.shortlisted = scoring.create_shortlist(evaluations)

        print("✓ Scores calculated")
        self._log(state, f"scored and ranked {len(evaluations)} candidates")

    def _validate_evidence(self, state: RecruitingState):
        print("\n🔎 Step 7 — Validating evidence")
        issues = []

        mandatory_reqs = set(state.job_requirements.get("mandatory_skills", []))

        for ev in state.evaluations:
            cid = ev["candidate_id"]

            evaluated = {m["requirement"] for m in ev["mandatory_matches"]}
            missing = mandatory_reqs - evaluated
            if missing:
                issues.append(f"{cid}: mandatory requirements not evaluated: {sorted(missing)}")

            for m in ev["mandatory_matches"] + ev["preferred_matches"]:
                if m["status"] in ("MATCH", "PARTIAL_MATCH") and not m.get("evidence", "").strip():
                    issues.append(f"{cid}: '{m['requirement']}' marked {m['status']} with no evidence text")

            total = ev["score_breakdown"]["total"]
            if not (0 <= total <= 100):
                issues.append(f"{cid}: score {total} out of bounds")

            violations = tools.validate_profile_privacy(ev["profile"])
            if violations:
                issues.append(f"{cid}: prohibited fields present in profile: {violations}")

        state.quality_issues = issues
        if issues:
            print(f"  ⚠ {len(issues)} issue(s) found:")
            for i in issues:
                print(f"    - {i}")
        else:
            print("✓ Evidence validation completed — no issues found")
        self._log(state, f"quality check found {len(issues)} issue(s)")

    def _generate_report(self, state: RecruitingState):
        print("\n🏆 Step 8 — Ranking candidates and generating report")

        ranked_summaries = [
            {
                "candidate_id": ev["candidate_id"],
                "rank": ev["rank"],
                "score": ev["score_breakdown"]["total"],
                "mandatory_coverage": ev["mandatory_coverage"],
                "strengths": ev["strengths"],
                "gaps": ev["gaps"],
            }
            for ev in state.evaluations
        ]

        summary_text = ""
        try:
            summary_text = tools.synthesize_report_summary(
                self.llm, state.job_requirements, ranked_summaries
            )
        except GeminiError as exc:
            summary_text = (
                "Executive summary could not be generated automatically "
                f"({exc}). See the candidate ranking and detailed evaluations below."
            )

        state.final_report_md = _build_markdown_report(state, summary_text)
        state.final_report_json = _build_json_report(state)

        print("✓ Report generated")
        self._log(state, "final report generated")

    # -------------------------------------------------------------
    def _log(self, state: RecruitingState, message: str):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        state.trace.append(f"[{stamp}] {message}")


# ---------------------------------------------------------------------
# Report builders (pure Python, no Gemini)
# ---------------------------------------------------------------------

def _status_symbol(status: str) -> str:
    return {"MATCH": "✓", "PARTIAL_MATCH": "~", "NO_EVIDENCE": "-"}.get(status, "?")


def _build_markdown_report(state: RecruitingState, summary_text: str) -> str:
    lines = []
    role = state.job_requirements.get("role", "unknown")
    lines.append(f"# Recruiting Report — {role}")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(summary_text or "_No summary available._")
    lines.append("")

    lines.append("## Job Requirements")
    lines.append(f"**Role:** {role}")
    lines.append("")
    lines.append("**Mandatory:** " + ", ".join(state.job_requirements.get("mandatory_skills", [])) or "_none extracted_")
    lines.append("")
    lines.append("**Preferred:** " + ", ".join(state.job_requirements.get("preferred_skills", [])) or "_none extracted_")
    lines.append("")
    min_exp = state.job_requirements.get("minimum_experience_years")
    lines.append(f"**Minimum experience:** {min_exp if min_exp is not None else 'not specified'} years")
    lines.append("")

    lines.append("## Candidate Ranking")
    lines.append("")
    for ev in state.evaluations:
        lines.append(f"{ev['rank']}. Candidate `{ev['candidate_id']}` — {ev['score_breakdown']['total']}/100")
    lines.append("")

    lines.append("## Recommended Shortlist")
    lines.append("")
    if state.shortlisted:
        for ev in state.shortlisted:
            lines.append(f"- `{ev['candidate_id']}` — {ev['score_breakdown']['total']}/100 — recommended for human review")
    else:
        lines.append("_No candidates cleared the minimum mandatory-coverage threshold._")
    lines.append("")

    lines.append("## Detailed Candidate Evaluation")
    for ev in state.evaluations:
        lines.append("")
        lines.append(f"### Candidate `{ev['candidate_id']}`")
        lines.append(f"**Overall Score:** {ev['score_breakdown']['total']}/100")
        lines.append("**Status:** Recommended for Human Review")
        lines.append(f"**Mandatory Skill Coverage:** {round(ev['mandatory_coverage'] * 100)}%")
        lines.append("")
        lines.append("**Score Breakdown:**")
        for key, val in ev["score_breakdown"].items():
            if key == "total":
                continue
            lines.append(f"- {key.replace('_', ' ').title()}: {val['score']}/{val['max']}")
        lines.append("")
        lines.append("**Mandatory Requirements:**")
        for m in ev["mandatory_matches"]:
            symbol = _status_symbol(m["status"])
            evidence = f" — {m['evidence']}" if m.get("evidence") else ""
            lines.append(f"- {symbol} {m['requirement']} ({m['status']}){evidence}")
        lines.append("")
        lines.append("**Preferred Requirements:**")
        for m in ev["preferred_matches"]:
            symbol = _status_symbol(m["status"])
            evidence = f" — {m['evidence']}" if m.get("evidence") else ""
            lines.append(f"- {symbol} {m['requirement']} ({m['status']}){evidence}")
        lines.append("")
        lines.append("**Strengths:**")
        for s in ev["strengths"]:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Potential Gaps:**")
        for g in ev["gaps"]:
            lines.append(f"- {g}")
        lines.append("")
        lines.append("**Recommendation:** Strong match based on provided evidence. Recommended for human review.")

    lines.append("")
    lines.append("## Human Review Considerations")
    lines.append(
        "- Missing evidence for a requirement does not mean the candidate lacks that skill — "
        "it means the resume did not mention it.\n"
        "- Scores reflect resume content only, not interview performance or references.\n"
        "- Recruiters should verify high-impact claims directly with candidates."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append(
        "- Resume quality directly affects extraction quality.\n"
        "- LLM-based extraction can contain errors and should be spot-checked.\n"
        "- Scores depend on the configurable weighting in `config.py`.\n"
        "- This prototype is not a production hiring system."
    )
    lines.append("")
    lines.append("---")
    lines.append(config.DISCLAIMER)

    return "\n".join(lines)


def _build_json_report(state: RecruitingState) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job": state.job_requirements,
        "gemini_calls": state.gemini_calls,
        "candidates_processed": len(state.candidate_profiles),
        "load_errors": state.load_errors,
        "quality_issues": state.quality_issues,
        "candidates": [
            {
                "candidate_id": ev["candidate_id"],
                "rank": ev["rank"],
                "score": ev["score_breakdown"]["total"],
                "score_breakdown": ev["score_breakdown"],
                "mandatory_coverage": ev["mandatory_coverage"],
                "mandatory_matches": ev["mandatory_matches"],
                "preferred_matches": ev["preferred_matches"],
                "strengths": ev["strengths"],
                "gaps": ev["gaps"],
                "priority": "SHORTLISTED" if ev in state.shortlisted else "REVIEW",
            }
            for ev in state.evaluations
        ],
        "shortlist": [ev["candidate_id"] for ev in state.shortlisted],
        "disclaimer": config.DISCLAIMER,
    }


# ---------------------------------------------------------------------
# Convenience runner used by both the CLI and the notebook
# ---------------------------------------------------------------------

def run_demo(api_key: str, job_description_path: str = None, resume_dir: str = None):
    job_description_path = job_description_path or config.JOB_DESCRIPTION_PATH
    resume_dir = resume_dir or config.RESUME_DIR

    with open(job_description_path, "r", encoding="utf-8") as f:
        job_description = f.read()

    llm = GeminiLLM(api_key=api_key)
    agent = RecruitingAgent(llm)

    print("=" * 56)
    print("🤖 AGENTIC RECRUITING AGENT")
    print("=" * 56)
    print(f"\n💼 Job description loaded from: {job_description_path}")

    state = agent.run(job_description, resume_dir)

    print("\n" + "=" * 56)
    print("SHORTLIST")
    print("=" * 56)
    for ev in state.shortlisted:
        print(f"{ev['rank']}. {ev['candidate_id']} — {ev['score_breakdown']['total']}/100")
    print(f"\nGemini calls: {state.gemini_calls}")
    print(f"Candidates processed: {len(state.candidate_profiles)}")
    print(f"Iterations: {state.iterations}")
    print(f"\n⚠ {config.DISCLAIMER}")

    return state


def export_outputs(state: RecruitingState, output_dir: str = None):
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "candidate_rankings.json")
    md_path = os.path.join(output_dir, "recruiting_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state.final_report_json, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(state.final_report_md)

    return json_path, md_path
