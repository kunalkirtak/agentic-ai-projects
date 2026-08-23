"""
tools.py
--------
The "tools" the agent calls. Each function here does one job:
- ask Gemini to extract structure from unstructured text, or
- do local bookkeeping (privacy filtering) that must never touch an LLM.

Four Gemini calls total in a normal run, all batched:
  1. analyze_job_description        -> one call
  2. extract_candidate_profiles     -> one call for ALL resumes together
  3. evaluate_candidates            -> one call for ALL candidates together
  4. synthesize_report              -> one call

That's the free-tier budget this whole project is built around.
"""

import json

import config
from llm import GeminiLLM


# ---------------------------------------------------------------------
# 1. Job requirement extraction
# ---------------------------------------------------------------------

def analyze_job_description(llm: GeminiLLM, job_description: str) -> dict:
    prompt = f"""You are analyzing a job description for a recruiting tool.
Extract structured requirements. Only use information present in the text
below - do not invent requirements that aren't there.

Return ONLY valid JSON in exactly this shape:
{{
  "role": "<job title>",
  "mandatory_skills": ["skill", ...],
  "preferred_skills": ["skill", ...],
  "minimum_experience_years": <number or null>,
  "responsibilities": ["responsibility", ...]
}}

Job description:
\"\"\"
{job_description}
\"\"\"
"""
    result = llm.generate_json(prompt)
    result.setdefault("role", "unknown")
    result.setdefault("mandatory_skills", [])
    result.setdefault("preferred_skills", [])
    result.setdefault("minimum_experience_years", None)
    result.setdefault("responsibilities", [])
    return result


# ---------------------------------------------------------------------
# 2. Candidate profile extraction (batched across all resumes)
# ---------------------------------------------------------------------

def extract_candidate_profiles(llm: GeminiLLM, resumes: list) -> list:
    """resumes: list of {'candidate_id', 'filename', 'text'}.

    Sends all resumes in ONE prompt and asks for one profile per resume,
    to keep Gemini usage inside the free tier budget.
    """
    resume_blocks = []
    for r in resumes:
        resume_blocks.append(f'--- RESUME ID: {r["candidate_id"]} ---\n{r["text"]}')
    joined = "\n\n".join(resume_blocks)

    prompt = f"""You are extracting structured candidate profiles from resumes for a
recruiting tool. Only extract information EXPLICITLY present in each
resume's text. If something is not mentioned, use "unknown" (for single
values) or an empty list (for lists). Do NOT guess or infer.

Do NOT extract or mention: age, gender, race, ethnicity, religion,
disability, health information, marital status, sexual orientation,
nationality, or anything about a photograph. If such information
appears in the text, ignore it entirely - it has no place in this profile.

Return ONLY a valid JSON array. One object per resume, in this exact shape:
[
  {{
    "candidate_id": "<the RESUME ID given above>",
    "skills": ["skill", ...],
    "experience_years": <number or "unknown">,
    "projects": ["short project description", ...],
    "education": ["degree/field", ...],
    "certifications": ["cert", ...],
    "evidence_notes": ["short quote or paraphrase supporting a skill claim", ...]
  }},
  ...
]

Resumes:
{joined}
"""
    result = llm.generate_json(prompt)
    if isinstance(result, dict) and "candidates" in result:
        result = result["candidates"]
    if not isinstance(result, list):
        raise ValueError("Expected a JSON array of candidate profiles from Gemini.")

    profiles = []
    for item in result:
        profiles.append(_strip_prohibited_fields(item))
    return profiles


def _strip_prohibited_fields(profile: dict) -> dict:
    """Defense in depth: even though the prompt tells Gemini not to
    include sensitive fields, we strip them locally too, so a scoring
    bug or a model hiccup can never let a prohibited field leak into
    scoring or the report.
    """
    cleaned = {}
    for key, value in profile.items():
        if key.lower() in config.PROHIBITED_FIELDS:
            continue
        cleaned[key] = value
    cleaned.setdefault("skills", [])
    cleaned.setdefault("experience_years", "unknown")
    cleaned.setdefault("projects", [])
    cleaned.setdefault("education", [])
    cleaned.setdefault("certifications", [])
    cleaned.setdefault("evidence_notes", [])
    return cleaned


# ---------------------------------------------------------------------
# 3. Requirement matching + gap analysis (batched across all candidates)
# ---------------------------------------------------------------------

def evaluate_candidates(llm: GeminiLLM, job_requirements: dict, profiles: list) -> dict:
    """One call that matches every candidate against every requirement
    and returns match status + evidence per requirement per candidate.

    This is the semantic-matching step (e.g. recognizing "built REST
    services with Flask" as partial evidence for "FastAPI"). The score
    itself is NOT computed here - scoring.py does that deterministically
    from the MATCH/PARTIAL_MATCH/NO_EVIDENCE labels this returns.
    """
    mandatory = job_requirements.get("mandatory_skills", [])
    preferred = job_requirements.get("preferred_skills", [])

    profiles_json = json.dumps(profiles, indent=2)
    prompt = f"""You are matching candidates against job requirements for a recruiting
tool. For EVERY requirement below, and EVERY candidate, decide a status:

- "MATCH": the candidate's profile clearly shows this skill/requirement.
- "PARTIAL_MATCH": related or adjacent experience, but not a clean match.
- "NO_EVIDENCE": the profile does not mention it. This does NOT mean the
  candidate lacks the skill - only that there is no evidence in the resume.

Never write "does not have" - always use NO_EVIDENCE for absence of
mention. For every MATCH or PARTIAL_MATCH, include a short evidence
string paraphrased from the candidate's profile (skills/projects/notes).
For NO_EVIDENCE, evidence should be an empty string "".

Mandatory requirements: {json.dumps(mandatory)}
Preferred requirements: {json.dumps(preferred)}

Candidate profiles:
{profiles_json}

Return ONLY valid JSON in exactly this shape:
{{
  "candidate_01": {{
    "mandatory_matches": [
      {{"requirement": "Python", "status": "MATCH", "evidence": "..."}},
      ...
    ],
    "preferred_matches": [
      {{"requirement": "Docker", "status": "NO_EVIDENCE", "evidence": ""}},
      ...
    ],
    "strengths": ["short phrase", ...],
    "gaps": ["Not evidenced in the provided resume: <requirement>", ...]
  }},
  ...
}}
Include one key per candidate_id given above.
"""
    result = llm.generate_json(prompt)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object keyed by candidate_id from Gemini.")
    return result


# ---------------------------------------------------------------------
# 4. Final report synthesis
# ---------------------------------------------------------------------

def synthesize_report_summary(llm: GeminiLLM, job_requirements: dict, ranked_summaries: list) -> str:
    """One call to write a short, plain-language executive summary.
    Everything factual (scores, ranks, coverage) is computed in Python
    and just handed to Gemini as context - Gemini is only writing prose
    here, not deciding any numbers.
    """
    prompt = f"""Write a short (150-250 word) executive summary for a recruiting
decision-support report. Be factual and neutral. Do NOT declare any
candidate "the best hire" or say a candidate "will succeed" - use
language like "recommended for human review" and "strong match based on
provided evidence". Do not mention age, gender, or any protected
characteristic. End by reminding the reader that final hiring decisions
require human review.

Role: {job_requirements.get('role', 'unknown')}

Ranked candidate summaries (already scored deterministically in Python):
{json.dumps(ranked_summaries, indent=2)}

Write only the summary text, no headers, no markdown title.
"""
    return llm.generate(prompt).strip()


# ---------------------------------------------------------------------
# Local validation (no LLM involved)
# ---------------------------------------------------------------------

def validate_profile_privacy(profile: dict) -> list:
    """Returns a list of violation strings if any prohibited field is
    present. Should always be empty after _strip_prohibited_fields runs,
    but kept as a standalone checkable function for tests / the agent's
    quality-check step.
    """
    violations = []
    for key in profile.keys():
        if key.lower() in config.PROHIBITED_FIELDS:
            violations.append(key)
    return violations
