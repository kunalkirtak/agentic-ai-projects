"""
sales_agent.py — The Agentic B2B Sales Research Agent.

This is a research and lead-qualification assistant, NOT an autonomous
salesperson. It never contacts anyone, never collects private personal
information, and never invents evidence. Every score is backed by a
piece of public text the agent actually retrieved, or explicitly marked
"unknown". Human review is required before any outreach.

Architecture:

    ICP -> ICP ANALYZER -> PLANNER -> MARKET SEARCH -> COMPANY DISCOVERY
        -> COMPANY RESEARCH -> QUALIFICATION -> LEAD SCORING
        -> PRIORITIZATION -> FINAL REPORT

The agent runs a bounded loop (MAX_ITERATIONS) that decides, at each
step, whether it has enough evidence to finish or needs another round
of research. Each pass only sends *new* companies to Gemini — companies
already discovered/researched/qualified in an earlier pass are never
re-sent, which keeps the whole run comfortably inside a free-tier
Gemini API key's request budget (see config.py).
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import (
    MAX_COMPANIES,
    MAX_COMPANY_EXCERPT_CHARS,
    MAX_COMPANY_PAGES,
    MAX_ITERATIONS,
    MAX_SEARCH_QUERIES,
)
from llm import GeminiError, GeminiLLM
from scoring import InvalidScoreError, score_lead
from tools import (
    dedupe_companies,
    fetch_page_text,
    normalize_domain,
    polite_pause,
    web_search,
)

VALID_ACTIONS = [
    "ANALYZE_ICP",
    "GENERATE_SEARCH_QUERIES",
    "SEARCH_MARKET",
    "RESEARCH_COMPANY",
    "QUALIFY_LEADS",
    "IDENTIFY_GAPS",
    "FINISH",
]

RESEARCH_PAGE_PATHS = ["", "/about", "/product", "/solutions", "/technology", "/industries", "/blog"]


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
@dataclass
class SalesState:
    icp_raw: dict
    icp_analysis: dict = field(default_factory=dict)
    plan: list = field(default_factory=list)
    search_queries: list = field(default_factory=list)
    search_results: list = field(default_factory=list)
    candidate_companies: list = field(default_factory=list)
    researched_companies: list = field(default_factory=list)
    qualified_leads: list = field(default_factory=list)
    insufficient_leads: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    iterations: int = 0
    action_log: list = field(default_factory=list)
    final_report: str = ""
    # Internal bookkeeping so each loop pass only processes *new* work
    # (new search results / new candidates) instead of re-sending
    # already-processed companies to Gemini on every iteration.
    discovered_result_count: int = 0

    def log(self, action: str, detail: str = ""):
        self.action_log.append({"iteration": self.iterations, "action": action, "detail": detail})


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class SalesAgent:
    def __init__(self, llm: GeminiLLM, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.search_call_count = 0
        self.companies_researched_count = 0

    def _say(self, message: str):
        if self.verbose:
            print(message)

    # -- ANALYZE_ICP ---------------------------------------------------
    def analyze_icp(self, state: SalesState) -> SalesState:
        self._say("\n🧠 Analyzing ICP...")
        prompt = f"""
You are a B2B sales research strategist. Turn this raw Ideal Customer
Profile (ICP) into a structured research strategy.

Raw ICP:
{json.dumps(state.icp_raw, indent=2)}

Return a JSON object with exactly these keys:
- "target_industry": string
- "target_geography": string
- "company_size": string
- "ideal_customer_problem": string
- "relevant_signals": array of short strings (public signals that would
  suggest a company fits this ICP, e.g. "publicly discusses manual
  processes", "recent hiring for automation roles")
- "search_terms": array of 3-5 short search-engine-style query strings
  (not full sentences) likely to surface real candidate companies
- "qualification_criteria": array of short strings describing what
  would count as strong evidence of fit
- "disqualification_criteria": array of short strings describing what
  would rule a company out
"""
        try:
            analysis = self.llm.generate_json(prompt)
        except GeminiError as exc:
            self._say(f"  ⚠ ICP analysis failed ({exc}); falling back to the raw ICP.")
            analysis = self._fallback_icp_analysis(state.icp_raw)

        state.icp_analysis = analysis
        state.plan = [
            "Generate search queries from ICP",
            "Search public web for candidate companies",
            "Research top candidates from public pages",
            "Qualify and score leads against ICP",
            "Identify information gaps",
            "Produce final research report",
        ]
        state.log("ANALYZE_ICP", "ICP structured into research strategy")
        self._say("✓ ICP analyzed")
        return state

    @staticmethod
    def _fallback_icp_analysis(icp_raw: dict) -> dict:
        """Deterministic fallback if Gemini is unavailable, so the agent can still run."""
        industry = icp_raw.get("industry", "")
        geography = icp_raw.get("geography", "")
        size = icp_raw.get("company_size", "")
        return {
            "target_industry": industry,
            "target_geography": geography,
            "company_size": size,
            "ideal_customer_problem": icp_raw.get("customer_problem", ""),
            "relevant_signals": ["publicly discusses relevant operational challenges"],
            "search_terms": [
                f"{industry} companies {geography}",
                f"{industry} startups {geography} {size} employees",
            ],
            "qualification_criteria": ["industry match", "geography match", "size match"],
            "disqualification_criteria": ["unrelated industry", "no public presence"],
        }

    # -- GENERATE_SEARCH_QUERIES ----------------------------------------
    def generate_search_queries(self, state: SalesState) -> SalesState:
        self._say("\n📋 Creating search strategy...")
        terms = state.icp_analysis.get("search_terms") or []
        queries = [t for t in terms if isinstance(t, str) and t.strip()][:MAX_SEARCH_QUERIES]
        if not queries:
            queries = [
                f"{state.icp_analysis.get('target_industry', '')} companies "
                f"{state.icp_analysis.get('target_geography', '')}"
            ]
        state.search_queries = queries
        state.log("GENERATE_SEARCH_QUERIES", f"{len(queries)} queries created")
        self._say(f"✓ {len(queries)} search queries created")
        return state

    # -- SEARCH_MARKET ---------------------------------------------------
    def search_market(self, state: SalesState) -> SalesState:
        for i, query in enumerate(state.search_queries, start=1):
            self._say(f"\n🔎 Market Search {i}/{len(state.search_queries)}: \"{query}\"")
            results = web_search(query)
            self.search_call_count += 1
            state.search_results.extend(results)
            self._say(f"✓ {len(results)} results found")
            polite_pause()
        state.log("SEARCH_MARKET", f"{len(state.search_results)} total results collected")
        return state

    # -- COMPANY DISCOVERY ------------------------------------------------
    def discover_companies(self, state: SalesState) -> SalesState:
        """Extract candidate companies from any search results not yet
        processed. Safe to call every loop pass: it's a no-op (and makes
        no Gemini call) once there's nothing new to look at."""
        new_results = state.search_results[state.discovered_result_count:]
        if not new_results:
            state.log("RESEARCH_COMPANY", "no new search results to extract companies from")
            return state

        remaining_slots = MAX_COMPANIES - len(state.candidate_companies)
        if remaining_slots <= 0:
            state.discovered_result_count = len(state.search_results)
            state.log("RESEARCH_COMPANY", "candidate list already full; skipping discovery call")
            return state

        self._say("\n🏢 Extracting candidate companies from search results...")

        # Batch all new search results into a single Gemini call to extract
        # structured company candidates (keeps LLM usage low).
        results_text = "\n".join(
            f"- title: {r['title']} | url: {r['url']} | snippet: {r['snippet']}"
            for r in new_results
            if r.get("url")
        )
        prompt = f"""
From these web search results, extract distinct real companies that
could plausibly be evaluated as B2B sales leads. Ignore directories,
news aggregators, "top 10 lists" articles (unless a specific company
is clearly named), and non-company pages.

Search results:
{results_text}

Return a JSON object with one key "companies", an array of up to
{remaining_slots} objects, each with:
- "company": company name
- "website": best-guess homepage URL (from the results, or your best
  inference from the domain in the result URL)
- "source_urls": array of the result URLs that mention this company
- "discovery_reason": one short sentence on why this looked like a
  plausible candidate given the search context
"""
        try:
            data = self.llm.generate_json(prompt)
            new_companies = data.get("companies", [])
        except GeminiError as exc:
            self._say(f"  ⚠ company extraction failed ({exc}); using raw search results as candidates.")
            new_companies = self._fallback_discovery(new_results)

        combined = dedupe_companies(state.candidate_companies + new_companies)[:MAX_COMPANIES]
        state.candidate_companies = combined
        state.discovered_result_count = len(state.search_results)
        state.log("RESEARCH_COMPANY", f"{len(combined)} candidate companies discovered so far")
        self._say(f"✓ Candidate companies discovered: {len(combined)}")
        return state

    @staticmethod
    def _fallback_discovery(search_results: list) -> list:
        """Deterministic fallback: treat each unique domain as a candidate."""
        seen = set()
        companies = []
        for r in search_results:
            domain = normalize_domain(r.get("url", ""))
            if not domain or domain in seen:
                continue
            seen.add(domain)
            companies.append(
                {
                    "company": r.get("title", domain).split("|")[0].strip(),
                    "website": r.get("url", ""),
                    "source_urls": [r.get("url", "")],
                    "discovery_reason": "Appeared in market search results (automated fallback, no Gemini call used).",
                }
            )
        return companies[:MAX_COMPANIES]

    # -- COMPANY RESEARCH ---------------------------------------------------
    def research_companies(self, state: SalesState) -> SalesState:
        """Fetch public pages for any candidate companies not yet researched.
        Uses only local, free HTTP requests — no Gemini calls."""
        already = {c.get("company") for c in state.researched_companies}
        pending = [c for c in state.candidate_companies if c.get("company") not in already]
        if not pending:
            state.log("RESEARCH_COMPANY", "no new candidate companies to research")
            return state

        for company in pending:
            name = company.get("company", "unknown company")
            website = company.get("website", "")
            self._say(f"\n🔍 Researching: {name}")

            pages_text = []
            if website:
                base = website.rstrip("/")
                for path in RESEARCH_PAGE_PATHS[:MAX_COMPANY_PAGES]:
                    url = base + path
                    text = fetch_page_text(url)
                    if text:
                        pages_text.append({"url": url, "text": text})
                    polite_pause()

            self.companies_researched_count += 1
            researched = {
                **company,
                "public_pages": pages_text,
                "has_public_data": bool(pages_text),
            }
            state.researched_companies.append(researched)

            if pages_text:
                self._say("✓ Public company information collected")
            else:
                self._say("  ⚠ no public page content retrieved (site unreachable or blocked)")

        state.log("RESEARCH_COMPANY", f"{len(state.researched_companies)} companies researched so far")
        return state

    # -- QUALIFY_LEADS ---------------------------------------------------
    def qualify_leads(self, state: SalesState) -> SalesState:
        """Score any researched companies not yet qualified. Skips the
        Gemini call entirely (and logs why) when there's nothing new,
        which is what prevents the agentic loop from re-scoring the same
        companies — and burning the same request budget — on every pass."""
        already_seen = {l["company"] for l in state.qualified_leads} | {
            i["company"] for i in state.insufficient_leads
        }
        pending = [c for c in state.researched_companies if c.get("company") not in already_seen]

        if not pending:
            if not state.researched_companies:
                state.log("QUALIFY_LEADS", "no researched companies to qualify")
            else:
                state.log("QUALIFY_LEADS", "no new companies since last pass; skipping duplicate Gemini call")
            return state

        self._say("\n📊 Qualifying leads...")

        # One batched Gemini call scores all newly researched companies together.
        companies_payload = []
        for c in pending:
            combined_text = " ".join(p["text"] for p in c.get("public_pages", []))[:MAX_COMPANY_EXCERPT_CHARS]
            companies_payload.append(
                {
                    "company": c.get("company"),
                    "website": c.get("website"),
                    "public_text_excerpt": combined_text or "(no public page text retrieved)",
                    "source_urls": c.get("source_urls", []),
                }
            )

        prompt = f"""
You are qualifying B2B sales leads against this ICP:
{json.dumps(state.icp_analysis, indent=2)}

For each company below, score these six criteria on a 0-3 scale:
0 = no evidence, 1 = weak, 2 = moderate, 3 = strong.
Criteria: industry_fit, geography_fit, company_size_fit, problem_fit,
technology_fit, evidence_quality.

CRITICAL RULES:
- Base every score ONLY on the provided public_text_excerpt or
  source_urls. Do not invent facts about the company.
- If the excerpt does not support a criterion, set that criterion's
  "score" to the string "unknown" rather than guessing a number.
- Every criterion needs a one-sentence "evidence" string quoting or
  paraphrasing what in the text supports the score, or "no public
  evidence found" if unknown.
- Suggest a cautious "sales_angle" (one sentence, phrased as a
  possibility, e.g. "Explore whether X could help with Y") ONLY if
  there is supporting evidence; otherwise use "insufficient evidence
  for a sales angle".

Companies:
{json.dumps(companies_payload, indent=2)}

Return a JSON object with key "qualifications": an array, one entry
per company, each with:
{{
  "company": "...",
  "criteria": {{
     "industry_fit": {{"score": 0-3 or "unknown", "evidence": "..."}},
     "geography_fit": {{...}},
     "company_size_fit": {{...}},
     "problem_fit": {{...}},
     "technology_fit": {{...}},
     "evidence_quality": {{...}}
  }},
  "sales_angle": "...",
  "missing_information": ["..."]
}}
"""
        try:
            data = self.llm.generate_json(prompt)
            qualifications = data.get("qualifications", [])
        except GeminiError as exc:
            self._say(f"  ⚠ qualification call failed ({exc}); using automated keyword-fallback scoring.")
            qualifications = self._fallback_qualify(companies_payload, state.icp_analysis)

        qual_by_company = {q.get("company"): q for q in qualifications}

        for c in pending:
            name = c.get("company")
            qual = qual_by_company.get(name)
            sources = c.get("source_urls", [])
            if not qual:
                state.insufficient_leads.append(
                    {
                        "company": name,
                        "website": c.get("website", ""),
                        "reason": "No qualification data returned for this company.",
                        "sources": sources,
                    }
                )
                continue
            try:
                lead = score_lead(name, qual.get("criteria", {}), sources)
                lead["website"] = c.get("website", "")
                lead["sales_angle"] = qual.get("sales_angle", "insufficient evidence for a sales angle")
                lead["missing_information"] = qual.get("missing_information", [])
                lead["discovery_reason"] = c.get("discovery_reason", "")
                state.qualified_leads.append(lead)
            except InvalidScoreError as exc:
                state.insufficient_leads.append(
                    {
                        "company": name,
                        "website": c.get("website", ""),
                        "reason": f"Malformed scoring data: {exc}",
                        "sources": sources,
                    }
                )

        state.qualified_leads.sort(key=lambda lead: lead["score"], reverse=True)
        state.log("QUALIFY_LEADS", f"{len(pending)} new companies scored this pass ({len(state.qualified_leads)} total leads)")
        self._say("✓ Lead score generated" if state.qualified_leads else "⚠ no leads could be scored")
        return state

    @staticmethod
    def _fallback_qualify(companies_payload: list, icp_analysis: dict) -> list:
        """Deterministic keyword-matching fallback used when Gemini is
        unavailable (API error, or the session's free-tier call budget has
        been used up). It never invents facts: criteria are only ever
        "unknown", 1 (weak keyword match), or 2 (direct keyword match) —
        never a confident 3 — and every evidence string says plainly that
        this is automated fallback scoring, not a Gemini judgement. This is
        what lets the notebook always finish with a real, populated report
        even if the free-tier key runs out of quota mid-run."""
        industry = str(icp_analysis.get("target_industry", "")).lower().strip()
        geography = str(icp_analysis.get("target_geography", "")).lower().strip()
        problem_words = [
            w for w in str(icp_analysis.get("ideal_customer_problem", "")).lower().split()
            if len(w) > 4
        ]
        signal_words = [str(s).lower() for s in icp_analysis.get("relevant_signals", []) if s]

        def crit(has_text: bool, hit: bool) -> dict:
            if not has_text:
                return {"score": "unknown", "evidence": "no public evidence found"}
            if hit:
                return {"score": 2, "evidence": "keyword match found in public page text (automated fallback scoring, no Gemini call used)"}
            return {"score": "unknown", "evidence": "public text present but no direct keyword match (automated fallback scoring)"}

        qualifications = []
        for c in companies_payload:
            text = str(c.get("public_text_excerpt", "")).lower()
            has_text = bool(text) and text != "(no public page text retrieved)"
            qualifications.append(
                {
                    "company": c.get("company"),
                    "criteria": {
                        "industry_fit": crit(has_text, bool(industry) and industry in text),
                        "geography_fit": crit(has_text, bool(geography) and geography in text),
                        "company_size_fit": {"score": "unknown", "evidence": "company size is not reliably inferable without a Gemini analysis pass"},
                        "problem_fit": crit(has_text, any(w in text for w in problem_words)),
                        "technology_fit": crit(has_text, any(w in text for w in signal_words)),
                        "evidence_quality": crit(has_text, has_text),
                    },
                    "sales_angle": "insufficient evidence for a sales angle",
                    "missing_information": [
                        "Full qualification unavailable this run — Gemini free-tier budget was reached, "
                        "so this company was scored with automated keyword matching instead."
                    ],
                }
            )
        return qualifications

    # -- IDENTIFY_GAPS ---------------------------------------------------
    def identify_gaps(self, state: SalesState) -> bool:
        """Returns True if evidence is sufficient to finish, False if another loop is warranted."""
        self._say("\n🔍 Checking information gaps...")
        gaps = []
        if not state.qualified_leads:
            gaps.append("No leads were successfully scored.")
        strong_leads = [l for l in state.qualified_leads if l["priority"] in ("HIGH PRIORITY", "MEDIUM PRIORITY")]
        if not strong_leads and state.iterations < MAX_ITERATIONS - 1:
            gaps.append("No HIGH/MEDIUM priority leads yet found.")
        state.gaps = gaps
        state.log("IDENTIFY_GAPS", "; ".join(gaps) if gaps else "sufficient evidence")

        if not gaps:
            self._say("✓ Sufficient evidence collected")
            return True

        if state.iterations >= MAX_ITERATIONS - 1:
            self._say(f"✓ Reached max iterations ({MAX_ITERATIONS}); finishing with available evidence")
            return True

        self._say(f"⚠ Gaps found: {gaps}. Will attempt additional research.")
        return False

    # -- FINAL REPORT ---------------------------------------------------
    def generate_report(self, state: SalesState) -> SalesState:
        self._say("\n🧠 Generating final sales research report...")
        stats = self.llm.stats()
        stats["search_calls"] = self.search_call_count
        stats["companies_discovered"] = len(state.candidate_companies)
        stats["companies_researched"] = self.companies_researched_count
        stats["agent_iterations"] = state.iterations

        report = build_markdown_report(state, stats)
        state.final_report = report
        state.log("FINISH", "final report generated")
        self._say("✓ Report generated")
        return state

    # -- AGENT LOOP ---------------------------------------------------
    def run(self, icp_raw: dict) -> SalesState:
        state = SalesState(icp_raw=icp_raw)

        self._say("=" * 56)
        self._say("🤖 AGENTIC SALES AGENT")
        self._say("=" * 56)
        self._say(f"\n🎯 Ideal Customer Profile\n")
        for key, value in icp_raw.items():
            self._say(f"{key.replace('_', ' ').title()}: {value}")

        state = self.analyze_icp(state)
        state = self.generate_search_queries(state)

        done = False
        while state.iterations < MAX_ITERATIONS and not done:
            state.iterations += 1
            self._say(f"\n--- Agent iteration {state.iterations}/{MAX_ITERATIONS} ---")

            if not state.search_results:
                state = self.search_market(state)
            state = self.discover_companies(state)
            state = self.research_companies(state)
            state = self.qualify_leads(state)
            done = self.identify_gaps(state)

            if not done:
                # Extra loop: broaden with the next unused search term, if
                # any, so the *next* pass has genuinely new search results
                # to discover/research/qualify from — otherwise stop to
                # respect MAX_ITERATIONS.
                extra_terms = state.icp_analysis.get("search_terms", [])
                used = set(state.search_queries)
                remaining = [t for t in extra_terms if t not in used]
                if remaining:
                    query = remaining[0]
                    state.search_queries.append(query)
                    self._say(f"\n🔎 Broadening search: \"{query}\"")
                    extra_results = web_search(query)
                    self.search_call_count += 1
                    state.search_results.extend(extra_results)
                    polite_pause()
                else:
                    done = True

        state = self.generate_report(state)

        self._say("\n" + "=" * 56)
        self._say("📈 Run complete. See state.final_report / outputs/ for details.")
        self._say("=" * 56)
        return state


# ---------------------------------------------------------------------------
# Report generation (Markdown)
# ---------------------------------------------------------------------------
def build_markdown_report(state: SalesState, stats: dict) -> str:
    icp = state.icp_analysis or state.icp_raw
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# 📈 Sales Research Report")
    lines.append(f"_Generated {now}_\n")

    lines.append("## Executive Summary")
    high = [l for l in state.qualified_leads if l["priority"] == "HIGH PRIORITY"]
    medium = [l for l in state.qualified_leads if l["priority"] == "MEDIUM PRIORITY"]
    lines.append(
        f"This report evaluates {len(state.researched_companies)} publicly "
        f"discoverable companies against the target ICP. "
        f"{len(high)} were classified HIGH PRIORITY and {len(medium)} MEDIUM "
        f"PRIORITY. All scores are derived from public web content only and "
        f"require human review before any outreach.\n"
    )

    lines.append("## Ideal Customer Profile")
    for key, value in icp.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    lines.append("")

    lines.append("## Research Methodology")
    lines.append(
        "1. The raw ICP was analyzed by Gemini into a structured research strategy.\n"
        "2. Multiple targeted web searches (DuckDuckGo, free tier) were run to surface candidate companies.\n"
        "3. Public company pages (homepage, about, product, etc.) were fetched for top candidates.\n"
        "4. Each company was scored against six criteria using only retrieved public text, with 'unknown' used instead of guessing where evidence was absent.\n"
        "5. Deterministic Python scoring (see `scoring.py`) converted per-criterion scores into a total and a priority bucket.\n"
    )

    lines.append("## Top Qualified Leads")
    if not state.qualified_leads:
        lines.append("_No leads met the scoring bar this run. See Limitations below._\n")
    else:
        for lead in state.qualified_leads[:5]:
            lines.append(f"- **{lead['company']}** — {lead['score']}/{lead['max_score']} ({lead['priority']})")
        lines.append("")

    lines.append("## Lead Scoring")
    lines.append(
        "Score = industry_fit + geography_fit + company_size_fit + problem_fit + "
        "technology_fit + evidence_quality (each 0-3, max 18).\n"
        "- 15-18 → HIGH PRIORITY\n"
        "- 11-14 → MEDIUM PRIORITY\n"
        "- 7-10 → LOW PRIORITY\n"
        "- below 7 → INSUFFICIENT DATA\n"
    )

    lines.append("## Company-by-Company Analysis")
    if not state.qualified_leads:
        lines.append("_No scored leads to display._\n")
    for lead in state.qualified_leads:
        lines.append(f"### {lead['company']}")
        lines.append(f"- Website: {lead.get('website', 'unknown')}")
        lines.append(f"- Score: {lead['score']}/{lead['max_score']} — **{lead['priority']}**")
        lines.append(f"- Why relevant: {lead.get('discovery_reason', 'n/a')}")
        lines.append("- Evidence by criterion:")
        for crit, entry in lead.get("criteria", {}).items():
            score = entry.get("score", "unknown")
            evidence = entry.get("evidence", "unknown")
            lines.append(f"  - {crit.replace('_', ' ').title()}: {score} — {evidence}")
        lines.append(f"- Recommended sales angle: {lead.get('sales_angle', 'insufficient evidence for a sales angle')}")
        if lead.get("missing_information"):
            lines.append(f"- Missing information: {', '.join(lead['missing_information'])}")
        if lead.get("sources"):
            lines.append(f"- Sources: {', '.join(lead['sources'])}")
        lines.append("")

    if state.insufficient_leads:
        lines.append("## Insufficient Data")
        for item in state.insufficient_leads:
            lines.append(f"- **{item['company']}** — {item['reason']}")
        lines.append("")

    lines.append("## Missing Information")
    if state.gaps:
        for gap in state.gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- None identified for this run.")
    lines.append("")

    lines.append("## Sources")
    all_sources = sorted({s for lead in state.qualified_leads for s in lead.get("sources", [])})
    if all_sources:
        for s in all_sources:
            lines.append(f"- {s}")
    else:
        lines.append("- No sources recorded.")
    lines.append("")

    lines.append("## Limitations")
    lines.append(
        "- This report reflects only what public web pages and search "
        "results made available at run time; it is not exhaustive.\n"
        "- Company size, revenue, and technology stack are frequently not "
        "publicly disclosed and are marked 'unknown' rather than guessed.\n"
        "- Scores reflect *apparent* fit based on public marketing "
        "content, not verified qualification — a human must confirm "
        "before any outreach.\n"
        "- No company listed here has been contacted, and none should be "
        "treated as having opted in to any communication.\n"
        "- If the Gemini free-tier request budget was reached mid-run, "
        "remaining companies were scored with an automated keyword "
        "fallback (clearly labeled in 'Missing information' above) "
        "instead of a Gemini judgement call.\n"
    )

    lines.append("## Runtime Statistics")
    for key, value in stats.items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)


def state_to_json(state: SalesState) -> dict:
    return {
        "icp": state.icp_analysis or state.icp_raw,
        "leads": state.qualified_leads,
        "insufficient_data": state.insufficient_leads,
        "gaps": state.gaps,
        "action_log": state.action_log,
    }
