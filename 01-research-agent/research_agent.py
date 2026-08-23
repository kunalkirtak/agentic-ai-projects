"""
research_agent.py
------------------
The core agentic system: state, prompt templates, and the agent loop that
ties the LLM (llm.py) and the tools (tools.py) together.

This file deliberately implements the agent loop BY HAND (no LangChain /
LangGraph / CrewAI / AutoGen) to demonstrate understanding of the
fundamentals of agentic AI:

    inspect_state -> decide_next_action -> execute_tool ->
    save_observation -> evaluate_information_gaps -> ... -> synthesize

The loop is bounded by config.max_agent_steps so it can never run away on
a free-tier API key.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import Config
from llm import GeminiLLM, GeminiError, safe_json_parse
from tools import web_search, fetch_webpage, extract_evidence

logger = logging.getLogger("research_agent.core")


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------
@dataclass
class ResearchState:
    """
    Explicit, persistent state for a single research run.

    Keeping this as one object (rather than scattering variables through
    the loop) is what makes this an *agent* with memory of its own
    progress, rather than a single stateless prompt->response call.
    """

    question: str
    plan: List[str] = field(default_factory=list)
    searches: List[Dict] = field(default_factory=list)     # {"query": ..., "result_count": ...}
    sources: List[Dict] = field(default_factory=list)       # raw search hits, deduped by URL
    evidence: List[Dict] = field(default_factory=list)       # structured evidence records
    gaps: List[str] = field(default_factory=list)            # open sub-questions still unanswered
    analyst_notes: List[Dict] = field(default_factory=list)  # raw analyst JSON per round
    final_report: str = ""
    steps: int = 0
    llm_calls: int = 0
    search_calls: int = 0

    def seen_urls(self) -> set:
        return {s["url"] for s in self.sources if s.get("url")}


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are a research planner. Break the user's research question
into 2 to 5 focused, non-overlapping sub-questions that, together, would let
someone write a thorough answer. Prefer sub-questions that map naturally to
things a web search could answer.

Research question: {question}

Respond with ONLY valid JSON in this exact shape, no extra commentary:
{{
  "sub_questions": ["...", "...", "..."]
}}
"""

ANALYST_PROMPT = """You are a careful research analyst. Below is evidence collected
so far for the research question: {question}

Sub-questions being investigated:
{sub_questions}

Evidence collected so far:
{evidence_block}

Analyze this evidence and respond with ONLY valid JSON in this exact shape:
{{
  "key_findings": ["...", "..."],
  "conflicting_information": ["..."],
  "missing_information": ["..."],
  "needs_more_research": true,
  "suggested_followup_queries": ["...", "..."]
}}

Rules:
- "needs_more_research" should be false only if the evidence already
  reasonably covers the sub-questions.
- "suggested_followup_queries" should be empty if needs_more_research is false.
- Only claim a conflict if sources genuinely disagree.
- Do not invent findings that are not supported by the evidence above.
"""

SYNTHESIS_PROMPT = """You are a research analyst writing a final report. Use ONLY the
evidence provided below -- do not invent facts, statistics, or sources that
are not present in it.

Research question: {question}

Research plan (sub-questions investigated):
{sub_questions}

All evidence collected (each item includes its source URL):
{evidence_block}

Analyst notes from the investigation:
{analyst_summary}

Write a final report with EXACTLY these sections, using Markdown headings:

## Executive Summary
A short (3-5 sentence) overview of the answer to the research question.

## Key Findings
A bulleted list of the most important, well-supported findings.

## Detailed Analysis
A few paragraphs synthesizing the evidence into a coherent narrative,
noting agreement/disagreement between sources where relevant.

## Evidence
A bulleted list connecting specific claims to specific sources (cite by
URL or title).

## Limitations
Note gaps in the research, remaining uncertainty, and the general caveat
that web search results are not guaranteed to be complete or accurate.

## Sources
A numbered list of every source URL actually used above. Do not invent
URLs that were not provided in the evidence.
"""


# ---------------------------------------------------------------------------
# Helper: format evidence for prompts (keeps prompts compact)
# ---------------------------------------------------------------------------
def _format_evidence_block(evidence: List[Dict]) -> str:
    if not evidence:
        return "(no evidence collected yet)"
    lines = []
    for i, item in enumerate(evidence, start=1):
        points = "; ".join(item.get("key_points", []))
        lines.append(f"[{i}] {item.get('title', 'Untitled')} ({item.get('source', 'no-url')})\n    {points}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The Research Agent
# ---------------------------------------------------------------------------
class ResearchAgent:
    """
    Orchestrates the full agentic research workflow:

        Plan -> Search -> Read -> Collect Evidence -> Identify Gaps ->
        (Search again if needed) -> Synthesize Final Report

    The agent decides for itself, based on the analyst's assessment,
    whether another round of searching is warranted -- bounded by
    config.max_agent_steps so it can never loop forever.
    """

    def __init__(self, config: Config, llm: Optional[GeminiLLM] = None):
        self.config = config
        self.llm = llm or GeminiLLM(api_key=config.gemini_api_key, model=config.gemini_model)

    # -- Step: Planning ----------------------------------------------------
    def _plan(self, state: ResearchState, log) -> None:
        log("PLAN", f"Planning research for: {state.question}")
        prompt = PLANNER_PROMPT.format(question=state.question)
        try:
            raw = self.llm.generate(prompt)
            state.llm_calls += 1
            parsed = safe_json_parse(raw)
            sub_qs = (parsed or {}).get("sub_questions") or []
            sub_qs = [q for q in sub_qs if isinstance(q, str) and q.strip()][:5]
        except GeminiError as exc:
            log("WARN", f"Planner LLM call failed ({exc}); falling back to the raw question.")
            sub_qs = []

        if not sub_qs:
            sub_qs = [state.question]

        state.plan = sub_qs
        log("OK", f"Research plan created ({len(sub_qs)} sub-questions).")

    # -- Step: Search + Read + Collect Evidence -----------------------------
    def _research_round(self, state: ResearchState, queries: List[str], log) -> None:
        for query in queries:
            if len(state.sources) >= self.config.max_sources:
                log("INFO", "Source limit reached; skipping further searches this round.")
                break

            log("SEARCH", f'"{query}"')
            results = web_search(query, max_results=self.config.max_search_results)
            state.search_calls += 1
            state.searches.append({"query": query, "result_count": len(results)})

            if not results:
                log("WARN", "No results found for this query.")
                continue
            log("OK", f"{len(results)} results found")

            for result in results:
                if len(state.sources) >= self.config.max_sources:
                    break
                url = result.get("url")
                if not url or url in state.seen_urls():
                    continue

                state.sources.append(result)

                log("READ", url)
                page_text = fetch_webpage(url, max_chars=self.config.max_page_chars, timeout=self.config.request_timeout)
                if not page_text:
                    # Fall back to the search snippet so the source isn't wasted.
                    page_text = result.get("snippet", "")

                evidence = extract_evidence(
                    source_title=result.get("title", ""),
                    source_url=url,
                    page_text=page_text,
                    query_context=query,
                )
                if evidence:
                    state.evidence.append(evidence)
                    log("OK", "Evidence collected")
                else:
                    log("WARN", "No usable evidence extracted from this source.")

    # -- Step: Analyze gaps --------------------------------------------------
    def _analyze(self, state: ResearchState, log) -> Dict:
        log("ANALYZE", "Analyzing research gaps...")
        prompt = ANALYST_PROMPT.format(
            question=state.question,
            sub_questions="\n".join(f"- {q}" for q in state.plan),
            evidence_block=_format_evidence_block(state.evidence),
        )
        try:
            raw = self.llm.generate(prompt)
            state.llm_calls += 1
            parsed = safe_json_parse(raw) or {}
        except GeminiError as exc:
            log("WARN", f"Analyst LLM call failed ({exc}); assuming research is sufficient.")
            parsed = {}

        parsed.setdefault("key_findings", [])
        parsed.setdefault("conflicting_information", [])
        parsed.setdefault("missing_information", [])
        parsed.setdefault("needs_more_research", False)
        parsed.setdefault("suggested_followup_queries", [])

        state.analyst_notes.append(parsed)
        state.gaps = parsed.get("missing_information", [])

        if parsed["needs_more_research"]:
            log("OK", "Additional research required")
        else:
            log("OK", "Evidence looks sufficient")

        return parsed

    # -- Step: Synthesize final report ---------------------------------------
    def _synthesize(self, state: ResearchState, log) -> None:
        log("SYNTHESIZE", "Synthesizing final report...")

        analyst_summary_lines = []
        for i, note in enumerate(state.analyst_notes, start=1):
            analyst_summary_lines.append(
                f"Round {i}: findings={note.get('key_findings')}; "
                f"conflicts={note.get('conflicting_information')}; "
                f"missing={note.get('missing_information')}"
            )
        analyst_summary = "\n".join(analyst_summary_lines) or "(no analyst notes)"

        prompt = SYNTHESIS_PROMPT.format(
            question=state.question,
            sub_questions="\n".join(f"- {q}" for q in state.plan),
            evidence_block=_format_evidence_block(state.evidence),
            analyst_summary=analyst_summary,
        )

        try:
            report = self.llm.generate(prompt, max_output_tokens=3072)
            state.llm_calls += 1
        except GeminiError as exc:
            log("WARN", f"Synthesis LLM call failed ({exc}); building a fallback report.")
            report = self._fallback_report(state)

        state.final_report = report

    def _fallback_report(self, state: ResearchState) -> str:
        """A deterministic, no-LLM report used only if Gemini is unavailable
        at the synthesis step, so the user always gets *something* back."""
        sources = "\n".join(f"- {s.get('title', 'Untitled')}: {s.get('url', '')}" for s in state.sources)
        evidence = _format_evidence_block(state.evidence)
        return (
            f"## Executive Summary\n"
            f"Automated synthesis was unavailable, so this is a raw evidence dump "
            f"for: {state.question}\n\n"
            f"## Evidence\n{evidence}\n\n"
            f"## Limitations\nThe final synthesis LLM call failed; this report is "
            f"unprocessed evidence only. Web search results are not guaranteed to "
            f"be complete or accurate.\n\n"
            f"## Sources\n{sources}\n"
        )

    # -- The Agent Loop -------------------------------------------------------
    def run(self, question: str, verbose: bool = True) -> ResearchState:
        """
        Execute the full agentic research workflow for a single question
        and return the populated ResearchState.
        """
        state = ResearchState(question=question)

        def log(tag: str, message: str) -> None:
            if verbose:
                icon = {
                    "PLAN": "\U0001F9E0", "SEARCH": "\U0001F50E", "READ": "\U0001F4C4",
                    "ANALYZE": "\U0001F9E0", "SYNTHESIZE": "\U0001F9E0",
                    "OK": "\u2713", "WARN": "\u26A0", "INFO": "\u2139",
                }.get(tag, "-")
                print(f"{icon} {message}")
            logger.info("[%s] %s", tag, message)

        if verbose:
            print("=" * 50)
            print("\U0001F916 AGENTIC RESEARCH AGENT")
            print("=" * 50)
            print(f"\n\U0001F3AF Goal:\n{question}\n")

        # 1) Plan
        state.steps += 1
        self._plan(state, log)

        # 2) Initial research round, based on the plan
        state.steps += 1
        if verbose:
            print()
        self._research_round(state, state.plan, log)

        # 3) Analyze / iterate while budget remains
        while state.steps < self.config.max_agent_steps:
            state.steps += 1
            if verbose:
                print()
            analysis = self._analyze(state, log)

            if not analysis.get("needs_more_research"):
                break

            followups = [q for q in analysis.get("suggested_followup_queries", []) if isinstance(q, str) and q.strip()]
            followups = [q for q in followups if q not in [s["query"] for s in state.searches]][:2]

            if not followups or len(state.sources) >= self.config.max_sources:
                log("INFO", "No further useful searches available within budget; moving to synthesis.")
                break

            if verbose:
                print()
            self._research_round(state, followups, log)

        # 4) Final synthesis (always runs, even if the loop hit the step cap)
        if verbose:
            print()
        self._synthesize(state, log)

        if verbose:
            print("\n" + "=" * 50)
            print("\U0001F4CA FINAL RESEARCH REPORT")
            print("=" * 50 + "\n")
            print(state.final_report)
            print("\n" + "-" * 50)
            print(
                f"Run summary: steps={state.steps} | llm_calls={state.llm_calls} | "
                f"search_calls={state.search_calls} | sources={len(state.sources)}"
            )
            print("-" * 50)

        return state


def build_agent(gemini_api_key: str) -> ResearchAgent:
    """Convenience factory used by the notebook / GitHub demo entry point."""
    config = Config(gemini_api_key=gemini_api_key)
    return ResearchAgent(config=config)
