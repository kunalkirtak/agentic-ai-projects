# 📈 Agentic Sales Agent

Project 3 of a four-project Agentic AI portfolio (Research Agent → Coding Agent → **Sales Agent** → Recruiting Agent). Runs entirely in Google Colab on the Gemini free tier.

## Overview

An autonomous **B2B sales research and lead-qualification assistant**. Give it an Ideal Customer Profile (ICP) and it plans a research strategy, searches the public web, extracts candidate companies, fetches their public pages, scores them against your ICP with transparent, evidence-backed criteria, and produces a ranked research report.

## Problem

Manually researching whether a company fits an ICP means opening a dozen tabs, reading marketing copy, and guessing at fit. It's slow and inconsistent, and it's easy to lose track of *why* a company seemed like a good fit.

## Solution

An agent that treats lead research as a multi-step workflow — not a single prompt. It plans what to search for, discovers candidates, gathers public evidence, scores each company against explicit criteria, and cites its sources for every score.

## Why This Is Agentic

This is **not** `User → Gemini → List of companies`. It is:

```
ICP → ICP Analyzer → Planner → Market Search → Company Discovery
    → Company Research → Qualification → Lead Scoring
    → Prioritization → Final Report
```

The agent maintains explicit state (`SalesState`), chooses actions from a fixed action set, runs local (free) tools between LLM calls, and loops — checking after each round whether it has enough evidence or needs to research further — inside a hard iteration cap.

## Architecture

```
                    USER ICP
                       │
                       ▼
              ┌─────────────────┐
              │ ICP ANALYZER    │  (Gemini call 1)
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    PLANNER      │  (deterministic)
              └────────┬────────┘
                       ▼
             ┌──────────────────┐
             │ MARKET SEARCH    │  (ddgs, free, local)
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ COMPANY DISCOVERY│  (Gemini call 2)
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ COMPANY RESEARCH │  (requests + bs4, local)
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ QUALIFICATION    │  (Gemini call 3)
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ LEAD SCORING     │  (deterministic, scoring.py)
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ PRIORITIZATION   │  (deterministic)
             └────────┬─────────┘
                      ▼
                FINAL REPORT       (Gemini call 4, synthesis + Markdown)
```

## Agent Loop

```
while iterations < MAX_ITERATIONS:
    inspect state
    decide next action
    execute search / research
    collect observations
    qualify companies
    check whether enough evidence exists
    if enough evidence: finish
    else: research more (bounded — never infinite)
```

`MAX_ITERATIONS = 4`. The loop always terminates, either because enough HIGH/MEDIUM priority leads were found, or because the iteration cap was hit — in which case the report is still generated with whatever evidence exists, and gaps are disclosed.

### Agent actions

`ANALYZE_ICP`, `GENERATE_SEARCH_QUERIES`, `SEARCH_MARKET`, `RESEARCH_COMPANY`, `QUALIFY_LEADS`, `IDENTIFY_GAPS`, `FINISH`

## ICP Analysis

Gemini turns a raw ICP (industry, geography, size, problem, product) into a structured strategy: target criteria, relevant public signals to look for, generated search terms, and explicit qualification/disqualification criteria — before any search happens.

## Lead Discovery

`ddgs` (free DuckDuckGo search) runs up to `MAX_SEARCH_QUERIES` targeted queries generated from the ICP. Results are batched into a single Gemini call that extracts distinct candidate companies (not directories or "top 10" articles), deduplicated by normalized name and domain.

## Company Research

For each candidate, the agent fetches a small set of public pages (`homepage`, `/about`, `/product`, etc., capped at `MAX_COMPANY_PAGES`) using `requests` + `beautifulsoup4`. No login-gated content, no aggressive crawling, no personal data.

## Lead Qualification

Every company is scored 0–3 on six criteria — industry fit, geography fit, company size fit, problem fit, technology fit, evidence quality — **only from the public text actually retrieved**. If the text doesn't support a criterion, it's marked `unknown`, never guessed.

## Lead Scoring

Scoring is **deterministic Python** (`scoring.py`), not left to the LLM to total up:

```python
def calculate_lead_score(criteria):
    ...  # sums 6 criteria, 0-3 each, "unknown" counts as 0
```

| Score | Priority |
|---|---|
| 15–18 | HIGH PRIORITY |
| 11–14 | MEDIUM PRIORITY |
| 7–10 | LOW PRIORITY |
| < 7 | INSUFFICIENT DATA |

## Evidence Tracking

Every criterion score carries a one-line evidence string and every lead carries its source URLs. Sales angles are only generated when there is supporting public evidence, and are phrased cautiously ("explore whether...") — never as certainties.

## Project Structure

```
03-sales-agent/
│
├── README.md
├── sales_agent.py        # state, agent loop, ICP/discovery/research/qualification, report builder
├── tools.py               # free web search, page fetching, dedupe/normalize helpers
├── llm.py                 # Gemini wrapper (google-genai), JSON-safe parsing, call tracking
├── config.py               # all tunable limits and model name
├── scoring.py              # deterministic scoring + priority classification
├── requirements.txt
├── .env.example
│
├── data/
│   └── sample_icps.json
│
├── outputs/
│   └── .gitkeep
│
└── tests/
    └── test_scoring.py
```

```
notebooks/
└── Agentic_Sales_Agent_Colab.ipynb
```

## Tech Stack

- Python 3
- `google-genai` (Gemini free tier)
- `ddgs` (free web search)
- `requests` + `beautifulsoup4` (public page extraction)
- No LangChain / LangGraph / CrewAI / AutoGen — the agent loop, state, and tool orchestration are implemented from first principles.

## Google Colab Setup

1. Open the notebook in Google Colab.
2. Run the dependency install cell.
3. Add your Gemini key: **Tools → Secrets → add `GEMINI_API_KEY`**, and grant this notebook access.
4. Run all cells top to bottom.

## Gemini API Setup

1. Get a free-tier key from [Google AI Studio](https://aistudio.google.com/).
2. Store it as a Colab Secret named `GEMINI_API_KEY` — never hard-code it in a cell.
3. `GEMINI_MODEL` defaults to `gemini-2.0-flash` and can be overridden (env var or Colab Secret) if a different model is available to your account.

## Running the Demo

The notebook includes two ready-made ICPs (SaaS/India and E-commerce/India, from `data/sample_icps.json`) and a prompt for a fully custom ICP. Each run prints a live agent trace, then writes:

- `outputs/leads.json` — structured lead data
- `outputs/sales_report.md` — human-readable report

## Example Output

```
🤖 AGENTIC SALES AGENT
🎯 Ideal Customer Profile
Industry: SaaS
Geography: India
...
🧠 Analyzing ICP...
✓ ICP analyzed
🔎 Market Search 1/4
✓ 5 results found
🏢 Candidate companies discovered: 8
🔍 Researching: Example Company
✓ Public company information collected
📊 Qualifying leads...
✓ Lead score generated
🔍 Checking information gaps...
✓ Sufficient evidence collected
📈 SALES RESEARCH REPORT
```

## Free-Tier Optimization

- Targets **2–4 Gemini calls per run**: ICP analysis, company discovery (batched), qualification (batched across all companies), final synthesis.
- All search, page-fetching, and scoring/prioritization logic runs locally with no LLM calls.
- Hard caps on iterations, queries, results, companies, and pages fetched (`config.py`).
- Runtime statistics (Gemini calls, search calls, companies discovered/researched, iterations) are printed and included in the report.

## Responsible AI

This is a **sales research assistant**, not an autonomous salesperson. It does not:

- send outreach (email, LinkedIn, or otherwise)
- scrape private or login-gated information
- make decisions about individuals
- infer sensitive personal characteristics
- guarantee lead quality

It only analyzes **publicly available business information** and always requires **human review before any sales outreach**.

## Security and Privacy

The agent does not collect personal email addresses, private phone numbers, home addresses, passwords, authentication information, or other sensitive personal data. It focuses exclusively on companies and public business information.

## Limitations

- Public web coverage is incomplete — many real companies won't surface from a handful of searches.
- Company size, revenue, and tech stack are often not publicly disclosed and will be marked `unknown`.
- Website structures vary; some public pages won't be found at the guessed paths.
- Scores reflect apparent fit from public marketing content, not verified qualification.
- Free-tier rate limits may cause occasional Gemini call failures — the agent degrades gracefully (deterministic fallbacks) rather than crashing.

## Future Improvements

- Pluggable search backends (e.g. a paid API) behind the same `tools.web_search` interface.
- Caching researched companies across runs to reduce repeat fetches.
- A lightweight review UI for a human to approve/reject leads before any handoff.
- Multi-language public page support.

## Author

Built as Project 3 of a four-project Agentic AI portfolio, developed and run entirely in Google Colab on the Gemini free tier.
