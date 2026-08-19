# Agentic Research Agent

Project 1 of a 4-part Agentic AI portfolio (`01 Research Agent → 02 Coding Agent → 03 Sales Agent → 04 Recruiting Agent`).

A research agent that **plans, searches, reads, gathers evidence, detects its own information gaps, searches again if needed, and synthesizes a sourced report** — built from scratch in Python, runnable entirely in Google Colab on a free Gemini API key.

## Overview

Most "AI research tools" are a single call:

```
Question → LLM → Answer
```

That's not an agent — it's a prompt. This project implements an actual **agentic loop**: the system maintains state across multiple steps, decides for itself which tool to call next, evaluates whether its own evidence is sufficient, and only stops when it has enough to answer (or it hits a hard step budget).

```
User Goal → Planning → Search → Read Sources → Collect Evidence →
Identify Information Gaps → Additional Search (if needed) →
Synthesize → Final Research Report
```

No LangChain, LangGraph, CrewAI, or AutoGen — the planning/tool-use/state loop is implemented directly so the mechanics of an agent are fully visible and understandable, not hidden behind a framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ResearchAgent                            │
│                                                                    │
│   ┌───────────┐    ┌──────────────┐    ┌────────────────────┐    │
│   │  Planner  │───▶│  Agent Loop  │───▶│  Research Analyst   │    │
│   │ (Gemini)  │    │ (state +     │◀───│      (Gemini)        │   │
│   └───────────┘    │  budget)     │    └────────────────────┘    │
│                     └──────┬───────┘                              │
│                            │                                      │
│                     ┌──────▼───────┐                              │
│                     │    Tools     │                              │
│                     │──────────────│                              │
│                     │ web_search   │ (ddgs)                       │
│                     │ fetch_webpage│ (requests + bs4)              │
│                     │ extract_     │                              │
│                     │  evidence    │                              │
│                     └──────┬───────┘                              │
│                            │                                      │
│                     ┌──────▼───────┐                              │
│                     │ ResearchState│  question, plan, searches,   │
│                     │  (dataclass) │  sources, evidence, gaps,     │
│                     │              │  final_report, steps          │
│                     └──────────────┘                              │
│                            │                                      │
│                     ┌──────▼───────┐                              │
│                     │ Synthesizer  │───▶ Final Markdown Report     │
│                     │  (Gemini)    │                              │
│                     └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## How the Agent Works

1. **Planning** — one Gemini call turns the user's question into 2–5 focused sub-questions.
2. **Search** — the agent runs `web_search()` (DuckDuckGo via `ddgs`, no paid API) for each sub-question.
3. **Read Sources** — each result URL is fetched with `fetch_webpage()` and cleaned into readable text (scripts/nav/styles stripped, truncated to a character budget).
4. **Collect Evidence** — `extract_evidence()` heuristically condenses each page into a small structured record (`source`, `title`, `key_points`, `relevance`) *without* an LLM call, so no Gemini quota is spent just to shrink HTML.
5. **Identify Gaps** — one Gemini call (the "Research Analyst") inspects all evidence collected so far and returns structured JSON: key findings, conflicts, missing information, and whether more research is needed.
6. **Additional Search (conditional)** — if the analyst says more research is needed *and* there's step/source budget left, the agent runs another search round on the analyst's suggested follow-up queries. Otherwise it moves straight to synthesis.
7. **Synthesize** — one final Gemini call turns all evidence + analyst notes into a structured Markdown report with cited sources.

This is what makes it *agentic* rather than a single prompt: the number of search rounds is not fixed in advance — the agent decides, based on its own evaluation of gaps, whether to keep researching.

## Agent Loop

The loop lives in `research_agent.py: ResearchAgent.run()`:

```
state = ResearchState(question)

plan()                                   # 1 Gemini call
research_round(state.plan)               # tools only, no LLM

while state.steps < MAX_AGENT_STEPS:
    analysis = analyze(state)            # 1 Gemini call
    if not analysis.needs_more_research:
        break
    research_round(analysis.followups)   # tools only, no LLM

synthesize(state)                        # 1 Gemini call
```

`MAX_AGENT_STEPS` (default 5) hard-caps the loop so it can never run away on a free-tier key. A typical run uses **2–4 Gemini calls total**.

## Tools

| Tool | File | Backing library | Cost |
|---|---|---|---|
| `web_search(query)` | `tools.py` | `ddgs` (DuckDuckGo) | free |
| `fetch_webpage(url)` | `tools.py` | `requests` + `beautifulsoup4` | free |
| `extract_evidence(...)` | `tools.py` | pure Python heuristics | free, no LLM |

All three tools **fail soft**: a bad URL, a timeout, an empty search, or a parsing error returns an empty result and logs a warning — it never crashes the run, so one broken source can't derail the whole research task.

## Project Structure

```
01-research-agent/
│
├── README.md              this file
├── research_agent.py       agent state, prompts, agent loop, ResearchAgent
├── tools.py                web_search, fetch_webpage, extract_evidence
├── llm.py                  GeminiLLM wrapper + safe_json_parse
├── config.py                Config dataclass, free-tier limits
├── tests.py                 unit tests (no API key needed) + notes on API tests
├── requirements.txt
├── .env.example
└── data/
    └── sample_queries.txt

notebooks/
└── Agentic_Research_Agent_Colab.ipynb   the runnable Colab demo
```

## Tech Stack

- **LLM**: Google Gemini via the `google-genai` SDK (free tier, no OpenAI key required)
- **Search**: `ddgs` (free DuckDuckGo search wrapper)
- **Extraction**: `requests` + `beautifulsoup4`
- **Agent framework**: none — a lightweight, hand-written state + loop (~4 small files)
- **Runtime**: Google Colab (no Docker, no local server, no Node.js)

## Google Colab Setup

1. Open `notebooks/Agentic_Research_Agent_Colab.ipynb` in Google Colab.
2. Click the **key icon (🔑 Secrets)** in the left sidebar.
3. Add a new secret named exactly `GEMINI_API_KEY` and paste your key as the value. Toggle "Notebook access" on.
4. Run the cells from top to bottom.

The notebook never asks you to paste your key into a cell — it's loaded via:

```python
from google.colab import userdata
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
```

## Gemini API Setup

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Add it to Colab Secrets as `GEMINI_API_KEY` (see above).
3. (Optional) Override the model by setting a `GEMINI_MODEL` env var / Colab Secret. The project defaults to `gemini-3.5-flash` but **Gemini model availability changes over time and by account/region** — if the default isn't available to you, change `GEMINI_MODEL` and nothing else needs to change.

## Running the Agent

Inside the notebook:

```python
agent = build_agent(GEMINI_API_KEY)
state = agent.run("What are the main components of modern AI agent architectures?")
```

Or from the generated project files (e.g. after cloning from GitHub):

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python -c "from research_agent import build_agent; build_agent('your-key').run('your question')"
```

## Example Research Tasks

```
1. What are the main components of modern AI agent architectures?
2. Compare RAG-based systems with long-context LLM approaches.
3. What are the major challenges in deploying autonomous AI agents?
```

More in `data/sample_queries.txt`. The notebook also accepts a custom question.

## Sample Agent Trace

```
==================================================
🤖 AGENTIC RESEARCH AGENT
==================================================

🎯 Goal:
What are the main components of modern AI agent architectures?

🧠 Planning...
✓ Research plan created (4 sub-questions).

🔎 "core components of an AI agent system"
✓ 5 results found
📄 https://example.org/agent-architecture
✓ Evidence collected
...

🧠 Analyzing research gaps...
✓ Additional research required

🔎 "AI agent memory and tool use design patterns"
✓ 4 results found
...

🧠 Synthesizing final report...

==================================================
📊 FINAL RESEARCH REPORT
==================================================
## Executive Summary
...
## Sources
1. https://example.org/agent-architecture
...
--------------------------------------------------
Run summary: steps=4 | llm_calls=3 | search_calls=2 | sources=7
--------------------------------------------------
```

## Free-Tier Optimization

Everything is bounded, and bounds are configurable in `config.py` / via env vars:

| Limit | Default | Purpose |
|---|---|---|
| `MAX_AGENT_STEPS` | 5 | caps total loop iterations |
| `MAX_SEARCH_RESULTS` | 5 | results per search call |
| `MAX_SOURCES` | 8 | total sources kept per run |
| `MAX_PAGE_CHARS` | 12000 | characters kept per fetched page |

A typical run makes **2–4 Gemini calls** (plan, 1–2 analysis rounds, synthesis) and never sends a full raw webpage to Gemini — pages are cleaned, truncated, and reduced to a handful of key points by `extract_evidence()` *before* any LLM sees them. The loop always logs LLM calls, search calls, sources collected, and steps taken at the end of a run.

## Limitations

- Web search results are not guaranteed to be complete or accurate; the agent surfaces what it finds and flags conflicts, but does not fact-check sources against ground truth.
- `extract_evidence()` is a heuristic (keyword-overlap) extractor, not a semantic one — it can occasionally keep a less-relevant paragraph over a more relevant one.
- Free-tier Gemini and free DuckDuckGo search both have rate limits; heavy or rapid re-running may need a short pause.
- The agent reasons over search snippets and page text only — it does not verify claims against multiple independent primary sources unless the search results themselves surface them.

## Responsible AI

- The agent **never fabricates sources** — the final report is instructed to cite only URLs that were actually collected during the run.
- Evidence is kept explicitly separate from the model's own inference; the synthesis prompt is constrained to the evidence provided.
- Conflicting information found across sources is surfaced rather than silently resolved.
- Uncertainty and research limitations are always included in the final report, not just on request.

## Future Improvements

- Add a lightweight re-ranking step for sources before extraction.
- Support parallel (rate-limited) page fetches for faster runs.
- Add a semantic (embedding-based) evidence extractor as an optional upgrade path.
- Persist run history to disk for comparison across research sessions.
- Reuse this `llm.py` / tool-system / agent-loop pattern for Projects 2–4 (Coding Agent, Sales Agent, Recruiting Agent).

## Author

Built as Project 1 of a 4-project Agentic AI portfolio, demonstrating a hand-built agent loop (planning, tool use, state, gap detection, synthesis) without relying on an agent framework.
