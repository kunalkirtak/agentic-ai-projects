# 🤖 Agentic AI Projects

**A 4-project portfolio of autonomous AI agents built from scratch in Python — no LangChain, no CrewAI, no AutoGen.**

Each project implements a real **agentic loop** (plan → act → observe → correct/repeat → report) with explicit state, bounded iterations, and tool use, running entirely on the **free-tier Gemini API** inside Google Colab. No paid APIs, no Docker, no local server required to try any of them.

| # | Project | What it does |
|---|---|---|
| 01 | [Research Agent](./01-research-agent) | Plans sub-questions, searches the web, reads sources, detects its own information gaps, re-searches if needed, and synthesizes a cited Markdown report. |
| 02 | [Coding Agent](./02-coding-agent) | Given a natural-language task, plans an implementation, writes code + unit tests, executes them, diagnoses failures, self-corrects, and re-tests in a bounded loop. |
| 03 | [Sales Agent](./03-sales-agent) | Given an Ideal Customer Profile (ICP), researches the public web for candidate companies, scores them against explicit criteria, and produces a ranked, evidence-backed lead report. |
| 04 | [Recruiting Agent](./04-recruiting-agent) | Given a job description and a folder of resumes, extracts requirements, parses resumes, matches candidates against evidence (never "does not have"), and produces an explainable, ranked shortlist. |

---

## Why this portfolio is different

Most "AI agent" demos are a single LLM call:

```
Input → LLM → Output
```

That's a prompt, not an agent. Every project here implements an actual control loop with **state that persists across steps**, a **stopping condition the agent evaluates itself**, and **tools the agent decides when to call** — all as plain, readable Python, so the mechanics of "agentic" behavior are fully visible instead of hidden inside a framework.

```
Plan → Act (call tools) → Observe (real output) → Self-correct / gap-check → Repeat → Final report
```

## Design principles across all four agents

- **No agent framework.** The planning loop, tool dispatch, and state machine are hand-written (~200–400 lines per project), so anyone reading the code can trace exactly what happens on every iteration.
- **Bounded execution.** Every loop has a hard step/iteration cap (`MAX_AGENT_STEPS`, `MAX_ITERATIONS`, etc.) so a free-tier API key can never be exhausted by a runaway agent.
- **Evidence over inference.** Agents are constrained to cite only sources/evidence they actually retrieved — conflicting or missing information is surfaced explicitly rather than guessed at or silently smoothed over.
- **Deterministic scoring where it matters.** In the Sales and Recruiting agents, final scores are computed in plain Python (`scoring.py`), not left for the LLM to total up — the LLM extracts structured evidence, code does the arithmetic.
- **Fail-soft tools.** Web fetches, searches, and parsers catch their own errors and return empty results with a logged warning instead of crashing the run.
- **Runs on free tools.** Google Gemini free tier + free DuckDuckGo search (`ddgs`) + `requests`/`beautifulsoup4` — zero paid API dependencies.

## Tech stack

| Layer | Tools |
|---|---|
| LLM | Google Gemini (`google-genai` SDK, free tier) |
| Search | `ddgs` (DuckDuckGo, free) |
| Web extraction | `requests`, `beautifulsoup4` |
| Resume/document parsing | `pypdf` |
| Language | Python 3, standard library (`dataclasses`, `subprocess`, `unittest`, `enum`, `json`) |
| Runtime | Google Colab (each project ships a runnable notebook) |
| Testing | `unittest`-based tests requiring no API key |

## Repository structure

```
agentic-ai-projects/
├── 01-research-agent/      research_agent.py, tools.py, llm.py, config.py, tests.py, notebooks/
├── 02-coding-agent/        coding_agent.py, tools.py, llm.py, config.py, tests/, notebooks/
├── 03-sales-agent/         sales_agent.py, scoring.py, tools.py, llm.py, config.py, tests/, notebook/
├── 04-recruiting-agent/    recruiting_agent.py, scoring.py, parser.py, tools.py, llm.py, config.py, tests/, notebooks/
└── LICENSE                 MIT
```

Every project is self-contained: its own `requirements.txt`, `config.py`, tests, and a Colab notebook, plus its own detailed README covering architecture, agent loop, tools, setup, and limitations.

## Getting started

Each agent runs standalone in Google Colab:

1. Open the project's notebook (`notebooks/*.ipynb`) in Google Colab.
2. Add a free [Google AI Studio](https://aistudio.google.com/apikey) API key as a Colab Secret named `GEMINI_API_KEY`.
3. Run all cells.

Or locally:

```bash
cd 01-research-agent   # or 02-, 03-, 04-
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
python -c "from research_agent import build_agent; build_agent('your-key').run('your question')"
```

See each project's own README for exact usage, sample traces, and configuration limits.

## Roadmap

- [ ] Shared `llm.py` / tool-system pattern extracted into a small internal library
- [ ] Parallel, rate-limited web fetches for faster research/sales runs
- [ ] Optional semantic (embedding-based) evidence extraction
- [ ] Persisted run history for comparing agent runs over time

## License

[MIT](./LICENSE) © Kunal B. Kirtak

---

*Built to demonstrate hand-rolled agentic loops — planning, tool use, state management, self-correction, and evidence-grounded synthesis — without relying on an agent framework.*
