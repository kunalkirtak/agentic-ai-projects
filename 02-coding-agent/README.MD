# 🤖 Agentic Coding Agent

**Project 2 of 4 — Agentic AI Portfolio**

```
01 — Research Agent
02 — Coding Agent        ← this project
03 — Sales Agent
04 — Recruiting Agent
```

Built and runnable **100% inside Google Colab**, using the **Gemini
free-tier API** — no OpenAI, no Docker, no LangChain, no paid services.

---

## Overview

This project is an autonomous coding agent. Given a natural-language
programming task, it plans an implementation, writes the code, writes
tests, runs everything locally, diagnoses failures, fixes the code,
and re-tests — all inside a bounded loop with an explicit state
object — before reporting a final result.

## Why This Is Agentic

```
LLM ≠ Agent
```

A single call to an LLM that returns code is not an agent — it's a
text completion. This project is agentic because it:

```
Plan → Act → Observe → Correct → Repeat
```

Concretely: the agent **plans** an approach, **acts** by generating
and executing code, **observes** the real stdout/stderr/test results
from that execution, **corrects** the code based on those
observations, and **repeats** until the tests pass or a hard
iteration limit is reached. The control flow, the state, and the
stopping condition are all explicit Python — not hidden inside a
prompt.

## Architecture

```
                  USER TASK
                      │
                      ▼
             ┌─────────────────┐
             │ REQUIREMENT     │
             │ ANALYZER        │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │    PLANNER      │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ CODE GENERATOR  │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ TEST GENERATOR  │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ CODE EXECUTOR   │
             └────────┬────────┘
                      ▼
                 TEST RESULT
                 /        \
              PASS        FAIL
               │            │
               │            ▼
               │    ┌───────────────┐
               │    │ FAILURE       │
               │    │ ANALYZER      │
               │    └───────┬───────┘
               │            ▼
               │    ┌───────────────┐
               │    │ CODE FIXER    │
               │    └───────┬───────┘
               │            │
               │            └──────► RETEST
               │
               ▼
        FINAL IMPLEMENTATION
```

Code layout mirrors this:

| Concern                        | File               |
|---------------------------------|--------------------|
| Configuration / tunables        | `config.py`        |
| Gemini client wrapper           | `llm.py`           |
| Tools (generate/run/fix)        | `tools.py`         |
| Agent state, actions, the loop  | `coding_agent.py`  |
| Local, no-API unit tests        | `tests/test_tools.py` |

## Agent Loop

```python
while not state.success and state.iteration < MAX_ITERATIONS:
    action = decide_next_action(state)   # reasoning — no side effects
    execute_action(action, state)        # tool execution — no reasoning
```

`_decide_next_action` is pure decision logic over the current
`CodingState` — it never calls Gemini or runs code itself. Every
action it can return maps to exactly one handler in
`_execute_action`, which is the only place tools are invoked. This
split is what keeps "what to do next" separate from "how it's done."

`MAX_ITERATIONS = 4` is a hard ceiling — the loop can never spin
forever, which matters a lot on a free-tier quota.

## Tools

| Tool               | Job                                                            |
|---------------------|----------------------------------------------------------------|
| `CodeGeneratorTool` | Task + requirements + plan → Python source                     |
| `TestGeneratorTool` | Code → `unittest`-based tests                                  |
| `CodeExecutor`      | Runs Python source in a temp dir with a timeout, captures I/O  |
| `TestExecutor`      | Runs the tests the same way, parses pass/fail counts           |
| `CodeFixerTool`     | Task + code + tests + failure output → diagnosis + fixed code  |

Each tool does one job and knows nothing about the agent loop —
they're called, not consulted.

## State Management

```python
@dataclass
class CodingState:
    task: str
    requirements: list
    plan: list
    code: str
    tests: str
    execution_results: list
    errors: list
    fixes: list
    iteration: int
    success: bool
    final_output: str
    gemini_calls_used: int
```

The same `CodingState` instance is threaded through every iteration
of the loop and mutated in place — this is what lets the agent
"remember" the plan, the current code, and prior failures instead of
starting from scratch each time.

## Code Generation

`CodeGeneratorTool` asks Gemini for a single fenced ` ```python ` block
implementing the task per the plan, with type hints, docstrings, and
basic error handling. Output is parsed with `strip_markdown_fence`,
which robustly handles fenced, unfenced, and mixed responses.

## Testing

`TestGeneratorTool` produces `unittest`-based tests (chosen over
pytest to avoid an extra dependency in Colab). `TestExecutor` runs
them with `python -m unittest ... -v` in an isolated temp directory
and parses a `passed / failed / errors` summary from the output.

## Self-Correction

When tests fail, `CodeFixerTool.diagnose_and_fix` sends the task,
current code, tests, and captured failure output to Gemini, gets back
a short JSON diagnosis (`problem`, `cause`, `fix_strategy`), and then
requests a corrected full module. The fix becomes the new `state.code`
and the loop re-runs the tests.

## Security Considerations

**This is an educational coding-agent prototype — not a hardened
sandbox.** Generated code is executed locally with:

- a fresh `tempfile.TemporaryDirectory()` per run
- a `subprocess.run(..., timeout=EXECUTION_TIMEOUT, capture_output=True)`
- no shell execution — only `python <file>` and `python -m unittest`
  are ever invoked, never arbitrary LLM-generated shell commands
- automatic cleanup when the temp dir context exits

**Never run this against untrusted tasks or in a multi-tenant
environment.** A production system would additionally need:

- container isolation
- non-root execution
- CPU/memory limits
- filesystem isolation
- network restrictions
- syscall restrictions (e.g. seccomp)
- stronger sandboxing (e.g. gVisor, Firecracker)
- audit logging

## Project Structure

```
02-coding-agent/
│
├── README.md
├── coding_agent.py
├── tools.py
├── llm.py
├── config.py
├── requirements.txt
├── .env.example
├── data/
│   └── sample_tasks.txt
├── screenshot/             # output pic after running file 
├── examples/
│   └── expense_tracker_task.txt
├── notebooks/
│   └── Agentic_Coding_Agent_Colab.ipynb
└── tests/
    └── test_tools.py


```

## Tech Stack

- Python 3 (standard library: `subprocess`, `tempfile`, `unittest`, `dataclasses`, `enum`, `json`)
- `google-genai` (Gemini free-tier API)
- Google Colab (Secrets, form fields, file generation, ZIP download)

No LangChain, LangGraph, CrewAI, AutoGen, Docker, Kubernetes,
databases, Redis, cloud deployment, or frontend framework is used —
the agent architecture is implemented from first principles in plain
Python so it's fully transparent.

## Google Colab Setup

1. Open `notebooks/Agentic_Coding_Agent_Colab.ipynb` in Google Colab.
2. Run the cells top to bottom.
3. The notebook generates every project file (`config.py`, `llm.py`,
   `tools.py`, `coding_agent.py`, tests, README, etc.) directly onto
   the Colab filesystem via `Path(...).write_text(...)` cells.

## Gemini API Setup

1. Get a free-tier API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. In Colab, click the 🔑 **Secrets** icon in the left sidebar.
3. Add a new secret named `GEMINI_API_KEY` with your key as the value.
4. Toggle **Notebook access** on for this secret.
5. The notebook loads it with:
   ```python
   from google.colab import userdata
   GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
   ```
   The key is never printed or hard-coded anywhere in the project.

## Running the Demo

The notebook includes three demo tasks (easy / medium / advanced) plus
a free-text cell for a fully custom task:

```python
agent = CodingAgent(api_key=GEMINI_API_KEY)
state = agent.run("Build a Python CLI expense tracker...")
print_final_report(state)
```

## Example Output

```
========================================================
🤖 AGENTIC CODING AGENT
========================================================

🎯 Task:
Build a Python CLI expense tracker.

🧠 Step — Understanding requirements & creating plan
✓ 5 requirements identified
✓ Plan created (5 steps)

💻 Step — Generating code
✓ Code generated

🧪 Step — Generating tests
✓ Tests generated

▶ Step — Running tests
✗ Tests failed (passed=3 failed=2 errors=0 (exit code 1))

🔍 Step — Diagnosing failure
✓ Root cause identified: JSON file not initialized before first read

🔧 Step — Fixing code
✓ Updated implementation

▶ Step — Running tests
✓ All tests passed (passed=5 failed=0 errors=0 (exit code 0))

========================================================
✅ CODING TASK COMPLETED
========================================================
```

## Free-Tier Optimization

Gemini calls are deliberately batched to stay within a **3–5 live
calls per task** budget:

| Call | Purpose                                     |
|------|----------------------------------------------|
| 1    | Requirements extraction + planning            |
| 2    | Code generation                               |
| 3    | Test generation                               |
| 4–5  | Diagnose (1 call) + fix (1 call) — only if tests fail |

Test execution, code execution, and re-testing all happen **locally**
with zero additional API calls. `MAX_ITERATIONS = 6` gives the agent
exactly one diagnose-and-fix cycle when tests fail; `EXECUTION_TIMEOUT
= 10` (seconds) bounds wall-clock time for running generated code.

On top of that call budget, `llm.py` layers three protections so a
small free-tier key still produces a clean, complete notebook run:

- **Disk cache** (`.gemini_cache/`) — identical `(model, prompt,
  temperature, max_tokens)` requests are served from disk. Re-running
  a cell while polishing this notebook costs zero quota.
- **Pacing** (`MIN_SECONDS_BETWEEN_CALLS`, default 6s) — a minimum gap
  is enforced between live calls so the notebook can't burst past a
  requests-per-minute limit.
- **Session budget** (`GEMINI_SESSION_CALL_BUDGET`, default 20) — a
  hard cap on live calls per notebook run. Once reached, the agent
  stops itself *gracefully* — printing partial results and a clear
  "paused" status — instead of crashing on a raw `429` mid-demo.

All three are tunable via environment variables in `config.py` if your
key has a different (or more generous) quota.

## Limitations

- Python-only; no multi-language support in this version.
- `subprocess` execution is a controlled demo, not a real sandbox.
- Test-result parsing is simple string parsing over `unittest -v`
  output, not a structured test report.
- Only the most recent failure is used for diagnosis, not full history.
- No persistence between runs — each `agent.run(task)` call starts a
  fresh `CodingState`.

## Future Improvements

- Add a lightweight structured test report (e.g. `unittest`'s
  `TestResult` object) instead of parsing stderr text.
- Support additional languages behind the same tool interface.
- Add a proper sandbox (container or microVM) for real isolation.
- Persist state across runs to resume interrupted tasks.
- Add a retrieval step so the agent can reuse patterns from prior
  successful tasks in this same project.

## Author

Built by Kunal as Project 2 of a 4-part Agentic AI portfolio,
demonstrating agent architecture, planning, tool use, code generation,
automated testing, error analysis, self-correction, and state
management — implemented from first principles in plain Python.
