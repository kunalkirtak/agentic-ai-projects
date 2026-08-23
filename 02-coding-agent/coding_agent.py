"""
coding_agent.py
================
The agent itself: explicit state, a small set of named actions, and a
bounded loop that decides what to do next based on the current state.

    reasoning/decision   -> CodingAgent._decide_next_action()
    tool execution       -> CodingAgent._execute_action() (delegates to tools.py)

This file has NO knowledge of Colab, files, or ZIPs — it's pure agent
logic so it can be unit-tested and reused anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from config import MAX_ITERATIONS, GEMINI_TEMPERATURE_PLANNING
from llm import GeminiLLM, GeminiError, GeminiQuotaError
from tools import build_tool_registry, TestRunResult


# ===========================================================================
# Actions
# ===========================================================================
class Action(str, Enum):
    PLAN = "PLAN"
    GENERATE_CODE = "GENERATE_CODE"
    GENERATE_TESTS = "GENERATE_TESTS"
    RUN_TESTS = "RUN_TESTS"
    ANALYZE_FAILURE = "ANALYZE_FAILURE"
    FIX_CODE = "FIX_CODE"
    FINISH = "FINISH"


# ===========================================================================
# Agent state
# ===========================================================================
@dataclass
class CodingState:
    """Everything the agent knows about the current task, at any point
    in the loop. This object is mutated in place as the loop progresses
    and is what makes the agent's behavior "agentic" rather than a
    single stateless prompt -> response call."""

    task: str
    requirements: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    code: str = ""
    tests: str = ""
    execution_results: list = field(default_factory=list)  # list[dict]
    errors: list = field(default_factory=list)              # list[str]
    fixes: list = field(default_factory=list)                # list[dict]
    iteration: int = 0
    success: bool = False
    final_output: str = ""
    last_test_result: Optional[TestRunResult] = field(default=None, repr=False)
    gemini_calls_used: int = 0
    stopped_reason: str = ""  # set if the loop had to stop early (quota/API)

    def log_execution(self, label: str, summary: str) -> None:
        self.execution_results.append({"step": label, "summary": summary})


# ===========================================================================
# The Agent
# ===========================================================================
class CodingAgent:
    """
    Orchestrates the agentic coding loop:

        while not success and iteration < MAX_ITERATIONS:
            analyze_state()
            action = decide_next_action()
            execute_action(action)
            record_observation()
            update_state()

    If a Gemini call fails permanently (including hitting the local
    free-tier session budget), the loop stops *gracefully* — whatever
    partial progress exists is still returned and reported cleanly,
    instead of raising and leaving a raw traceback in the notebook.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
    ) -> None:
        self.llm = GeminiLLM(api_key=api_key, model=model) if model else GeminiLLM(api_key=api_key)
        self.tools = build_tool_registry(self.llm)
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, task: str) -> CodingState:
        state = CodingState(task=task)
        self._log("=" * 56)
        self._log("🤖 AGENTIC CODING AGENT")
        self._log("=" * 56)
        self._log(f"\n🎯 Task:\n{task}\n")

        try:
            while not state.success and state.iteration < self.max_iterations:
                state.iteration += 1
                self._log(f"\n--- Iteration {state.iteration}/{self.max_iterations} ---")

                action = self._decide_next_action(state)
                self._execute_action(action, state)

                # Register success the moment tests actually pass, even if
                # this was the last allowed iteration — otherwise a task
                # that truly succeeded on its final try would incorrectly
                # be reported as "NEEDS REVIEW" simply because the loop
                # never got a further turn to reach Action.FINISH.
                if state.last_test_result is not None and state.last_test_result.all_passed:
                    state.success = True

                if action == Action.FINISH:
                    break
        except GeminiQuotaError as exc:
            state.stopped_reason = f"Session call budget reached: {exc}"
            self._log(f"\n⏸ Pausing — {state.stopped_reason}")
        except GeminiError as exc:
            state.stopped_reason = f"Gemini call failed: {exc}"
            self._log(f"\n⏸ Pausing — {state.stopped_reason}")

        self._finalize(state)
        return state

    # ------------------------------------------------------------------
    # Reasoning / decision layer
    # ------------------------------------------------------------------
    def _decide_next_action(self, state: CodingState) -> Action:
        """Pure decision logic based on current state — no tool calls
        happen here. This is what keeps 'deciding what to do' separate
        from 'doing it'."""
        if not state.plan:
            return Action.PLAN
        if not state.code:
            return Action.GENERATE_CODE
        if not state.tests:
            return Action.GENERATE_TESTS
        if state.last_test_result is None:
            return Action.RUN_TESTS
        if state.last_test_result.all_passed:
            return Action.FINISH
        if state.iteration >= self.max_iterations:
            return Action.FINISH
        # Tests exist and failed -> diagnose + fix, then re-test
        if not state.fixes or len(state.fixes) < state.iteration - 1:
            return Action.ANALYZE_FAILURE
        return Action.RUN_TESTS

    # ------------------------------------------------------------------
    # Execution layer — delegates to tools.py, never reasons
    # ------------------------------------------------------------------
    def _execute_action(self, action: Action, state: CodingState) -> None:
        handler = {
            Action.PLAN: self._act_plan,
            Action.GENERATE_CODE: self._act_generate_code,
            Action.GENERATE_TESTS: self._act_generate_tests,
            Action.RUN_TESTS: self._act_run_tests,
            Action.ANALYZE_FAILURE: self._act_analyze_and_fix,
            Action.FINISH: self._act_finish,
        }[action]
        handler(state)

    # -- Step 1: Understand requirements + plan (1 Gemini call) --------
    def _act_plan(self, state: CodingState) -> None:
        self._log("\n🧠 Step — Understanding requirements & creating plan")
        prompt = f"""Analyze this coding task and produce a JSON object
with the requirements and an implementation plan.

TASK:
{state.task}

Return JSON with exactly these keys:
{{
  "requirements": ["...", "..."],
  "plan": ["...", "..."]
}}
"""
        result = self.llm.generate_json(prompt, temperature=GEMINI_TEMPERATURE_PLANNING)
        state.gemini_calls_used += 1
        state.requirements = result.get("requirements", [])
        state.plan = result.get("plan", [])
        self._log(f"✓ {len(state.requirements)} requirements identified")
        self._log(f"✓ Plan created ({len(state.plan)} steps)")
        state.log_execution("plan", f"{len(state.requirements)} requirements, {len(state.plan)} plan steps")

    # -- Step 2: Generate code (1 Gemini call) --------------------------
    def _act_generate_code(self, state: CodingState) -> None:
        self._log("\n💻 Step — Generating code")
        state.code = self.tools["code_generator"].run(
            task=state.task, requirements=state.requirements, plan=state.plan
        )
        state.gemini_calls_used += 1
        self._log("✓ Code generated")
        state.log_execution("generate_code", f"{len(state.code.splitlines())} lines generated")

    # -- Step 3: Generate tests (part of the code-gen call budget) -----
    def _act_generate_tests(self, state: CodingState) -> None:
        self._log("\n🧪 Step — Generating tests")
        state.tests = self.tools["test_generator"].run(task=state.task, code=state.code)
        state.gemini_calls_used += 1
        self._log("✓ Tests generated")
        state.log_execution("generate_tests", f"{len(state.tests.splitlines())} lines generated")

    # -- Step 4/8: Run tests (local, no Gemini call) --------------------
    def _act_run_tests(self, state: CodingState) -> None:
        self._log("\n▶ Step — Running tests")
        result = self.tools["test_executor"].run(state.code, state.tests)
        state.last_test_result = result
        if result.all_passed:
            self._log(f"✓ All tests passed ({result.summary()})")
        else:
            self._log(f"✗ Tests failed ({result.summary()})")
            combined_output = (result.execution.stdout + "\n" + result.execution.stderr).strip()
            state.errors.append(combined_output or result.execution.summary())
        state.log_execution("run_tests", result.summary())

    # -- Step 6/7: Diagnose + fix (1 Gemini call) ------------------------
    def _act_analyze_and_fix(self, state: CodingState) -> None:
        self._log("\n🔍 Step — Diagnosing failure")
        error_output = state.errors[-1] if state.errors else "(no captured output)"
        diagnosis, fixed_code = self.tools["code_fixer"].diagnose_and_fix(
            task=state.task, code=state.code, tests=state.tests, error_output=error_output
        )
        state.gemini_calls_used += 1
        self._log(f"✓ Root cause identified: {diagnosis.get('problem', 'unknown')}")
        state.fixes.append(diagnosis)

        self._log("\n🔧 Step — Fixing code")
        state.code = fixed_code
        state.last_test_result = None  # force a re-test on the next iteration
        self._log("✓ Updated implementation")
        state.log_execution("fix_code", diagnosis.get("fix_strategy", "applied fix"))

    # -- Finalization ------------------------------------------------------
    def _act_finish(self, state: CodingState) -> None:
        if state.last_test_result is not None and state.last_test_result.all_passed:
            state.success = True

    def _finalize(self, state: CodingState) -> None:
        self._log("\n" + "=" * 56)
        if state.stopped_reason:
            status = "PAUSED — QUOTA/RATE LIMIT"
            self._log("⏸ STOPPED EARLY TO PROTECT YOUR FREE-TIER QUOTA")
            self._log(f"   {state.stopped_reason}")
            self._log("   Whatever code/tests were produced before this point are")
            self._log("   shown below. Re-run later (or raise the session budget /")
            self._log("   MIN_SECONDS_BETWEEN_CALLS in config.py) once quota resets.")
        elif state.success:
            status = "SUCCESS"
            self._log("✅ CODING TASK COMPLETED")
        else:
            status = "NEEDS REVIEW"
            self._log("⚠ Maximum iterations reached.")
            self._log("The generated solution still requires review.")
        self._log("=" * 56)

        state.final_output = state.code
        budget = self.llm.session_call_budget
        used = GeminiLLM.session_calls_used()
        self._log(f"\n📊 Live Gemini calls this session: {used}/{budget}")
        state.log_execution("finish", f"status={status}, gemini_calls={state.gemini_calls_used}")

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


# ===========================================================================
# Convenience: save final code to disk
# ===========================================================================
def save_final_code(state: CodingState, output_path: str = "final_solution.py") -> Path:
    path = Path(output_path)
    path.write_text(state.final_output or state.code, encoding="utf-8")
    return path


def print_final_report(state: CodingState) -> None:
    if state.stopped_reason:
        status = "PAUSED — QUOTA/RATE LIMIT"
    elif state.success:
        status = "SUCCESS"
    else:
        status = "NEEDS REVIEW"
    result = state.last_test_result
    print("\n" + "=" * 56)
    print("FINAL RESULT")
    print("=" * 56)
    print(f"\nTask:\n{state.task}\n")
    print(f"Iterations: {state.iteration}")
    print(f"Gemini calls used (this task): {state.gemini_calls_used}")
    print(f"Gemini calls used (this session): {GeminiLLM.session_calls_used()}")
    if result:
        print(f"Tests passed: {result.tests_passed}")
        print(f"Tests failed: {result.tests_failed + result.tests_errored}")
    print(f"\nStatus: {status}")
    if state.stopped_reason:
        print(f"Reason: {state.stopped_reason}")
    if state.code:
        print("\n--- Final Code (partial or complete, see status above) ---\n")
        print(state.final_output or state.code)
    else:
        print("\n(No code was generated before stopping.)")
