"""
tools.py
========
The Coding Agent's tool system.

Each tool is a plain Python callable/class with a single, well-defined
job. Tools are deliberately "dumb" — they don't decide *when* to run,
they just do one thing well when the agent (coding_agent.py) invokes
them. This separation is what makes the "reasoning vs. execution"
distinction visible in the architecture:

    reasoning/decision  -> coding_agent.py (the loop)
    tool execution      -> tools.py (this file)

SECURITY NOTE
-------------
`CodeExecutor` and `TestExecutor` run LLM-generated Python source with
`subprocess.run(...)` inside a temporary directory, with a timeout and
captured output. This is a *controlled educational demonstration only*
— it is NOT a hardened security sandbox. See the README's "Security
Considerations" section before using this with untrusted input.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm import GeminiLLM, strip_markdown_fence
from config import EXECUTION_TIMEOUT, GEMINI_TEMPERATURE_CODEGEN, GEMINI_TEMPERATURE_FIX


# ===========================================================================
# Tool 1 — Code Generator
# ===========================================================================
class CodeGeneratorTool:
    """Generates a Python implementation from task + requirements + plan."""

    def __init__(self, llm: GeminiLLM) -> None:
        self.llm = llm

    def run(self, task: str, requirements: list[str], plan: list[str]) -> str:
        prompt = f"""You are an expert Python engineer. Write a complete,
runnable Python module that implements the following task.

TASK:
{task}

REQUIREMENTS:
{_bullet(requirements)}

IMPLEMENTATION PLAN:
{_bullet(plan)}

Rules:
- Return ONLY Python source code, no explanations.
- Wrap the code in a single ```python fenced block.
- Use functions and/or classes as appropriate, with type hints and
  docstrings.
- Include basic error handling where it matters.
- The module must be self-contained (standard library only, unless the
  task clearly requires a specific well-known package).
- If the task implies a CLI, guard the entry point with
  `if __name__ == "__main__":`.
"""
        raw = self.llm.generate(prompt, temperature=GEMINI_TEMPERATURE_CODEGEN)
        code = strip_markdown_fence(raw)
        if not code.strip():
            raise ValueError("Code generation returned empty output.")
        return code


# ===========================================================================
# Tool 2 — Test Generator
# ===========================================================================
class TestGeneratorTool:
    """Generates `unittest`-based tests for a given code module."""

    def __init__(self, llm: GeminiLLM) -> None:
        self.llm = llm

    def run(self, task: str, code: str) -> str:
        prompt = f"""You are an expert Python test engineer. Write
`unittest`-based tests for the module below.

TASK CONTEXT:
{task}

MODULE UNDER TEST (this exact source will be saved as `solution.py`,
so import from it with `from solution import ...`):
```python
{code}
```

Rules:
- Return ONLY Python source code, no explanations.
- Wrap the code in a single ```python fenced block.
- Use the standard library `unittest` module only.
- Import the module under test with `from solution import ...`.
- Cover normal cases, edge cases, and at least one failure/invalid case.
- End the file with:
  if __name__ == "__main__":
      unittest.main()
"""
        raw = self.llm.generate(prompt, temperature=GEMINI_TEMPERATURE_CODEGEN)
        tests = strip_markdown_fence(raw)
        if not tests.strip():
            raise ValueError("Test generation returned empty output.")
        return tests


# ===========================================================================
# Shared execution result type
# ===========================================================================
@dataclass
class ExecutionResult:
    success: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str = ""

    def summary(self) -> str:
        if self.timed_out:
            return f"TIMEOUT after {EXECUTION_TIMEOUT}s"
        if self.error:
            return f"ERROR: {self.error}"
        return f"exit code {self.returncode}"


# ===========================================================================
# Tool 3 — Code Executor
# ===========================================================================
class CodeExecutor:
    """
    Runs a Python source string in an isolated temporary directory.

    This is a *controlled educational demonstration*, not a security
    sandbox. It only ever invokes the current Python interpreter on a
    file we wrote ourselves (`python solution.py`) — it never executes
    arbitrary shell commands produced by the LLM.
    """

    def __init__(self, timeout: int = EXECUTION_TIMEOUT) -> None:
        self.timeout = timeout

    def run(self, code: str, filename: str = "solution.py") -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="coding_agent_") as tmpdir:
            file_path = Path(tmpdir) / filename
            file_path.write_text(code, encoding="utf-8")
            return self._execute([sys.executable, str(file_path)], cwd=tmpdir)

    def _execute(self, cmd: list[str], cwd: str) -> ExecutionResult:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                success=False,
                returncode=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                success=False,
                returncode=None,
                stdout="",
                stderr="",
                error=str(exc),
            )
        # Temporary directory and all files inside it are removed
        # automatically when the `with` block exits.


# ===========================================================================
# Tool 4 — Test Executor
# ===========================================================================
class TestExecutor:
    """
    Runs `unittest` tests against a code module in an isolated temp dir.

    Writes `solution.py` and `test_solution.py` side by side, then runs
    `python -m unittest test_solution.py -v` and parses a pass/fail
    summary out of the captured output.
    """

    def __init__(self, timeout: int = EXECUTION_TIMEOUT) -> None:
        self.timeout = timeout
        self._executor = CodeExecutor(timeout=timeout)

    def run(self, code: str, tests: str) -> "TestRunResult":
        with tempfile.TemporaryDirectory(prefix="coding_agent_tests_") as tmpdir:
            (Path(tmpdir) / "solution.py").write_text(code, encoding="utf-8")
            (Path(tmpdir) / "test_solution.py").write_text(tests, encoding="utf-8")

            exec_result = self._executor._execute(
                [sys.executable, "-m", "unittest", "test_solution.py", "-v"],
                cwd=tmpdir,
            )
            passed, failed, errors = _parse_unittest_output(exec_result.stdout, exec_result.stderr)
            return TestRunResult(
                execution=exec_result,
                tests_passed=passed,
                tests_failed=failed,
                tests_errored=errors,
            )


@dataclass
class TestRunResult:
    execution: ExecutionResult
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errored: int = 0

    @property
    def all_passed(self) -> bool:
        return (
            self.execution.success
            and self.tests_failed == 0
            and self.tests_errored == 0
        )

    def summary(self) -> str:
        return (
            f"passed={self.tests_passed} failed={self.tests_failed} "
            f"errors={self.tests_errored} ({self.execution.summary()})"
        )


def _parse_unittest_output(stdout_text: str, stderr_text: str) -> tuple[int, int, int]:
    """
    `unittest -v` normally writes its progress/summary lines to stderr,
    but a module-level failure during test *collection* (e.g. a
    SyntaxError, ImportError, or ModuleNotFoundError in solution.py or
    test_solution.py) can surface as a plain traceback that is not
    formatted as a per-test "... ok/FAIL/ERROR" line, and depending on
    how it is raised may show up on stdout rather than stderr. Scan
    both streams so collection-time failures are always counted as an
    error instead of silently reporting zero failures. This is
    intentionally simple string parsing, not a full test-report parser.
    """
    passed = failed = errored = 0
    for raw_line in stderr_text.splitlines() + stdout_text.splitlines():
        line = raw_line.strip()
        if line.endswith("... ok"):
            passed += 1
        elif line.endswith("... FAIL"):
            failed += 1
        elif line.endswith("... ERROR"):
            errored += 1
        elif any(
            marker in line
            for marker in ("SyntaxError", "ImportError", "ModuleNotFoundError")
        ):
            errored += 1
    return passed, failed, errored


# ===========================================================================
# Tool 5 — Code Analyzer / Fixer
# ===========================================================================
class CodeFixerTool:
    """
    Sends failure details to Gemini and asks for a corrected full module.
    Used for both "code raised an error" and "tests failed" cases.
    """

    def __init__(self, llm: GeminiLLM) -> None:
        self.llm = llm

    def diagnose_and_fix(
        self,
        task: str,
        code: str,
        tests: str,
        error_output: str,
    ) -> tuple[dict, str]:
        """
        Returns (diagnosis_dict, fixed_code_str).

        A single Gemini call is used to both diagnose the failure and
        produce the fix, to conserve free-tier quota. The diagnosis is
        requested as a JSON preamble embedded in a larger response, but
        to keep parsing simple and robust we issue two lightweight
        prompts sharing the same context instead of one fragile combined
        format — this still counts as ONE logical "fix step" in the
        agent's call budget documentation, but uses `generate_json` +
        `generate` under the hood. See README's Free-Tier Optimization
        section for how call budget is tracked.
        """
        diagnosis_prompt = f"""A Python solution failed. Diagnose the
root cause briefly.

TASK:
{task}

CODE:
```python
{code}
```

TESTS:
```python
{tests}
```

FAILURE / ERROR OUTPUT:
```
{error_output}
```

Return a JSON object with exactly these keys:
{{"problem": "...", "cause": "...", "fix_strategy": "..."}}
"""
        diagnosis = self.llm.generate_json(diagnosis_prompt)

        fix_prompt = f"""Fix the Python module below so it satisfies the
task and passes the tests. Apply this fix strategy: {diagnosis.get('fix_strategy', 'N/A')}

TASK:
{task}

CURRENT CODE:
```python
{code}
```

TESTS IT MUST PASS:
```python
{tests}
```

FAILURE / ERROR OUTPUT:
```
{error_output}
```

Return ONLY the complete corrected Python module in a single ```python
fenced block. Do not include explanations.
"""
        raw = self.llm.generate(fix_prompt, temperature=GEMINI_TEMPERATURE_FIX)
        fixed_code = strip_markdown_fence(raw)
        if not fixed_code.strip():
            raise ValueError("Code fixer returned empty output.")
        return diagnosis, fixed_code


# ===========================================================================
# Small shared helpers
# ===========================================================================
def _bullet(items: list[str]) -> str:
    if not items:
        return "- (none specified)"
    return "\n".join(f"- {item}" for item in items)


# ===========================================================================
# Tool registry (used by the agent + exercised by unit tests)
# ===========================================================================
def build_tool_registry(llm: GeminiLLM) -> dict:
    """
    Constructs and returns all tools keyed by name. Keeping this in one
    place makes it easy to see (and test) exactly what tools the agent
    has available.
    """
    return {
        "code_generator": CodeGeneratorTool(llm),
        "test_generator": TestGeneratorTool(llm),
        "code_executor": CodeExecutor(),
        "test_executor": TestExecutor(),
        "code_fixer": CodeFixerTool(llm),
    }
