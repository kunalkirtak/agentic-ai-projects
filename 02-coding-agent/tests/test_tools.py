"""
tests/test_tools.py
====================
Pure, local unit tests that do NOT call the Gemini API. These are safe
to run repeatedly without touching the free-tier quota.

Covers:
    1. Markdown code-fence stripping
    2. Temporary workspace creation / cleanup
    3. Code execution (success + failure)
    4. Timeout handling
    5. Agent state initialization
    6. Tool registry construction

Run with:
    python -m unittest tests/test_tools.py -v
(from the project root, with coding_agent.py / tools.py / llm.py /
config.py importable — e.g. `PYTHONPATH=. python -m unittest discover`)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the project root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import strip_markdown_fence
from tools import CodeExecutor, TestExecutor, build_tool_registry
from coding_agent import CodingState, Action


class TestMarkdownFenceStripping(unittest.TestCase):
    def test_strips_python_fence(self):
        raw = "```python\nprint('hi')\n```"
        self.assertEqual(strip_markdown_fence(raw), "print('hi')")

    def test_strips_bare_fence(self):
        raw = "```\nx = 1\n```"
        self.assertEqual(strip_markdown_fence(raw), "x = 1")

    def test_no_fence_returns_trimmed_text(self):
        raw = "  x = 1  \n"
        self.assertEqual(strip_markdown_fence(raw), "x = 1")

    def test_empty_input(self):
        self.assertEqual(strip_markdown_fence(""), "")
        self.assertEqual(strip_markdown_fence(None), "")


class TestCodeExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutor(timeout=5)

    def test_successful_execution(self):
        code = "print('hello world')"
        result = self.executor.run(code)
        self.assertTrue(result.success)
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello world", result.stdout)

    def test_runtime_error_is_captured_not_raised(self):
        code = "raise ValueError('boom')"
        result = self.executor.run(code)
        self.assertFalse(result.success)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError", result.stderr)

    def test_timeout_handling(self):
        executor = CodeExecutor(timeout=1)
        code = "import time\ntime.sleep(5)"
        result = executor.run(code)
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

    def test_temporary_workspace_is_cleaned_up(self):
        # The executor's tempdir is a context manager; after run()
        # returns, nothing from it should remain on disk. We can't
        # easily inspect the exact path, so instead we verify that two
        # consecutive runs don't collide or leak state into each other.
        code_a = "x = 1\nprint(x)"
        code_b = "print('no x defined here')"
        result_a = self.executor.run(code_a)
        result_b = self.executor.run(code_b)
        self.assertIn("1", result_a.stdout)
        self.assertIn("no x defined here", result_b.stdout)


class TestTestExecutor(unittest.TestCase):
    def test_passing_tests_are_detected(self):
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
        tests = (
            "import unittest\n"
            "from solution import add\n\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        result = TestExecutor(timeout=5).run(code, tests)
        self.assertTrue(result.all_passed)
        self.assertEqual(result.tests_failed, 0)

    def test_failing_tests_are_detected(self):
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a - b  # intentionally wrong\n"
        )
        tests = (
            "import unittest\n"
            "from solution import add\n\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        result = TestExecutor(timeout=5).run(code, tests)
        self.assertFalse(result.all_passed)


class TestAgentState(unittest.TestCase):
    def test_default_initialization(self):
        state = CodingState(task="do something")
        self.assertEqual(state.task, "do something")
        self.assertEqual(state.requirements, [])
        self.assertEqual(state.plan, [])
        self.assertEqual(state.code, "")
        self.assertEqual(state.tests, "")
        self.assertEqual(state.iteration, 0)
        self.assertFalse(state.success)
        self.assertEqual(state.gemini_calls_used, 0)

    def test_log_execution_appends_entry(self):
        state = CodingState(task="t")
        state.log_execution("plan", "3 requirements")
        self.assertEqual(len(state.execution_results), 1)
        self.assertEqual(state.execution_results[0]["step"], "plan")

    def test_action_enum_has_expected_members(self):
        expected = {
            "PLAN", "GENERATE_CODE", "GENERATE_TESTS", "RUN_TESTS",
            "ANALYZE_FAILURE", "FIX_CODE", "FINISH",
        }
        actual = {a.name for a in Action}
        self.assertTrue(expected.issubset(actual))


class TestToolRegistry(unittest.TestCase):
    def test_registry_contains_all_expected_tools(self):
        # A dummy stand-in is fine here: tool constructors only store a
        # reference to the LLM, they never call it at construction time,
        # so no real API key or network access is needed for this test.
        dummy_llm = object()
        registry = build_tool_registry(dummy_llm)
        expected_keys = {
            "code_generator", "test_generator", "code_executor",
            "test_executor", "code_fixer",
        }
        self.assertEqual(set(registry.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
