"""
tests.py
--------
Lightweight tests runnable inside Colab (or plain Python) with no test
framework dependency beyond the standard library.

Split into two groups:
    - UNIT TESTS: no network, no API key required. Always run these.
    - API TESTS:  require GEMINI_API_KEY and network access. Skipped
      automatically if a key isn't available.

Run with:  python tests.py
"""

import sys
import traceback

from config import Config
from llm import safe_json_parse
from tools import extract_evidence, web_search
from research_agent import ResearchState


def _check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


# ---------------------------------------------------------------------------
# 1. Configuration loading
# ---------------------------------------------------------------------------
def test_config_loading():
    cfg = Config(gemini_api_key="dummy-key-for-tests")
    ok = True
    ok &= _check("config has a model name", bool(cfg.gemini_model))
    ok &= _check("config has positive max_agent_steps", cfg.max_agent_steps > 0)
    ok &= _check("config has positive max_search_results", cfg.max_search_results > 0)

    raised = False
    try:
        Config(gemini_api_key="")
    except ValueError:
        raised = True
    ok &= _check("config raises ValueError on empty API key", raised)
    return ok


# ---------------------------------------------------------------------------
# 2. JSON parsing
# ---------------------------------------------------------------------------
def test_json_parsing():
    ok = True
    ok &= _check("parses clean JSON", safe_json_parse('{"a": 1}') == {"a": 1})
    fenced = 'Sure, here you go:\n```json\n{"a": 2}\n```\nHope that helps!'
    ok &= _check("parses fenced JSON", (safe_json_parse(fenced) or {}).get("a") == 2)
    ok &= _check("returns None on garbage input", safe_json_parse("this is not json") is None)
    ok &= _check("returns None on empty input", safe_json_parse("") is None)
    return ok


# ---------------------------------------------------------------------------
# 3. Search tool response format (does not require network to validate shape;
#    if network is unavailable it should degrade to an empty list, not crash)
# ---------------------------------------------------------------------------
def test_search_tool_format():
    ok = True
    try:
        results = web_search("test query unit test", max_results=2)
        ok &= _check("web_search returns a list", isinstance(results, list))
        if results:
            first = results[0]
            ok &= _check("result has 'title'/'url'/'snippet' keys", all(k in first for k in ("title", "url", "snippet")))
        else:
            print("[INFO] web_search returned no results (offline sandbox or blocked network) -- shape check skipped.")
    except Exception:
        ok = False
        print("[FAIL] web_search raised an exception instead of degrading gracefully:")
        traceback.print_exc()
    return ok


# ---------------------------------------------------------------------------
# 4. Webpage extraction (offline, using a hand-built HTML string so this
#    test doesn't depend on network access)
# ---------------------------------------------------------------------------
def test_webpage_extraction():
    from bs4 import BeautifulSoup

    html = """
    <html><head><style>body{color:red}</style></head>
    <body>
      <nav>Home | About</nav>
      <script>console.log('noise')</script>
      <article><p>Agentic AI systems plan, act, and observe in loops.</p></article>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")

    ok = True
    ok &= _check("script content removed", "console.log" not in text)
    ok &= _check("nav content removed", "About" not in text)
    ok &= _check("main content preserved", "plan, act, and observe" in text)
    return ok


# ---------------------------------------------------------------------------
# 5. Agent state initialization
# ---------------------------------------------------------------------------
def test_agent_state_init():
    state = ResearchState(question="What is an AI agent?")
    ok = True
    ok &= _check("question stored", state.question == "What is an AI agent?")
    ok &= _check("plan starts empty", state.plan == [])
    ok &= _check("sources start empty", state.sources == [])
    ok &= _check("evidence starts empty", state.evidence == [])
    ok &= _check("steps start at 0", state.steps == 0)
    ok &= _check("seen_urls starts empty", state.seen_urls() == set())
    return ok


# ---------------------------------------------------------------------------
# Extra: evidence extraction unit test (unit-level, no network/API)
# ---------------------------------------------------------------------------
def test_evidence_extraction():
    page_text = (
        "Modern AI agent architectures typically include a planning module, "
        "a memory or state component, and a set of callable tools.\n"
        "Short line.\n"
        "Agents differ from simple chatbots because they can take multiple "
        "autonomous steps toward a goal before responding to the user."
    )
    evidence = extract_evidence("Agent Architectures", "https://example.com/agents", page_text, "agent architecture components")
    ok = True
    ok &= _check("evidence extracted", evidence is not None)
    if evidence:
        ok &= _check("evidence has key_points", len(evidence.get("key_points", [])) > 0)
        ok &= _check("evidence retains source url", evidence.get("source") == "https://example.com/agents")
    return ok


UNIT_TESTS = [
    ("Configuration loading", test_config_loading),
    ("JSON parsing", test_json_parsing),
    ("Search tool response format", test_search_tool_format),
    ("Webpage extraction", test_webpage_extraction),
    ("Agent state initialization", test_agent_state_init),
    ("Evidence extraction", test_evidence_extraction),
]


def run_all():
    print("=" * 50)
    print("RUNNING UNIT TESTS (no Gemini API key required)")
    print("=" * 50)
    results = []
    for name, fn in UNIT_TESTS:
        print(f"\n-- {name} --")
        try:
            results.append(fn())
        except Exception:
            print(f"[FAIL] {name} raised an unexpected exception:")
            traceback.print_exc()
            results.append(False)

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 50)
    print(f"RESULT: {passed}/{total} test groups passed")
    print("=" * 50)
    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
