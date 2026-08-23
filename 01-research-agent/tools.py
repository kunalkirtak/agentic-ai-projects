"""
tools.py
--------
The agent's tool system: web search, webpage fetching, and evidence
extraction. These are plain Python functions/classes -- no framework,
no tool-calling protocol -- because the agent loop in research_agent.py
decides explicitly which tool to call and when.

Tools implemented:
    1. web_search(query)       -> list[dict]  (title, url, snippet)
    2. fetch_webpage(url)      -> str          (cleaned text)
    3. extract_evidence(...)   -> dict         (structured evidence)

All tools fail soft: on error they return an empty result plus a logged
warning, never an unhandled exception, so one bad source never kills a
research run.
"""

import logging
import re
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("research_agent.tools")

USER_AGENT = (
    "Mozilla/5.0 (compatible; AgenticResearchAgent/1.0; "
    "+https://github.com/) research-bot"
)

TRUSTED_DOMAIN_HINTS = (
    ".gov", ".edu", "wikipedia.org", "arxiv.org", "nature.com",
    "acm.org", "ieee.org", "who.int", "un.org", "nih.gov",
    "docs.", "developer.", ".org",
)


def _looks_reliable(url: str) -> bool:
    """Cheap heuristic used only for ranking/sorting, never for filtering out."""
    url_lower = (url or "").lower()
    return any(hint in url_lower for hint in TRUSTED_DOMAIN_HINTS)


# ---------------------------------------------------------------------------
# Tool 1: Web Search
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Free web search using the `ddgs` package (DuckDuckGo search wrapper).

    Returns a list of dicts: {"title": ..., "url": ..., "snippet": ...}
    Returns an empty list (never raises) if the search fails or the
    dependency is unavailable.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("The 'ddgs' package is not installed. Run: pip install -q ddgs")
        return []

    results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": item.get("title", "").strip(),
                        "url": item.get("href") or item.get("url", ""),
                        "snippet": item.get("body", "").strip(),
                    }
                )
    except Exception as exc:
        logger.warning("web_search failed for query %r: %s", query, exc)
        return []

    # Prefer results that look like reliable sources, without discarding others.
    results.sort(key=lambda r: _looks_reliable(r.get("url", "")), reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Tool 2: Webpage Fetcher
# ---------------------------------------------------------------------------
def fetch_webpage(url: str, max_chars: int = 12000, timeout: int = 10) -> str:
    """
    Fetch a URL and return cleaned, readable text (scripts/styles/nav
    stripped). Truncated to max_chars to keep Gemini prompts small.

    Returns "" (never raises) on any failure -- invalid URL, timeout,
    non-HTML content, etc.
    """
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        logger.warning("fetch_webpage: invalid URL skipped: %r", url)
        return ""

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("fetch_webpage failed for %s: %s", url, exc)
        return ""

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        logger.warning("fetch_webpage: skipping non-HTML content at %s (%s)", url, content_type)
        return ""

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "svg", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Collapse excessive whitespace/blank lines left behind by decompose().
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text[:max_chars]
    except Exception as exc:
        logger.warning("fetch_webpage: extraction failed for %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Tool 3: Source Extractor / Evidence Collector
# ---------------------------------------------------------------------------
def extract_evidence(
    source_title: str,
    source_url: str,
    page_text: str,
    query_context: str,
    max_key_points: int = 5,
    max_point_chars: int = 240,
) -> Optional[Dict]:
    """
    Convert raw page text into a compact, structured evidence record
    WITHOUT calling the LLM (kept purely heuristic to save Gemini quota).

    This is a lightweight, deterministic extractor: it splits the page
    into candidate sentences/paragraphs and keeps the ones most likely to
    be informative (reasonable length, contains some overlap with the
    query context). The LLM analyst step later decides what this evidence
    actually means -- this tool's job is just to shrink noisy HTML into a
    few candidate facts.

    Returns None if no usable text was found.
    """
    if not page_text or not page_text.strip():
        return None

    # Split into paragraph-like chunks.
    chunks = [c.strip() for c in re.split(r"\n+", page_text) if c.strip()]
    # Keep chunks of reasonable "sentence-like" length.
    candidates = [c for c in chunks if 40 <= len(c) <= 500]

    if not candidates:
        candidates = [c for c in chunks if len(c) > 20][:max_key_points]

    query_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query_context or "")}

    def relevance_score(chunk: str) -> int:
        chunk_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", chunk)}
        return len(query_terms & chunk_terms)

    candidates.sort(key=relevance_score, reverse=True)

    key_points = []
    for chunk in candidates[: max_key_points * 3]:
        snippet = chunk if len(chunk) <= max_point_chars else chunk[:max_point_chars].rsplit(" ", 1)[0] + "..."
        if snippet not in key_points:
            key_points.append(snippet)
        if len(key_points) >= max_key_points:
            break

    if not key_points:
        return None

    return {
        "source": source_url,
        "title": source_title or source_url,
        "key_points": key_points,
        "relevance": "high" if relevance_score(" ".join(key_points)) > 0 else "unknown",
    }
