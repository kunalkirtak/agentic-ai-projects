"""
tools.py — Local tools the agent uses that do NOT require Gemini:
free web search, lightweight webpage fetching/extraction, and small
normalization helpers (URL/domain dedup).

Keeping these separate from llm.py is what makes the agent "agentic"
rather than a single prompt-to-Gemini call: search and extraction are
deterministic, inspectable, and free.
"""

import re
import time
from urllib.parse import urlparse

import requests

from config import (
    MAX_PAGE_CHARS,
    MAX_SEARCH_RESULTS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)


class SearchError(RuntimeError):
    """Raised when the search backend fails entirely (not just zero results)."""


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list:
    """
    Free web search using ddgs (DuckDuckGo Search).
    Returns a list of {"title", "url", "snippet"} dicts. Never raises for
    "no results" — only for a hard backend failure, and even then it
    returns an empty list so the agent can continue.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name, fallback
        except ImportError as exc:
            raise SearchError(
                "No search backend installed. Run: pip install -q ddgs"
            ) from exc

    results = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": (item.get("title") or "").strip(),
                        "url": (item.get("href") or item.get("url") or "").strip(),
                        "snippet": (item.get("body") or item.get("snippet") or "").strip(),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ search failed for query '{query}': {exc}")
        return []

    return results[:max_results]


def normalize_domain(url: str) -> str:
    """Extract a comparable root domain from a URL, e.g. 'www.acme.com' -> 'acme.com'."""
    if not url:
        return ""
    try:
        netloc = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
        netloc = re.sub(r"^www\.", "", netloc)
        return netloc
    except Exception:  # noqa: BLE001
        return url.lower().strip()


def normalize_company_name(name: str) -> str:
    """Lowercase + strip common suffixes so 'Acme Inc.' and 'Acme' can be matched."""
    if not name:
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[.,]", "", cleaned)
    cleaned = re.sub(
        r"\b(inc|ltd|llc|pvt|private|limited|corp|corporation|co)\b", "", cleaned
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """
    Fetch a single public webpage and return cleaned, truncated text.
    Returns an empty string (never raises) on any failure — timeouts,
    blocks, non-HTML content, etc. — so the agent can keep going with
    partial information rather than crashing.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  ⚠ beautifulsoup4 not installed; skipping page extraction")
        return ""

    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except requests.exceptions.RequestException as exc:
        print(f"  ⚠ could not fetch {url}: {exc}")
        return ""
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ unexpected error fetching {url}: {exc}")
        return ""


def polite_pause(seconds: float = 0.5):
    """Small delay between outbound requests to be a good web citizen."""
    time.sleep(seconds)


def dedupe_companies(companies: list) -> list:
    """
    Remove duplicate candidate companies by normalized name OR normalized
    domain, keeping the first (best-ranked) occurrence and merging any
    extra source URLs into it.
    """
    seen_names = {}
    seen_domains = {}
    deduped = []

    for company in companies:
        name_key = normalize_company_name(company.get("company", ""))
        domain_key = normalize_domain(company.get("website", ""))

        existing_index = seen_names.get(name_key) if name_key else None
        if existing_index is None and domain_key:
            existing_index = seen_domains.get(domain_key)

        if existing_index is not None:
            existing = deduped[existing_index]
            for url in company.get("source_urls", []):
                if url not in existing["source_urls"]:
                    existing["source_urls"].append(url)
            continue

        deduped.append(company)
        index = len(deduped) - 1
        if name_key:
            seen_names[name_key] = index
        if domain_key:
            seen_domains[domain_key] = index

    return deduped
