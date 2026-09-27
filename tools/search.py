"""
tools/search.py
================
PHASE 3: "Give the researcher live search with source tracking."

THEORY: This file replaces `researcher.stub_search` with something real,
but the Researcher's own code barely changes — that's the payoff of
having designed the seam as a swappable `search_backend` function back
in Phase 1.

Three concrete requirements from the brief, and where each is implemented:
1. "Fetch and extract the page rather than trusting search snippets,
   which are truncated and often misleading." -> `fetch_and_extract()`
2. "Deduplicate by URL and cap results per domain so a single site cannot
   dominate a report." -> `_dedup_and_cap()`
3. "Handle each failure path explicitly: no results, rate limited,
   paywalled, timed out." -> every branch below returns a `SearchStatus`
   instead of letting an exception escape.
"""

from __future__ import annotations
import os
import time
from urllib.parse import urlparse
import httpx
import trafilatura
from tavily import TavilyClient
from dotenv import load_dotenv
from schemas import SearchStatus

load_dotenv()

_tavily = None


def _client() -> TavilyClient:
    global _tavily
    if _tavily is None:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        _tavily = TavilyClient(api_key=key)
    return _tavily


MAX_RESULTS_PER_DOMAIN = 2
MAX_RESULTS_TOTAL = 5
FETCH_TIMEOUT_SECONDS = 8


def _dedup_and_cap(raw_results: list[dict]) -> list[dict]:
    """Dedup by exact URL, then cap how many results come from one domain."""
    seen_urls: set[str] = set()
    per_domain_count: dict[str, int] = {}
    kept: list[dict] = []

    for r in raw_results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        domain = urlparse(url).netloc
        if per_domain_count.get(domain, 0) >= MAX_RESULTS_PER_DOMAIN:
            continue
        seen_urls.add(url)
        per_domain_count[domain] = per_domain_count.get(domain, 0) + 1
        kept.append(r)
        if len(kept) >= MAX_RESULTS_TOTAL:
            break
    return kept


def fetch_and_extract(url: str) -> str | None:
    """
    Downloads a page and extracts its main readable text with trafilatura,
    instead of trusting the (often truncated/misleading) search snippet.
    Returns None on any failure — caller decides whether that's fatal for
    this particular source or just means "fall back to the snippet".
    """
    try:
        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0 (research-assistant-bot)"},
            follow_redirects=True,
        )
        if resp.status_code == 403 or resp.status_code == 402:
            return None  # treat as paywalled/blocked, not a crash
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False)
        return text
    except (httpx.TimeoutException, httpx.ConnectError):
        return None
    except httpx.HTTPStatusError:
        return None
    except Exception:
        return None


def tavily_search_backend(query: str) -> tuple[SearchStatus, list[dict]]:
    """
    THEORY: this matches the `Callable[[str], ...]` shape `researcher.py`
    expects, but now returns (SearchStatus, results) instead of a bare
    list, because Phase 3 requires the search layer itself to report
    *why* it failed, not just *that* it failed.
    """
    try:
        response = _client().search(query=query, max_results=8, search_depth="basic")
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg:
            return SearchStatus(ok=False, reason="rate_limited", detail=str(e)), []
        if "timeout" in msg or "timed out" in msg:
            return SearchStatus(ok=False, reason="timeout", detail=str(e)), []
        return SearchStatus(ok=False, reason="error", detail=str(e)), []

    raw_results = response.get("results", [])
    if not raw_results:
        return SearchStatus(ok=False, reason="no_results"), []

    capped = _dedup_and_cap(raw_results)

    enriched = []
    for r in capped:
        full_text = fetch_and_extract(r["url"])
        if full_text is None:
            # THEORY: a single source failing to fetch is NOT a run
            # failure — we fall back to Tavily's own snippet rather than
            # dropping the source entirely or raising.
            snippet = r.get("content", "")[:800]
        else:
            snippet = full_text[:1500]

        if not snippet.strip():
            # THEORY: page fetched but had nothing extractable (e.g. a
            # pure paywall wall or JS-only page) -> treat as unusable
            # source, skip it rather than feed empty context to the LLM.
            continue

        enriched.append(
            {"title": r.get("title", ""), "url": r["url"], "snippet": snippet}
        )

    if not enriched:
        return SearchStatus(ok=False, reason="paywalled", detail="all sources unextractable"), []

    return SearchStatus(ok=True, reason="ok"), enriched
