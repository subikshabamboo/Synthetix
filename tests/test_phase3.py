"""
tests/test_phase3.py
====================
Tests Phase 3 search tool features:
- Deduplication by URL
- Capping per domain (max 2) and total (max 5)
- Text extraction with trafilatura / fallback handling
- SearchStatus failure handling
"""

from unittest.mock import patch, MagicMock
from tools.search import _dedup_and_cap, fetch_and_extract, tavily_search_backend
from schemas import SearchStatus


def test_dedup_and_cap():
    raw_results = [
        {"url": "https://a.com/page1", "title": "A1"},
        {"url": "https://a.com/page1", "title": "A1-duplicate"},
        {"url": "https://a.com/page2", "title": "A2"},
        {"url": "https://a.com/page3", "title": "A3-excess"},  # Should be capped (max 2 for a.com)
        {"url": "https://b.com/page1", "title": "B1"},
        {"url": "https://c.com/page1", "title": "C1"},
        {"url": "https://d.com/page1", "title": "D1"},
        {"url": "https://e.com/page1", "title": "E1-excess"},  # Should be capped (max 5 total)
    ]

    capped = _dedup_and_cap(raw_results)
    assert len(capped) == 5
    urls = [r["url"] for r in capped]
    assert urls == [
        "https://a.com/page1",
        "https://a.com/page2",
        "https://b.com/page1",
        "https://c.com/page1",
        "https://d.com/page1",
    ]


def test_fetch_and_extract_success():
    html_content = "<html><body><p>This is the main extracted content of the article.</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_resp):
        extracted = fetch_and_extract("https://example.com/article")
        assert extracted is not None
        assert "main extracted content" in extracted


def test_fetch_and_extract_paywall_or_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch("httpx.get", return_value=mock_resp):
        extracted = fetch_and_extract("https://example.com/paywalled")
        assert extracted is None


def test_tavily_search_backend_no_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}

    with patch("tools.search._client", return_value=mock_client):
        status, results = tavily_search_backend("gibberish query")
        assert status.ok is False
        assert status.reason == "no_results"
        assert results == []


def test_tavily_search_backend_rate_limited():
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Rate limit exceeded 429")

    with patch("tools.search._client", return_value=mock_client):
        status, results = tavily_search_backend("query")
        assert status.ok is False
        assert status.reason == "rate_limited"
