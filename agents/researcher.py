"""
agents/researcher.py
=====================
THEORY: "Researcher executes one sub-question with search and returns
findings with citations."

Phase 1 note: we do NOT wire real Tavily search here yet. Per the build
order ("test each agent alone before wiring them together"), Phase 1 uses
a `search_backend` function you can swap out. Right now it's a stub that
returns canned results, so you can validate the Researcher's PROMPTING
and PARSING logic in total isolation from network flakiness. Phase 3
replaces `stub_search` with real Tavily + page fetch/extract, and adds
the SearchStatus failure handling. The Researcher's own code barely
changes — that's the point of the seam.
"""

from pydantic import BaseModel
from schemas import Finding, ResearchResult, SearchStatus
from llm import call_structured
from typing import Callable

SYSTEM_PROMPT = """You are a research extraction function. You will be given \
a sub-question and a list of search results (title, url, snippet). Extract \
specific, checkable claims that answer the sub-question, each backed by \
exactly one of the given sources. Do not invent claims not supported by the \
provided snippets. If nothing in the sources answers the sub-question, \
return an empty findings list — do not guess.
"""


class _FindingsOnly(BaseModel):
    """Internal schema: the LLM only needs to produce findings, not full
    ResearchResult (status is computed by code, not the model — status is
    infrastructure information the model can't actually know)."""
    findings: list[Finding]


def stub_search(query: str) -> tuple[SearchStatus, list[dict]]:
    """
    THEORY: this is the seam Phase 3 replaces with tools.search.tavily_search_backend.
    Keeping it as a free function (not a method) makes swapping trivial:
    researcher.py just imports a different callable of the same shape.
    Kept around for offline testing / smoke tests with no API keys.
    """
    return (
        SearchStatus(ok=True, reason="ok"),
        [
            {
                "title": "Example Source on " + query,
                "url": "https://example.com/article",
                "snippet": f"(stub) Relevant snippet discussing: {query}",
            }
        ],
    )


def _default_backend():
    # Imported lazily so this module still imports fine (e.g. for the
    # stub-based smoke test) without TAVILY_API_KEY being set.
    from tools.search import tavily_search_backend
    return tavily_search_backend


def research_sub_question(
    sub_question_id: str,
    question_text: str,
    search_backend: Callable[[str], tuple[SearchStatus, list[dict]]] | None = None,
) -> tuple[ResearchResult, int]:
    """Returns (ResearchResult, tokens_used)."""
    backend = search_backend or _default_backend()
    try:
        status, raw_results = backend(question_text)
    except Exception as e:
        # THEORY: a backend that raises anyway (bug, unexpected exception
        # type) is still contained here rather than killing the whole run.
        return (
            ResearchResult(
                sub_question_id=sub_question_id,
                status=SearchStatus(ok=False, reason="error", detail=str(e)),
                findings=[],
            ),
            0,
        )

    if not status.ok:
        return (
            ResearchResult(sub_question_id=sub_question_id, status=status, findings=[]),
            0,
        )

    sources_block = "\n".join(
        f"- title: {r['title']}\n  url: {r['url']}\n  snippet: {r['snippet']}"
        for r in raw_results
    )
    user_prompt = (
        f"Sub-question: {question_text}\n\n"
        f"Search results:\n{sources_block}\n\n"
        f"Return JSON with a 'findings' list. Each finding needs sub_question_id="
        f"'{sub_question_id}', a specific claim, a confidence level, and a source "
        f"object with url/title/snippet copied from the matching search result above."
    )

    result = call_structured(SYSTEM_PROMPT, user_prompt, _FindingsOnly)
    findings = result.parsed.findings
    for idx, f in enumerate(findings, start=1):
        f.sub_question_id = sub_question_id  # enforce, don't trust the model
        f.id = f"{sub_question_id}_f{idx}"  # ensure globally unique finding ID across all sub-questions

    return (
        ResearchResult(sub_question_id=sub_question_id, status=status, findings=findings),
        result.total_tokens,
    )


if __name__ == "__main__":
    import json
    # THEORY: run with the offline stub by default so this smoke test
    # never requires TAVILY_API_KEY. Pass search_backend=None (the default)
    # to exercise real Tavily instead, once your .env is filled in.
    r, toks = research_sub_question("sq1", "What are Redis persistence options?", search_backend=stub_search)
    print(json.dumps(r.model_dump(), indent=2, default=str))
    print(f"\ntokens used: {toks}")
