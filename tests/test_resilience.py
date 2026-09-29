"""
tests/test_resilience.py
========================
Tests for the end-to-end reliability fixes found during live E2E
verification of the full API + LangGraph + Redis + Gemini pipeline:

1. llm.call_structured must ride out sustained 503 "high demand" storms
   with real exponential backoff and only then fall back to alternate
   models — previously 3 attempts ~3s apart gave up and crashed runs.
2. graph.node_researcher must isolate a per-sub-question LLM crash into
   a structured error result — previously one 503 killed the whole run
   even though SearchStatus was invented for exactly this failure mode.
3. The API layer must not launch two identical graph executions for the
   same run_id while the first is still in flight.
4. /health must report Redis dependency status instead of a blind ok.
"""

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from api.main import app, _INFLIGHT_RUNS, _inflight_lock
from graph import node_research_worker
from llm import LLMCallResult, LLMStructuredOutputError
from schemas import (
    ResearchPlan,
    ResearchResult,
    RunState,
    SearchStatus,
    SubQuestion,
)

client = TestClient(app)


class _FlakyLLM:
    """Callable that raises `fail_times` times, then succeeds."""

    def __init__(self, fail_times: int, error: Exception):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        fake_response = MagicMock()
        fake_response.text = '{"sub_questions": [], "original_question": "q"}'
        fake_response.usage_metadata = None
        return fake_response


_DEMAND_503 = Exception(
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': "
    "'This model is currently experiencing high demand.', "
    "'status': 'UNAVAILABLE'}}"
)


def test_call_structured_retries_through_503_storm(monkeypatch):
    """A 503 storm that clears on the 3rd attempt must NOT raise."""
    from llm import call_structured
    from schemas import ResearchPlan

    flaky = _FlakyLLM(fail_times=2, error=_DEMAND_503)
    sleeps = []
    monkeypatch.setattr("llm.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("llm._get_client", lambda: MagicMock())
    monkeypatch.setattr("llm.MODEL", "gemini-flash-latest")

    with patch("google.genai.types.GenerateContentConfig", MagicMock()):
        # Patch generate_content to our flaky callable
        fake_client = MagicMock()
        fake_client.models.generate_content = flaky
        monkeypatch.setattr("llm._get_client", lambda: fake_client)

        result = call_structured("sys", "user", ResearchPlan)

    assert result.parsed.original_question == "q"
    assert flaky.calls == 3  # two failures, then success
    assert sleeps == [2, 4]  # real exponential backoff, not 3s flat


def test_call_structured_falls_back_to_alternate_model_when_primary_overloaded(monkeypatch):
    """Primary exhausted by 503s -> alternate model is tried before giving up."""
    from llm import call_structured
    from schemas import ResearchPlan

    flaky = _FlakyLLM(fail_times=99, error=_DEMAND_503)  # always fails
    sleeps = []
    monkeypatch.setattr("llm.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("llm.MODEL", "gemini-flash-latest")

    fake_client = MagicMock()
    fake_client.models.generate_content = flaky
    monkeypatch.setattr("llm._get_client", lambda: fake_client)

    with patch("google.genai.types.GenerateContentConfig", MagicMock()):
        try:
            call_structured("sys", "user", ResearchPlan)
        except LLMStructuredOutputError:
            pass  # expected: everything is overloaded
        else:
            raise AssertionError("expected LLMStructuredOutputError when all models are down")

    # 4 attempts on the primary (full exponential ladder) + 2 quick attempts
    # on EACH of the three alternates (independent capacity pools) = 10 calls.
    # Worst-case spend ≈ 30s + 3×(2s+2s) ≈ 42s per LLM call.
    assert flaky.calls == 10
    assert sleeps == [2, 4, 8, 16, 2, 2, 2, 2, 2, 2]


def test_call_structured_does_not_mask_non_availability_errors(monkeypatch):
    """A bad API key (400-ish) must raise immediately — no retry theater."""
    from llm import call_structured
    from schemas import ResearchPlan

    flaky = _FlakyLLM(fail_times=99, error=Exception("400 API key not valid"))
    sleeps = []
    monkeypatch.setattr("llm.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("llm.MODEL", "gemini-flash-latest")

    fake_client = MagicMock()
    fake_client.models.generate_content = flaky
    monkeypatch.setattr("llm._get_client", lambda: fake_client)

    with patch("google.genai.types.GenerateContentConfig", MagicMock()):
        try:
            call_structured("sys", "user", ResearchPlan)
        except LLMStructuredOutputError:
            pass
        else:
            raise AssertionError("expected LLMStructuredOutputError on non-availability error")

    assert flaky.calls == 1  # fail fast, no retries
    assert sleeps == []


def test_node_researcher_isolates_llm_crash_per_sub_question():
    """
    THE live-E2E failure: one sub-question raising during LLM extraction
    killed the whole run. In the parallel architecture each sub-question
    runs in its own worker node, so a crash is contained to that worker:
    it must return a structured error DELTA (not raise past the node) and
    the other workers are unaffected by construction.
    """
    def boom(sq_id, question):
        raise LLMStructuredOutputError("Gemini API call failed after retries: 503")

    state = RunState(
        question="Test",
        plan=ResearchPlan(
            original_question="Test",
            sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
        ),
    )

    with patch("graph.research_sub_question", side_effect=boom):
        out = node_research_worker({
            "sub_question": state.plan.sub_questions[0].model_dump(),
            "run_state": state.model_dump(),
        })

    failed = out["results"][0]
    assert failed["sub_question_id"] == "sq1"
    assert failed["status"]["ok"] is False
    assert failed["status"]["reason"] == "error"
    assert "503" in failed["status"]["detail"]
    assert out["_worker_tokens"] == 0
    assert out["_worker_searches"] == 1  # the search attempt still counts


def test_api_does_not_double_start_run_already_in_flight(monkeypatch):
    """POST /research while a run is mid-flight must not fork a second execution."""
    started = []

    def fake_run_and_persist(run_id):
        started.append(run_id)

    monkeypatch.setattr("api.main.run_and_persist", fake_run_and_persist)

    # Simulate the run already executing in another thread
    with _inflight_lock:
        _INFLIGHT_RUNS.add("already-running")

    resp = client.post("/research", json={"question": "Q", "run_id": "already-running"})
    assert resp.status_code == 200
    assert started == []  # second execution was suppressed

    with _inflight_lock:
        _INFLIGHT_RUNS.discard("already-running")


def test_health_reports_redis_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["redis"] is True  # Redis is up in the test environment


def test_report_schema_preserves_sections_and_conclusion():
    """
    Regression guard for the schema round-trip: sections/conclusion written
    by the writer must survive Redis persistence (model_dump/model_validate).
    """
    from schemas import Citation, Report

    report = Report(
        question="q",
        summary="s",
        sections=[{"heading": "H", "body": "B"}],
        conclusion="C",
        citations=[Citation(claim="c", source_url="https://x.com", finding_id="f1")],
    )
    restored = Report.model_validate(report.model_dump())
    assert restored.sections[0].heading == "H"
    assert restored.conclusion == "C"
