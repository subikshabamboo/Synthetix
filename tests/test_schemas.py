"""
tests/test_schemas.py
=====================
Tests all Pydantic contracts and budget invariants in schemas.py.
"""

from datetime import datetime, timezone, timedelta
import pytest
from schemas import (
    SubQuestion,
    ResearchPlan,
    SourceRecord,
    Finding,
    SearchStatus,
    ResearchResult,
    Citation,
    Report,
    RunBudget,
    RunState,
)


def test_research_plan_truncation_guard():
    # Enforces budget in schema: max_sub_questions=3 should truncate list of 5 to 3
    sqs = [
        SubQuestion(question=f"Q{i}", rationale=f"Why {i}")
        for i in range(5)
    ]
    plan = ResearchPlan(
        original_question="Main Q",
        sub_questions=sqs,
        max_sub_questions=3,
    )
    assert len(plan.sub_questions) == 3
    assert plan.sub_questions[0].question == "Q0"
    assert plan.sub_questions[2].question == "Q2"


def test_source_record_and_finding():
    source = SourceRecord(
        url="https://example.com/doc",
        title="Example Doc",
        snippet="Relevant text snippet",
    )
    finding = Finding(
        sub_question_id="sq1",
        claim="Claim A",
        source=source,
        confidence="high",
    )
    assert finding.id is not None
    assert finding.source.url == "https://example.com/doc"
    assert finding.confidence == "high"


def test_search_status():
    ok_status = SearchStatus(ok=True, reason="ok")
    assert ok_status.ok is True

    err_status = SearchStatus(ok=False, reason="no_results", detail="No pages found")
    assert err_status.ok is False
    assert err_status.reason == "no_results"


def test_run_budget_exhaustion():
    budget = RunBudget(
        max_sub_questions=2,
        max_searches_per_sub_question=2,
        max_total_tokens=1000,
        wall_clock_seconds=10,
    )
    assert not budget.searches_exhausted()
    assert not budget.tokens_exhausted()
    assert not budget.time_exhausted()

    # Exhaust searches
    budget.searches_used = 4
    assert budget.searches_exhausted()

    # Exhaust tokens
    budget.tokens_used = 1000
    assert budget.tokens_exhausted()

    # Exhaust time
    budget.started_at = datetime.now(timezone.utc) - timedelta(seconds=15)
    assert budget.time_exhausted()


def test_run_state_defaults_and_validation():
    state = RunState(question="Test question?")
    assert state.run_id is not None
    assert state.status == "planning"
    assert state.plan is None
    assert state.results == []
    assert state.report is None
    assert state.revision_count == 0
    assert isinstance(state.budget, RunBudget)
