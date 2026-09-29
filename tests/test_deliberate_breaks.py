"""
tests/test_deliberate_breaks.py
===============================
Tests the exact 4 deliberate failure/edge cases specified in the README:
1. Kill the budget on purpose: Supervisor proceeds to Writer with gap in trace instead of looping/hanging.
2. Feed Researcher query returning nothing (gibberish): SearchStatus(reason="no_results") -> Writer produces graceful insufficient evidence report.
3. Manually corrupt citation: GET /research/{id} surfaces "failed_citation_validation".
4. Re-invoking graph on completed run: Idempotent no-op.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from schemas import (
    RunState,
    RunBudget,
    ResearchPlan,
    SubQuestion,
    Finding,
    SourceRecord,
    SearchStatus,
    ResearchResult,
    Report,
    Citation,
)
from graph import node_supervisor, run_research
from agents.researcher import research_sub_question
from agents.writer import write_report
from storage.redis_store import save_state, load_state, run_and_persist
from api.main import app

client = TestClient(app)


def test_break_1_kill_budget_on_purpose():
    # Construct state with exhausted budget and missing subquestions
    budget = RunBudget(max_sub_questions=1, wall_clock_seconds=0)
    state = RunState(
        question="Budget kill test",
        budget=budget,
        plan=ResearchPlan(
            original_question="Budget kill test",
            sub_questions=[
                SubQuestion(id="sq1", question="Q1", rationale="R1"),
                SubQuestion(id="sq2", question="Q2", rationale="R2"),
            ],
            max_sub_questions=2,
        ),
        results=[],  # nothing completed
        revision_count=0,
    )
    assert state.budget.time_exhausted()

    # Supervisor should refuse to re-fan (dead budget) and hand off with a
    # documented gap in trace rather than route back to researchers.
    result_dict = node_supervisor(state.model_dump())
    assert result_dict["status"] == "auditing"
    assert "_refan" not in result_dict  # dead budget: no more research
    assert any("proceeding to writer with gaps" in t["note"] for t in result_dict["trace"])


def test_break_2_no_search_results_gibberish():
    # Empty search results backend
    def no_results_backend(query: str):
        return SearchStatus(ok=False, reason="no_results"), []

    res, tokens = research_sub_question("sq_gibberish", "asdfghjklqwerty12345", search_backend=no_results_backend)
    assert res.status.ok is False
    assert res.status.reason == "no_results"
    assert res.findings == []

    # Writer given empty findings
    report, w_tokens = write_report("asdfghjklqwerty12345", res.findings)
    assert "Insufficient evidence" in report.summary
    assert report.citations == []


def test_break_3_corrupt_citation_in_redis():
    state = RunState(
        question="Corrupt citation test",
        status="done",
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True),
                findings=[
                    Finding(
                        id="finding_true_id",
                        sub_question_id="sq1",
                        claim="True claim",
                        source=SourceRecord(url="https://valid.com", snippet="Valid text"),
                    )
                ],
            )
        ],
        report=Report(
            question="Corrupt citation test",
            summary="Some report summary.",
            citations=[
                Citation(
                    claim="True claim",
                    source_url="https://valid.com",
                    finding_id="fake_invented_id",  # Corrupted ID
                )
            ],
        ),
    )
    save_state(state)

    resp = client.get(f"/research/{state.run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed_citation_validation"
    assert any("unknown finding_id=fake_invented_id" in p for p in data["validation_problems"])


def test_break_4_idempotency_on_completed_run():
    state = RunState(
        question="Idempotency check",
        status="done",
        report=Report(question="Idempotency check", summary="Summary", citations=[]),
    )
    save_state(state)

    with patch("graph.build_graph") as mock_build:
        final = run_and_persist(state.run_id)
        assert final.status == "done"
        assert not mock_build.called
