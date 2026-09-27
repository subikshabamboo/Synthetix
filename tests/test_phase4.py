"""
tests/test_phase4.py
====================
Tests Phase 4 Redis persistence:
- save_state and load_state
- get_or_create idempotency
- run_and_persist resume behavior and idempotency on done runs
"""

from unittest.mock import patch
from schemas import (
    RunState,
    ResearchPlan,
    SubQuestion,
    Finding,
    SourceRecord,
    SearchStatus,
    ResearchResult,
    Report,
    Citation,
)
from storage.redis_store import save_state, load_state, get_or_create, run_and_persist


def test_save_and_load_state():
    state = RunState(question="Persistence test?")
    save_state(state)

    loaded = load_state(state.run_id)
    assert loaded is not None
    assert loaded.run_id == state.run_id
    assert loaded.question == "Persistence test?"
    assert loaded.status == "planning"


def test_get_or_create_idempotency():
    # Calling get_or_create with existing run_id returns existing state without overwriting
    state = get_or_create(None, "Question 1")
    state.status = "researching"
    save_state(state)

    same_state = get_or_create(state.run_id, "Different question")
    assert same_state.run_id == state.run_id
    assert same_state.status == "researching"
    assert same_state.question == "Question 1"


def test_run_and_persist_idempotent_on_done():
    # If run status is already "done", run_and_persist returns immediately without running graph
    state = RunState(
        question="Finished run test",
        status="done",
        report=Report(question="Finished run test", summary="Summary", citations=[]),
    )
    save_state(state)

    with patch("storage.redis_store.build_graph") as mock_build_graph:
        final = run_and_persist(state.run_id)
        assert final.status == "done"
        assert not mock_build_graph.called


def test_run_and_persist_full_execution_persists_steps():
    state = get_or_create(None, "Full persistence test")

    mock_plan = ResearchPlan(
        original_question="Full persistence test",
        sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
    )
    mock_findings = [
        Finding(
            id="f1",
            sub_question_id="sq1",
            claim="Persisted finding",
            source=SourceRecord(url="https://example.com", snippet="Snippet"),
        )
    ]
    mock_report = Report(
        question="Full persistence test",
        summary="Done summary",
        citations=[Citation(claim="Persisted finding", source_url="https://example.com", finding_id="f1")],
    )

    with patch("graph.run_planner", return_value=(mock_plan, 10)), \
         patch("graph.research_sub_question", return_value=(ResearchResult(sub_question_id="sq1", status=SearchStatus(ok=True), findings=mock_findings), 20)), \
         patch("graph.write_report", return_value=(mock_report, 30)):

        final = run_and_persist(state.run_id)
        assert final.status == "done"
        assert final.report is not None

        # Verify it was saved to Redis
        persisted = load_state(state.run_id)
        assert persisted.status == "done"
        assert persisted.report.summary == "Done summary"
