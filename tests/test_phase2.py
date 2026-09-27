"""
tests/test_phase2.py
====================
Tests Phase 2 LangGraph StateGraph:
- Graph construction and node transitions
- Supervisor gap detection and single-retry rule
- Budget enforcement stopping loops
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
from graph import build_graph, run_research, node_supervisor, GraphState


def test_supervisor_routes_to_writer_on_full_coverage():
    state = RunState(
        question="Test question",
        plan=ResearchPlan(
            original_question="Test question",
            sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
        ),
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True, reason="ok"),
                findings=[
                    Finding(
                        id="f1",
                        sub_question_id="sq1",
                        claim="C1",
                        source=SourceRecord(url="https://example.com", snippet="S1"),
                    )
                ],
            )
        ],
    )
    result_dict = node_supervisor(state.model_dump())
    assert result_dict["status"] == "writing"
    assert any("full coverage" in t["note"] for t in result_dict["trace"])


def test_supervisor_routes_to_researcher_once_on_gap():
    state = RunState(
        question="Test question",
        plan=ResearchPlan(
            original_question="Test question",
            sub_questions=[
                SubQuestion(id="sq1", question="Q1", rationale="R1"),
                SubQuestion(id="sq2", question="Q2", rationale="R2"),
            ],
        ),
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True, reason="ok"),
                findings=[
                    Finding(
                        id="f1",
                        sub_question_id="sq1",
                        claim="C1",
                        source=SourceRecord(url="https://example.com", snippet="S1"),
                    )
                ],
            )
            # sq2 has no findings / is missing
        ],
        revision_count=0,
    )

    # First pass: gap detected -> routes back to researcher (revision 1/1)
    result_dict = node_supervisor(state.model_dump())
    assert result_dict["status"] == "researching"
    assert result_dict["revision_count"] == 1

    # Second pass: still missing -> cannot loop again (max 1 retry) -> routes to writer
    second_pass = node_supervisor(result_dict)
    assert second_pass["status"] == "writing"
    assert second_pass["revision_count"] == 1


def test_supervisor_proceeds_to_writer_when_budget_dead():
    state = RunState(
        question="Test question",
        plan=ResearchPlan(
            original_question="Test question",
            sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
        ),
        results=[],  # sq1 missing
        revision_count=0,
    )
    # Exhaust tokens
    state.budget.tokens_used = state.budget.max_total_tokens + 100

    result_dict = node_supervisor(state.model_dump())
    assert result_dict["status"] == "writing"
    assert any("budget_dead=True" in t["note"] for t in result_dict["trace"])


def test_full_graph_execution_with_mocks():
    mock_plan = ResearchPlan(
        original_question="Full graph test",
        sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
    )
    mock_findings = [
        Finding(
            id="f1",
            sub_question_id="sq1",
            claim="Finding 1",
            source=SourceRecord(url="https://example.com/1", snippet="Snippet 1"),
        )
    ]
    mock_report = Report(
        question="Full graph test",
        summary="Summary of research.",
        citations=[Citation(claim="Finding 1", source_url="https://example.com/1", finding_id="f1")],
    )

    with patch("graph.run_planner", return_value=(mock_plan, 50)), \
         patch("graph.research_sub_question", return_value=(ResearchResult(sub_question_id="sq1", status=SearchStatus(ok=True), findings=mock_findings), 60)), \
         patch("graph.write_report", return_value=(mock_report, 70)):

        final_state = run_research("Full graph test")
        assert final_state.status == "done"
        assert final_state.report is not None
        assert final_state.report.summary == "Summary of research."
        assert len(final_state.trace) >= 4  # planner, researcher, supervisor, writer
        assert final_state.budget.tokens_used == 180
