"""
tests/test_phase1.py
====================
Tests Phase 1 agents in isolation: Planner, Researcher (stub), Writer, and Phase 1 chain.
"""

from unittest.mock import patch
from schemas import (
    ResearchPlan,
    SubQuestion,
    Finding,
    SourceRecord,
    Report,
    Citation,
)
from llm import LLMCallResult
from agents.planner import plan
from agents.researcher import research_sub_question, stub_search, _FindingsOnly
from agents.writer import write_report


def test_planner_agent():
    mock_plan = ResearchPlan(
        original_question="What are agent patterns?",
        sub_questions=[
            SubQuestion(id="sq1", question="What is supervisor pattern?", rationale="Core orchestration"),
            SubQuestion(id="sq2", question="What is peer pattern?", rationale="Decentralized orchestration"),
        ],
    )

    with patch("agents.planner.call_structured", return_value=LLMCallResult(mock_plan, 100, 50)):
        result_plan, tokens = plan("What are agent patterns?", max_sub_questions=5)
        assert len(result_plan.sub_questions) == 2
        assert result_plan.sub_questions[0].id == "sq1"
        assert tokens == 150


def test_researcher_with_stub_search():
    demo_source = SourceRecord(
        url="https://example.com/article",
        title="Example Source on sq1",
        snippet="(stub) Relevant snippet discussing: sq1",
    )
    demo_findings = [
        Finding(
            id="f1",
            sub_question_id="sq1",
            claim="Stub finding claim",
            source=demo_source,
            confidence="high",
        )
    ]
    mock_findings_only = _FindingsOnly(findings=demo_findings)

    with patch("agents.researcher.call_structured", return_value=LLMCallResult(mock_findings_only, 80, 40)):
        result, tokens = research_sub_question("sq1", "What are agent patterns?", search_backend=stub_search)
        assert result.status.ok is True
        assert result.status.reason == "ok"
        assert len(result.findings) == 1
        assert result.findings[0].sub_question_id == "sq1"
        assert tokens == 120


def test_writer_with_findings():
    demo_findings = [
        Finding(
            id="f1",
            sub_question_id="sq1",
            claim="Redis AOF provides persistence.",
            source=SourceRecord(url="https://redis.io", title="Redis", snippet="AOF logs writes..."),
            confidence="high",
        )
    ]
    mock_report = Report(
        question="How does Redis persist?",
        summary="Redis provides AOF and RDB persistence mechanisms.",
        citations=[
            Citation(
                claim="Redis AOF provides persistence.",
                source_url="https://redis.io",
                finding_id="f1",
            )
        ],
    )

    with patch("agents.writer.call_structured", return_value=LLMCallResult(mock_report, 120, 60)):
        report, tokens = write_report("How does Redis persist?", demo_findings)
        assert report.summary == "Redis provides AOF and RDB persistence mechanisms."
        assert len(report.citations) == 1
        assert report.citations[0].finding_id == "f1"
        assert tokens == 180


def test_writer_with_no_findings_returns_insufficient_evidence():
    report, tokens = write_report("Unanswerable question", [])
    assert "Insufficient evidence" in report.summary
    assert report.citations == []
    assert tokens == 0
