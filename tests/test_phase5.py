"""
tests/test_phase5.py
====================
Tests Phase 5 FastAPI endpoints and citation validation:
- POST /research
- GET /research/{run_id}
- GET /research/{run_id}/trace
- GET /health
- Citation validation logic (success and detection of forged/mismatched citations)
"""

from fastapi.testclient import TestClient
from api.main import app, _validate_citations
from schemas import (
    RunState,
    ResearchResult,
    Finding,
    SourceRecord,
    SearchStatus,
    Report,
    Citation,
)
from storage.redis_store import save_state

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "redis" in body  # health surfaces the Redis dependency state


def test_citation_validation_logic_success():
    state = RunState(
        question="Citation test",
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True),
                findings=[
                    Finding(
                        id="finding_123",
                        sub_question_id="sq1",
                        claim="Valid claim",
                        source=SourceRecord(url="https://valid.com", snippet="Snippet"),
                    )
                ],
            )
        ],
        report=Report(
            question="Citation test",
            summary="Valid summary.",
            citations=[
                Citation(
                    claim="Valid claim",
                    source_url="https://valid.com",
                    finding_id="finding_123",
                )
            ],
        ),
    )
    problems = _validate_citations(state)
    assert problems == []


def test_citation_validation_logic_detects_corrupt_id_and_url():
    state = RunState(
        question="Citation test",
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True),
                findings=[
                    Finding(
                        id="real_id",
                        sub_question_id="sq1",
                        claim="Real claim",
                        source=SourceRecord(url="https://real.com", snippet="Snippet"),
                    )
                ],
            )
        ],
        report=Report(
            question="Citation test",
            summary="Corrupt summary.",
            citations=[
                # Corrupt finding ID
                Citation(claim="Fake", source_url="https://real.com", finding_id="fake_id"),
                # URL mismatch
                Citation(claim="Real", source_url="https://hacked.com", finding_id="real_id"),
            ],
        ),
    )
    problems = _validate_citations(state)
    assert len(problems) == 2
    assert "unknown finding_id=fake_id" in problems[0]
    assert "source_url mismatch" in problems[1]


def test_api_get_research_with_citation_validation():
    # Setup state in Redis with valid report
    state = RunState(
        question="API citation test",
        status="done",
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True),
                findings=[
                    Finding(
                        id="f1",
                        sub_question_id="sq1",
                        claim="Claim 1",
                        source=SourceRecord(url="https://example.com/1", snippet="S1"),
                    )
                ],
            )
        ],
        report=Report(
            question="API citation test",
            summary="API summary.",
            citations=[Citation(claim="Claim 1", source_url="https://example.com/1", finding_id="f1")],
        ),
    )
    save_state(state)

    resp = client.get(f"/research/{state.run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["report"]["summary"] == "API summary."

    # Now corrupt citation in Redis
    state.report.citations[0].finding_id = "nonexistent_id"
    save_state(state)

    resp_corrupt = client.get(f"/research/{state.run_id}")
    assert resp_corrupt.status_code == 200
    data_corrupt = resp_corrupt.json()
    assert data_corrupt["status"] == "failed_citation_validation"
    assert "validation_problems" in data_corrupt


def test_api_get_trace():
    state = RunState(question="Trace test")
    state.trace.append({"node": "planner", "note": "planned 3 subquestions"})
    save_state(state)

    resp = client.get(f"/research/{state.run_id}/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == state.run_id
    assert len(data["trace"]) == 1
    assert data["trace"][0]["node"] == "planner"


def test_api_get_recent_runs():
    state = RunState(question="Recent run test")
    save_state(state)

    resp = client.get("/research/recent")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert any(r["run_id"] == state.run_id for r in data["runs"])


def test_api_corrupt_citation_endpoint():
    state = RunState(
        question="Corrupt test",
        status="done",
        results=[
            ResearchResult(
                sub_question_id="sq1",
                status=SearchStatus(ok=True),
                findings=[
                    Finding(
                        id="valid_id",
                        sub_question_id="sq1",
                        claim="Real",
                        source=SourceRecord(url="https://valid.com", snippet="s"),
                    )
                ],
            )
        ],
        report=Report(
            question="Corrupt test",
            summary="Sum",
            citations=[Citation(claim="Real", source_url="https://valid.com", finding_id="valid_id")],
        ),
    )
    save_state(state)

    resp = client.post(f"/research/{state.run_id}/corrupt-citation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed_citation_validation"
    assert len(data["validation_problems"]) > 0


def test_frontend_static_serving():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Synthetix" in resp.text


