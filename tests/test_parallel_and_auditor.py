"""
tests/test_parallel_and_auditor.py
==================================
Tests for the two spec stretch goals:

1. PARALLEL RESEARCHER FAN-OUT: after planning, one `Send` per
   sub-question launches concurrent research workers; reducers merge
   their deltas; the supervisor books cumulative spend once and re-fans
   only still-missing sub-questions on its single revision pass.
2. CONTRADICTION DETECTION: `node_auditor` cross-checks findings from
   DIFFERENT sub-questions with pure logic (negation pairs, numeric
   disagreement, opposing comparatives) — no LLM, no tokens.
"""

from unittest.mock import patch

from graph import (
    build_graph,
    node_auditor,
    node_research_worker,
    node_supervisor,
    route_after_planner,
)
from schemas import (
    Citation,
    Finding,
    Report,
    ResearchPlan,
    ResearchResult,
    RunState,
    SearchStatus,
    SourceRecord,
    SubQuestion,
)


def _sq(sid: str, q: str) -> SubQuestion:
    return SubQuestion(id=sid, question=q, rationale=f"rationale for {sid}")


def _plan(*sqs: SubQuestion) -> ResearchPlan:
    return ResearchPlan(original_question="base question", sub_questions=list(sqs))


def _ok_result(sid: str, findings: list[Finding]) -> ResearchResult:
    return ResearchResult(
        sub_question_id=sid,
        status=SearchStatus(ok=True, reason="ok"),
        findings=findings,
    )


def _finding(fid: str, sid: str, claim: str, url: str = "https://example.com") -> Finding:
    return Finding(
        id=fid,
        sub_question_id=sid,
        claim=claim,
        source=SourceRecord(url=url, snippet="s"),
    )


def _state_with_plan(sqs: list[SubQuestion], **kw) -> RunState:
    return RunState(question="base question", plan=_plan(*sqs), **kw)


# ---------------------------------------------------------------------------
# Fan-out routing
# ---------------------------------------------------------------------------


def test_router_emits_one_send_per_sub_question():
    state = _state_with_plan([_sq("sq1", "Q1"), _sq("sq2", "Q2"), _sq("sq3", "Q3")])
    sends = route_after_planner(state.model_dump())

    assert len(sends) == 3
    assert all(s.node == "research_worker" for s in sends)
    assert {s.arg["sub_question"]["id"] for s in sends} == {"sq1", "sq2", "sq3"}
    # Each worker carries the shared run state for its own idempotency check.
    assert all(s.arg["run_state"]["question"] == "base question" for s in sends)


def test_router_skips_research_entirely_when_budget_dead():
    state = _state_with_plan([_sq("sq1", "Q1")])
    state.budget.tokens_used = state.budget.max_total_tokens + 1
    sends = route_after_planner(state.model_dump())

    assert len(sends) == 1
    assert sends[0].node == "supervisor"  # straight to the join, gaps documented


def test_worker_delta_contract():
    """Worker returns ONLY its own result + spend deltas — never full lists."""
    state = _state_with_plan([_sq("sq1", "Q1")])
    mock_result = _ok_result("sq1", [_finding("f1", "sq1", "claim one")])

    with patch(
        "graph.research_sub_question",
        return_value=(mock_result, 25),
    ):
        out = node_research_worker({
            "sub_question": _sq("sq1", "Q1").model_dump(),
            "run_state": state.model_dump(),
        })

    assert out["results"] == [mock_result.model_dump()]
    assert out["_worker_tokens"] == 25
    assert out["_worker_searches"] == 1
    assert "plan" not in out and "budget" not in out  # delta, not snapshot


def test_worker_skips_already_covered_sub_question():
    """Idempotency: a re-fanned worker must not re-spend on a covered SQ."""
    state = _state_with_plan(
        [_sq("sq1", "Q1")],
        results=[_ok_result("sq1", [_finding("f1", "sq1", "claim one")])],
    )

    calls = []
    with patch(
        "graph.research_sub_question",
        side_effect=lambda *a, **k: calls.append(a) or (_ok_result("sq1", []), 99),
    ):
        out = node_research_worker({
            "sub_question": _sq("sq1", "Q1").model_dump(),
            "run_state": state.model_dump(),
        })

    assert calls == []  # research function never invoked
    assert "results" not in out  # no duplicate result delta
    assert out["trace"][0]["note"].endswith("already covered, skipping")


# ---------------------------------------------------------------------------
# Supervisor as join: budget booking + re-fan only what's missing
# ---------------------------------------------------------------------------


def test_supervisor_books_cumulative_worker_spend_once():
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "claim a")]),
            _ok_result("sq2", [_finding("f2", "sq2", "claim b")]),
        ],
    )
    g = state.model_dump()
    g["_worker_tokens"] = 40
    g["_worker_searches"] = 2

    out = node_supervisor(g)
    assert out["budget"]["tokens_used"] == 40  # booked exactly once
    assert out["budget"]["searches_used"] == 2
    assert out["status"] == "auditing"


def test_refan_targets_only_missing_sub_questions():
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2"), _sq("sq3", "Q3")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "claim a")]),
            ResearchResult(
                sub_question_id="sq2",
                status=SearchStatus(ok=False, reason="error", detail="boom"),
                findings=[],
            ),
        ],
    )
    out = node_supervisor(state.model_dump())

    assert out["status"] == "researching"
    assert out["revision_count"] == 1
    assert [sq["id"] for sq in out["_refan"]] == ["sq2", "sq3"]  # only gaps


# ---------------------------------------------------------------------------
# Full graph: parallel execution + reducer merge (map side verified end to end)
# ---------------------------------------------------------------------------


def test_full_graph_fans_out_and_merges_three_workers():
    """Three sub-questions -> three worker deltas merged; supervisor sees all."""
    plan = _plan(_sq("sq1", "Q1"), _sq("sq2", "Q2"), _sq("sq3", "Q3"))
    report = Report(question="base question", summary="S", citations=[
        Citation(claim="c", source_url="https://x.com", finding_id="f1")
    ])

    seen_sq_ids = []

    def fake_research(sq_id, question, *a, **k):
        seen_sq_ids.append(sq_id)
        return (_ok_result(sq_id, [_finding(f"f-{sq_id}", sq_id, f"claim {sq_id}")]), 10)

    with patch("graph.run_planner", return_value=(plan, 5)), \
         patch("graph.research_sub_question", side_effect=fake_research), \
         patch("graph.write_report", return_value=(report, 7)):

        final = build_graph().invoke(RunState(question="base question").model_dump())

    assert sorted(seen_sq_ids) == ["sq1", "sq2", "sq3"]  # every SQ got a worker
    assert {r["sub_question_id"] for r in final["results"]} == {"sq1", "sq2", "sq3"}
    assert final["budget"]["tokens_used"] == 42  # 5 + 3×10 + 7, booked once
    assert final["budget"]["searches_used"] == 3
    assert final["status"] == "done"
    assert any("full coverage" in t["note"] for t in final["trace"])


# ---------------------------------------------------------------------------
# Auditor: contradiction detection
# ---------------------------------------------------------------------------


def test_auditor_detects_negation_contradiction_across_sources():
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "Solar panels generate electricity during daylight")]),
            _ok_result("sq2", [_finding("f2", "sq2", "Solar panels do not generate electricity during daylight")]),
        ],
    )
    out = node_auditor(state.model_dump())

    assert len(out["conflicts"]) == 1
    assert out["conflicts"][0]["reason"] == "negation"
    assert out["conflicts"][0]["claim_a"]["finding_id"] == "f1"
    assert out["status"] == "writing"


def test_auditor_detects_numeric_disagreement():
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "Wind capacity reached 1000 gigawatts globally")]),
            _ok_result("sq2", [_finding("f2", "sq2", "Global wind capacity reached 740 gigawatts")]),
        ],
    )
    out = node_auditor(state.model_dump())
    assert [c["reason"] for c in out["conflicts"]] == ["numeric_disagreement"]


def test_auditor_detects_opposing_comparative():
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "Postgres reads are faster than Mongo reads")]),
            _ok_result("sq2", [_finding("f2", "sq2", "Postgres reads are slower than Mongo reads")]),
        ],
    )
    out = node_auditor(state.model_dump())
    assert [c["reason"] for c in out["conflicts"]] == ["opposing_comparative"]


def test_auditor_skips_same_sub_question_pairs():
    """Same-SQ twins share one source context — never paired as conflict;
    an unrelated cross-source finding must not collide with either."""
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [
                _finding("f1", "sq1", "The system processes 100 requests per second"),
                _finding("f1b", "sq1", "The system does not process 100 requests per second"),
            ]),
            _ok_result("sq2", [_finding("f2", "sq2", "Kubernetes orchestrates containers across clusters")]),
        ],
    )
    out = node_auditor(state.model_dump())
    # The ONLY lexically-overlapping pair here is f1×f1b — same SQ, skipped.
    assert out["conflicts"] == []
    assert "no contradictions" in out["trace"][0]["note"]


def test_auditor_accepts_numbers_within_tolerance():
    """100 vs 105 requests/sec is measurement noise, not a contradiction."""
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "The system processes 100 requests per second")]),
            _ok_result("sq2", [_finding("f2", "sq2", "The system processes 105 requests per second")]),
        ],
    )
    out = node_auditor(state.model_dump())
    assert out["conflicts"] == []


def test_auditor_sees_through_hedged_claims():
    """Hedges are stripped before comparison — softening a claim doesn't
    let a source dodge the auditor."""
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "The battery lasts roughly 400 miles")]),
            _ok_result("sq2", [_finding("f2", "sq2", "The battery lasts 300 miles")]),
        ],
    )
    out = node_auditor(state.model_dump())
    assert [c["reason"] for c in out["conflicts"]] == ["numeric_disagreement"]


def test_auditor_two_negated_claims_agree_not_conflict():
    """Two denials of the same thing agree — no negation-pair flag."""
    state = _state_with_plan(
        [_sq("sq1", "Q1"), _sq("sq2", "Q2")],
        results=[
            _ok_result("sq1", [_finding("f1", "sq1", "This database does not support distributed transactions")]),
            _ok_result("sq2", [_finding("f2", "sq2", "This database never supports distributed transactions")]),
        ],
    )
    out = node_auditor(state.model_dump())
    assert out["conflicts"] == []


def test_auditor_sets_writing_status_and_counts_findings():
    state = _state_with_plan(
        [_sq("sq1", "Q1")],
        results=[_ok_result("sq1", [_finding("f1", "sq1", "a claim about something")])],
    )
    out = node_auditor(state.model_dump())
    assert out["status"] == "writing"
    assert out["conflicts"] == []
    assert "across 1 findings" in out["trace"][0]["note"]
