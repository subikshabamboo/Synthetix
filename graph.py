"""
graph.py
========
PHASE 2 (and the spec's stretch goals): "Model the system as a LangGraph
state graph. Nodes are agents, edges are supervisor routing decisions, and
the shared state object is the single source of truth."

THEORY, read this before the code:

- GraphState below is a TypedDict, NOT the Pydantic RunState from
  schemas.py. This is a deliberate, slightly annoying translation layer:
  LangGraph's StateGraph is built around plain dict-shaped state so it
  can do partial updates (each node returns only the keys it changed,
  and the graph shallow-merges them into the running state). Pydantic
  models are how WE keep type safety INSIDE a node's own logic; the
  TypedDict is the wire format the graph itself understands.

- NODE CONTRACT (discipline that makes parallelism safe): every node
  returns a DELTA — only the keys it changed, and for reducer channels
  (results / trace / worker spend) only the NEW entries. Returning a
  full state snapshot through an `add`-reducer channel would append the
  whole old list to itself (classic LangGraph duplication bug). Nodes
  that want to *read* the full state convert via RunState.model_validate;
  they just never *write* the full lists back.

- STRETCH GOAL, "fan out researchers in parallel and join results":
  genuine map-reduce. After planning, a router emits one
  `Send("research_worker", ...)` per sub-question — N workers execute
  concurrently (search + LLM extraction are I/O-bound, so wall-clock is
  roughly the SLOWEST sub-question, not the sum). Workers return deltas;
  reducers merge them; the supervisor is the JOIN node: it books
  cumulative worker spend into the authoritative budget once, checks
  coverage, and on revision re-fans ONLY still-missing sub-questions.

- STRETCH GOAL, "detect contradictions between sources": implemented as
  `node_auditor` — a PURE-LOGIC, zero-LLM pairwise contradiction check
  over finding claims (negation pairs, incompatible numbers about the
  same quantity, opposing comparatives between the same entities).
  Deterministic code beats a "contradiction-detector prompt" for the same
  reason the supervisor is code: testable with asserts, no tokens, no
  hallucinated conflicts.

- "Enforce budgets in the graph rather than in prompts." All budget
  checks are plain Python `if` statements on `RunBudget`, checked in the
  router and the supervisor before any further research happens.
"""

from __future__ import annotations
import re
from typing import TypedDict, Literal, Annotated
from datetime import datetime, timezone
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from schemas import (
    RunState,
    ResearchResult,
    SearchStatus,
    Finding,
    SubQuestion,
)
from agents.planner import plan as run_planner
from agents.researcher import research_sub_question
from agents.writer import write_report


class GraphState(TypedDict, total=False):
    run_id: str
    question: str
    status: str
    plan: dict | None
    # Reducer channels: concurrent writers append; last-writer-wins would
    # silently drop all but one worker's output.
    results: Annotated[list[dict], add]
    trace: Annotated[list[dict], add]
    # Worker spend deltas, booked into RunBudget by the supervisor (join).
    _worker_tokens: Annotated[int, add]
    _worker_searches: Annotated[int, add]
    # Plain channels: exactly one node writes each per step.
    conflicts: list[dict]
    report: dict | None
    budget: dict
    revision_count: int
    _refan: list[dict]


def _to_state(g: GraphState) -> RunState:
    return RunState.model_validate(g)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_entry(node: str, note: str, tokens_so_far: int | None) -> dict:
    return {
        "node": node,
        "at": _now_iso(),
        "note": note,
        "tokens_used_so_far": tokens_so_far,
    }


# ---------------------------------------------------------------------------
# Nodes — ALL returns are deltas (see NODE CONTRACT above)
# ---------------------------------------------------------------------------

def node_planner(g: GraphState) -> dict:
    state = _to_state(g)
    research_plan, tokens = run_planner(state.question, state.budget.max_sub_questions)
    budget = state.budget.model_copy(update={"tokens_used": state.budget.tokens_used + tokens})
    return {
        "plan": research_plan.model_dump(),
        "budget": budget.model_dump(),
        "status": "researching",
        "trace": [_trace_entry("planner", f"produced {len(research_plan.sub_questions)} sub-questions", budget.tokens_used)],
    }


# ---------------------------------------------------------------------------
# Parallel researcher fan-out (map-reduce)
# ---------------------------------------------------------------------------

def route_after_planner(g: GraphState) -> list[Send]:
    """
    Router: emit one research worker per sub-question. Budget belongs
    here (single reader of shared state), not inside the workers. If the
    budget is already dead, route straight to the supervisor with a trace
    note — the supervisor's 'proceed with gaps' path handles it, which
    keeps 'dead budget ⇒ writer with documented gaps' true even before
    any research runs.
    """
    state = _to_state(g)
    if (state.budget.searches_exhausted() or state.budget.time_exhausted()
            or state.budget.tokens_exhausted()):
        return [Send("supervisor", {
            **g,
            "trace": g.get("trace", []) + [
                _trace_entry("researcher", "budget exhausted before any sub-question; skipping research", state.budget.tokens_used)
            ],
        })]
    return [
        Send("research_worker", {"sub_question": sq.model_dump(), "run_state": g})
        for sq in state.plan.sub_questions
    ]


def node_research_worker(payload: dict) -> dict:
    """
    One worker = one sub-question, researched start to finish.

    Contract: returns a PARTIAL state update — its own result, its own
    trace entry, its own spend deltas — never the whole state. Reducers
    merge concurrent outputs; the supervisor (single join node) books
    cumulative spend and derives status deterministically.
    """
    sq = SubQuestion.model_validate(payload["sub_question"])
    state = RunState.model_validate(payload["run_state"])

    # Idempotency: don't re-spend on a sub-question that already succeeded
    # in a prior pass (workers re-run when the supervisor loops a revision).
    if any(r.sub_question_id == sq.id and r.status.ok for r in state.results):
        return {"trace": [_trace_entry("researcher", f"{sq.id}: already covered, skipping", None)]}

    # THEORY: per-sub-question fault isolation. `research_sub_question`
    # converts search failures into structured SearchStatus data — but an
    # LLM extraction failure raised above that seam (503 storms happen in
    # practice) must ALSO be a structured error result, not a dead run.
    try:
        result, tokens = research_sub_question(sq.id, sq.question)
    except Exception as e:
        result = ResearchResult(
            sub_question_id=sq.id,
            status=SearchStatus(ok=False, reason="error", detail=str(e)),
            findings=[],
        )
        tokens = 0

    return {
        "results": [result.model_dump()],
        "_worker_tokens": tokens,
        "_worker_searches": 1,
        "trace": [_trace_entry(
            "researcher",
            f"{sq.id}: status={result.status.reason} findings={len(result.findings)}",
            None,
        )],
    }


# ---------------------------------------------------------------------------
# Supervisor (join point) + auditor + writer
# ---------------------------------------------------------------------------

def node_supervisor(g: GraphState) -> dict:
    """
    THEORY: this node makes NO tool/LLM calls. It's pure logic over typed
    state — "the supervisor checks the output against the plan" is a
    deterministic comparison, not another prompt. Cheap, fast, testable
    with plain asserts.

    It is also the JOIN point of the parallel fan-out: the single node
    after all workers complete, so it sees the fully merged state. Here it
    books the workers' cumulative spend into the authoritative budget —
    once, in one deterministic place — and on revision re-fans ONLY the
    still-missing sub-questions.
    """
    state = _to_state(g)

    budget = state.budget.model_copy(update={
        "tokens_used": state.budget.tokens_used + g.get("_worker_tokens", 0),
        "searches_used": state.budget.searches_used + g.get("_worker_searches", 0),
    })

    covered_ids = {r.sub_question_id for r in state.results if r.status.ok and r.findings}
    plan_ids = {sq.id for sq in state.plan.sub_questions}
    missing_ids = plan_ids - covered_ids

    budget_dead = (
        budget.tokens_exhausted()
        or budget.time_exhausted()
        or budget.searches_exhausted()
    )

    base = {"budget": budget.model_dump()}

    if missing_ids and not budget_dead and state.revision_count < 1:
        note = (
            f"coverage gap on {sorted(missing_ids)}, sending back for one retry pass "
            f"(revision {state.revision_count + 1}/1)"
        )
        return {
            **base,
            "status": "researching",
            "revision_count": state.revision_count + 1,
            "trace": [_trace_entry("supervisor", note, budget.tokens_used)],
            "_refan": [sq.model_dump() for sq in state.plan.sub_questions if sq.id in missing_ids],
        }

    if missing_ids:
        note = (f"proceeding to writer with gaps: {sorted(missing_ids)} "
                f"(budget_dead={budget_dead}, revisions_used={state.revision_count})")
    else:
        note = "full coverage, proceeding to auditor"
    return {
        **base,
        "status": "auditing",
        "revision_count": state.revision_count,
        "trace": [_trace_entry("supervisor", note, budget.tokens_used)],
    }


def route_after_supervisor(g: GraphState) -> Literal["refan", "auditor"]:
    return "refan" if g.get("_refan") else "auditor"


def node_refan(g: GraphState) -> list[Send]:
    """Re-emit workers for just the missing sub-questions (revision pass)."""
    missing = g.get("_refan", [])
    return [Send("research_worker", {"sub_question": sq, "run_state": g}) for sq in missing]


# ---------------------------------------------------------------------------
# Contradiction auditor (stretch goal: "detect contradictions between
# sources") — pure logic, no LLM, no tokens.
# ---------------------------------------------------------------------------

_NUM_TOKENS = r"[-+]?\d[\d,]*(?:\.\d+)?"
_NEGATIONS = (" not ", " no ", " never ", " cannot ", " can't ", " doesn't ",
              " does not ", " isn't ", " is not ", " without ")
_HEDGES = ("may", "might", "could", "possibly", "likely", "about",
           "approximately", "around", "roughly", "estimated", "~")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _has_negation(text: str) -> bool:
    padded = f" {_norm(text)} "
    return any(n in padded for n in _NEGATIONS)


def _strip_hedges(text: str) -> str:
    t = _norm(text)
    for h in _HEDGES:
        t = t.replace(f" {h} ", " ")
    return " ".join(t.split())


def _extract_numbers(text: str) -> set[float]:
    out: set[float] = set()
    for m in re.finditer(_NUM_TOKENS, text):
        tok = m.group(0).replace(",", "").rstrip("%")
        try:
            out.add(float(tok))
        except ValueError:
            continue
    return out


def _comparative(text: str) -> str | None:
    t = " " + text.lower() + " "
    for comp in ("faster than", "slower than", "better than", "worse than",
                 "more than", "less than", "cheaper than", "outperforms",
                 "underperforms"):
        if comp in t:
            return comp
    return None


_OPPOSITE_PAIRS = {
    frozenset({"faster than", "slower than"}),
    frozenset({"better than", "worse than"}),
    frozenset({"more than", "less than"}),
}


def _comparative_is_reversed(a: str, b: str) -> bool:
    ca, cb = _comparative(a), _comparative(b)
    if not ca or not cb or ca == cb:
        return False
    return frozenset({ca, cb}) in _OPPOSITE_PAIRS


def _lexical_overlap(a: str, b: str, threshold: float) -> bool:
    wa = {w for w in a.split() if len(w) > 4}
    wb = {w for w in b.split() if len(w) > 4}
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _contradicts(a: Finding, b: Finding) -> str | None:
    """
    Return a reason string if findings a and b conflict, else None.
    Deliberately conservative: flag only structurally-grounded conflicts
    (overlapping subject vocabulary + opposing predicate). Topical overlap
    alone never flags — false positives would poison trust in the auditor.
    """
    ca, cb = _strip_hedges(a.claim), _strip_hedges(b.claim)
    if not ca or not cb:
        return None

    # 1. Negation pair with high lexical overlap ("X is safe" vs "X is not safe").
    if _has_negation(ca) != _has_negation(cb) and _lexical_overlap(ca, cb, 0.5):
        return "negation"

    # 2. Same quantity, incompatible numbers (no pair within 15%).
    na, nb = _extract_numbers(ca), _extract_numbers(cb)
    if na and nb:
        compatible = bool(na & nb) or any(
            abs(x - y) / max(abs(y), 1e-9) <= 0.15 for x in na for y in nb
        )
        if not compatible and _lexical_overlap(ca, cb, 0.4):
            return "numeric_disagreement"

    # 3. Opposing comparatives about the same two entities — overlap is
    # required, otherwise "GPU is faster than CPU" and "Postgres is slower
    # than Redis" would falsely pair as a contradiction.
    if _comparative_is_reversed(ca, cb) and _lexical_overlap(ca, cb, 0.3):
        return "opposing_comparative"

    return None


def node_auditor(g: GraphState) -> dict:
    """
    Cross-checks every pair of findings from DIFFERENT sub-questions
    (same-sub-question pairs share one source context and rarely conflict).
    Deterministic, zero-LLM: costs nothing, cannot hallucinate conflicts,
    unit-testable with plain asserts. Conflicting findings still flow to
    the writer (the report stays complete) — each conflict is recorded in
    state and surfaced through the report endpoint and UI so the reader
    can weigh the disagreement themselves.
    """
    state = _to_state(g)

    findings = [f for r in state.results if r.status.ok for f in r.findings]
    conflicts: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, a in enumerate(findings):
        for b in findings[i + 1:]:
            if a.sub_question_id == b.sub_question_id:
                continue
            key = tuple(sorted((a.id, b.id)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            reason = _contradicts(a, b)
            if reason:
                conflicts.append({
                    "reason": reason,
                    "claim_a": {"finding_id": a.id, "sub_question_id": a.sub_question_id,
                                "claim": a.claim, "source_url": a.source.url},
                    "claim_b": {"finding_id": b.id, "sub_question_id": b.sub_question_id,
                                "claim": b.claim, "source_url": b.source.url},
                })

    if conflicts:
        note = f"{len(conflicts)} potential contradiction(s) detected across sources"
    else:
        note = f"no contradictions across {len(findings)} findings"
    return {
        "conflicts": conflicts,
        "status": "writing",
        "trace": [_trace_entry("auditor", note, state.budget.tokens_used)],
    }


def node_writer(g: GraphState) -> dict:
    state = _to_state(g)

    # Dedupe by finding id: resumed runs can hold results from more than
    # one pass; the writer must cite each evidence record exactly once.
    seen: set[str] = set()
    all_findings: list[Finding] = []
    for r in state.results:
        for f in r.findings:
            if f.id not in seen:
                seen.add(f.id)
                all_findings.append(f)

    report, tokens = write_report(state.question, all_findings)
    budget = state.budget.model_copy(update={"tokens_used": state.budget.tokens_used + tokens})
    return {
        "report": report.model_dump(),
        "budget": budget.model_dump(),
        "status": "done",
        "trace": [_trace_entry("writer", f"report written with {len(report.citations)} citations", budget.tokens_used)],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("planner", node_planner)
    builder.add_node("research_worker", node_research_worker)
    builder.add_node("supervisor", node_supervisor)
    builder.add_node("refan", node_refan)
    builder.add_node("auditor", node_auditor)
    builder.add_node("writer", node_writer)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"research_worker": "research_worker", "supervisor": "supervisor"},
    )
    # All workers converge on the supervisor (the join point).
    builder.add_edge("research_worker", "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_after_supervisor, {"refan": "refan", "auditor": "auditor"}
    )
    builder.add_edge("refan", "research_worker")
    builder.add_edge("auditor", "writer")
    builder.add_edge("writer", END)
    return builder.compile()


def run_research(question: str) -> RunState:
    """Convenience wrapper used by the smoke test and (later) the API layer."""
    graph = build_graph()
    initial = RunState(question=question)
    final_dict = graph.invoke(initial.model_dump())
    return RunState.model_validate(final_dict)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the key risks of running multi-agent LLM systems in production?"
    final_state = run_research(q)
    print(f"\n=== STATUS: {final_state.status} ===")
    print(f"revisions used: {final_state.revision_count}")
    print(f"tokens used: {final_state.budget.tokens_used}, searches used: {final_state.budget.searches_used}")
    print(f"conflicts: {len(final_state.conflicts)}")
    print("\n--- REPORT ---")
    print(final_state.report.summary)
    print("\n--- CITATIONS ---")
    for c in final_state.report.citations:
        print(f"  - {c.claim}\n    -> {c.source_url}")
    print("\n--- TRACE ---")
    for step in final_state.trace:
        print(f"  [{step['node']}] {step['note']}")
