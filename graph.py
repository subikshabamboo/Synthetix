"""
graph.py
========
PHASE 2: "Model the system as a LangGraph state graph. Nodes are agents,
edges are supervisor routing decisions, and the shared state object is
the single source of truth."

THEORY, read this before the code:

- GraphState below is a TypedDict, NOT the Pydantic RunState from
  schemas.py. This is a deliberate, slightly annoying translation layer:
  LangGraph's StateGraph is built around plain dict-shaped state so it
  can do partial updates (each node returns only the keys it changed,
  and the graph shallow-merges them into the running state). Pydantic
  models are how WE keep type safety INSIDE a node's own logic; the
  TypedDict is the wire format the graph itself understands. We convert
  at the edges: dict -> RunState.model_validate(...) at the top of a
  node, RunState.model_dump() at the bottom.

- "After each researcher pass, the supervisor checks the output against
  the plan: is this sub-question answered, are sources missing, is
  another pass justified." That check lives in `node_supervisor` below,
  and its ONLY two outcomes are "send back to research, once" or
  "proceed to writer" — never an unbounded loop.

- "Enforce budgets in the graph rather than in prompts... A prompt asking
  the model to be efficient is not a budget." All budget checks below
  are plain Python `if` statements on `RunBudget`, checked in
  `node_supervisor` before deciding to route back for more research.
"""

from __future__ import annotations
from typing import TypedDict, Literal
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END

from schemas import RunState, ResearchResult, SearchStatus
from agents.planner import plan as run_planner
from agents.researcher import research_sub_question
from agents.writer import write_report


class GraphState(TypedDict, total=False):
    run_id: str
    question: str
    status: str
    plan: dict | None
    results: list[dict]
    report: dict | None
    budget: dict
    trace: list[dict]
    revision_count: int


def _to_state(g: GraphState) -> RunState:
    return RunState.model_validate(g)


def _log(state: RunState, node: str, note: str) -> None:
    # THEORY (Phase 5 preview): this is the step-level trace the brief
    # says is "what makes this project interview proof — when they ask
    # how you would debug a bad answer, you open the trace instead of
    # describing one." Built now so every phase after this one is
    # automatically traced, not bolted on later.
    state.trace.append(
        {
            "node": node,
            "at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "tokens_used_so_far": state.budget.tokens_used,
        }
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_planner(g: GraphState) -> dict:
    state = _to_state(g)
    research_plan, tokens = run_planner(state.question, state.budget.max_sub_questions)
    state.plan = research_plan
    state.budget.tokens_used += tokens
    state.status = "researching"
    _log(state, "planner", f"produced {len(research_plan.sub_questions)} sub-questions")
    return state.model_dump()


def node_researcher(g: GraphState) -> dict:
    state = _to_state(g)
    already_done = {r.sub_question_id for r in state.results if r.status.ok}

    for sq in state.plan.sub_questions:
        if sq.id in already_done:
            continue  # THEORY: idempotency — don't re-spend budget on a
            # sub-question that already succeeded in a prior pass.
        if state.budget.searches_exhausted() or state.budget.time_exhausted():
            _log(state, "researcher", f"budget exhausted, stopping before {sq.id}")
            break

        # THEORY: per-sub-question fault isolation. `research_sub_question`
        # already converts search failures into structured `SearchStatus`
        # data — but an LLM extraction failure raised *above* that seam
        # (planner proved 503 storms happen in practice) would escape this
        # node and kill the whole run. That is exactly the fragile failure
        # mode SearchStatus exists to prevent, so we honor the contract
        # here too: a failed sub-question becomes a failed ResearchResult,
        # the supervisor sees the gap and spends its one revision on it,
        # and the writer still ships a report from whatever evidence exists.
        try:
            result, tokens = research_sub_question(sq.id, sq.question)
        except Exception as e:
            result = ResearchResult(
                sub_question_id=sq.id,
                status=SearchStatus(ok=False, reason="error", detail=str(e)),
                findings=[],
            )
            tokens = 0
        state.budget.tokens_used += tokens
        state.budget.searches_used += 1

        # replace any prior (failed) result for this sub-question
        state.results = [r for r in state.results if r.sub_question_id != sq.id]
        state.results.append(result)
        _log(
            state,
            "researcher",
            f"{sq.id}: status={result.status.reason} findings={len(result.findings)}",
        )

    state.status = "reviewing"
    return state.model_dump()


def node_supervisor(g: GraphState) -> dict:
    """
    THEORY: this node makes NO tool/LLM calls. It's pure logic over typed
    state — "the supervisor checks the output against the plan" is a
    deterministic comparison, not another prompt. Cheap, fast, testable
    with plain asserts.
    """
    state = _to_state(g)

    covered_ids = {r.sub_question_id for r in state.results if r.status.ok and r.findings}
    plan_ids = {sq.id for sq in state.plan.sub_questions}
    missing = plan_ids - covered_ids

    budget_dead = (
        state.budget.tokens_exhausted()
        or state.budget.time_exhausted()
        or state.budget.searches_exhausted()
    )

    if missing and not budget_dead and state.revision_count < 1:
        state.revision_count += 1
        _log(
            state,
            "supervisor",
            f"coverage gap on {sorted(missing)}, sending back for one retry pass "
            f"(revision {state.revision_count}/1)",
        )
        state.status = "researching"
    else:
        if missing:
            _log(state, "supervisor", f"proceeding to writer with gaps: {sorted(missing)} "
                                       f"(budget_dead={budget_dead}, revisions_used={state.revision_count})")
        else:
            _log(state, "supervisor", "full coverage, proceeding to writer")
        state.status = "writing"

    return state.model_dump()


def node_writer(g: GraphState) -> dict:
    state = _to_state(g)
    all_findings = [f for r in state.results for f in r.findings]
    report, tokens = write_report(state.question, all_findings)
    state.budget.tokens_used += tokens
    state.report = report
    state.status = "done"
    _log(state, "writer", f"report written with {len(report.citations)} citations")
    return state.model_dump()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_supervisor(g: GraphState) -> Literal["researcher", "writer"]:
    return "researcher" if g["status"] == "researching" else "writer"


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("planner", node_planner)
    builder.add_node("researcher", node_researcher)
    builder.add_node("supervisor", node_supervisor)
    builder.add_node("writer", node_writer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_after_supervisor, {"researcher": "researcher", "writer": "writer"}
    )
    builder.add_edge("writer", END)
    return builder.compile()


def run_research(question: str) -> RunState:
    """Convenience wrapper used by the smoke test and (later) the API layer."""
    graph = build_graph()
    initial = RunState(question=question)
    final_dict = graph.invoke(initial.model_dump())
    return RunState.model_validate(final_dict)


if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "What are the key risks of running multi-agent LLM systems in production?"
    final_state = run_research(q)
    print(f"\n=== STATUS: {final_state.status} ===")
    print(f"revisions used: {final_state.revision_count}")
    print(f"tokens used: {final_state.budget.tokens_used}, searches used: {final_state.budget.searches_used}")
    print("\n--- REPORT ---")
    print(final_state.report.summary)
    print("\n--- CITATIONS ---")
    for c in final_state.report.citations:
        print(f"  - {c.claim}\n    -> {c.source_url}")
    print("\n--- TRACE ---")
    for step in final_state.trace:
        print(f"  [{step['node']}] {step['note']}")
