"""
storage/redis_store.py
=======================
PHASE 4: "Write graph state after every node: run id, plan, completed
sub-questions, findings so far, and cumulative token spend. Now a crashed
or redeployed process resumes instead of replaying — the same store gives
you idempotency, since re-entering a completed node returns the cached
result, and it powers a status endpoint so callers can see progress."

THEORY: notice this file does NOT change graph.py's node functions at
all. Persistence is bolted on as a wrapper around `graph.invoke`, using
LangGraph's own per-step streaming (`graph.stream(...)`) to write to
Redis after each node completes — exactly "after every node", not just
at the end. This is the same principle as the search-backend seam in
Phase 3: keep the thing that changes (how state is durably stored)
decoupled from the thing that doesn't (what each node computes).

Key design:
- Redis key: f"run:{run_id}" -> JSON-serialized RunState
- `get_or_create_run_id(run_id, question)`: if a run already exists in
  Redis, return its current state (resume); otherwise seed a fresh one.
- `run_and_persist(run_id)`: streams the graph, writing state to Redis
  after every single node transition — this IS the "resume where it
  stopped" guarantee, because if the process dies mid-run, the last
  completed node's output is already durable.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
import redis
from dotenv import load_dotenv
from schemas import RunState
from graph import build_graph

load_dotenv()

_redis_client = None


def _r() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    return _redis_client


def _key(run_id: str) -> str:
    return f"run:{run_id}"


def save_state(state: RunState) -> None:
    _r().set(_key(state.run_id), state.model_dump_json())


def load_state(run_id: str) -> RunState | None:
    raw = _r().get(_key(run_id))
    if raw is None:
        return None
    return RunState.model_validate(json.loads(raw))


def get_or_create(run_id: str | None, question: str) -> RunState:
    """
    THEORY: this is the idempotency entry point. Same run_id, same
    (or already-completed) state -> no wasted LLM/search calls. A caller
    that retries a POST /research request with the same run_id after a
    timeout will NOT trigger duplicate planning/research/writing.
    """
    if run_id:
        existing = load_state(run_id)
        if existing is not None:
            return existing
        state = RunState(run_id=run_id, question=question)
    else:
        state = RunState(question=question)
    save_state(state)
    return state


def run_and_persist(run_id: str) -> RunState:
    """
    Runs the graph starting from whatever is currently in Redis for
    run_id, persisting after EVERY node transition (not just at the end).

    THEORY: `graph.stream(..., stream_mode="values")` yields the full
    state dict after each node finishes. Writing to Redis on every yield
    is what makes a killed process resumable: the last value we saw
    before the crash is already sitting in Redis when the process comes
    back and calls this function again.
    """
    state = load_state(run_id)
    if state is None:
        raise ValueError(f"No run found for run_id={run_id}; call get_or_create first")

    if state.status == "done":
        return state  # THEORY: idempotent re-entry — already finished, no-op.

    graph = build_graph()
    final_state_dict = None
    for step_state_dict in graph.stream(state.model_dump(), stream_mode="values"):
        final_state_dict = step_state_dict
        save_state(RunState.model_validate(final_state_dict))

    return RunState.model_validate(final_state_dict)


def _get_run_timestamp(s: RunState) -> datetime:
    if s.budget and s.budget.started_at:
        dt = s.budget.started_at
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.min.replace(tzinfo=timezone.utc)


def list_recent_runs(limit: int = 15) -> list[RunState]:
    """
    Scans for 'run:*' keys and returns the most recent runs (parsed as RunState).
    """
    try:
        keys = _r().keys("run:*")
    except Exception:
        return []

    runs: list[RunState] = []
    for k in keys:
        raw = _r().get(k)
        if raw:
            try:
                runs.append(RunState.model_validate(json.loads(raw)))
            except Exception:
                continue

    runs.sort(
        key=_get_run_timestamp,
        reverse=True,
    )
    return runs[:limit]




if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How does Redis persistence differ from Postgres for agent state?"
    state = get_or_create(None, q)
    print(f"Created run {state.run_id}, status={state.status}")
    final = run_and_persist(state.run_id)
    print(f"\nFinal status: {final.status}")
    print(f"Report: {final.report.summary if final.report else '(none)'}")

    # demonstrate idempotent resume: calling again on a DONE run is a no-op
    again = run_and_persist(state.run_id)
    assert again.status == "done"
    print("\nRe-running on a completed run_id was a no-op (idempotency confirmed).")
