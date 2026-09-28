"""
api/main.py
===========
PHASE 5: "Ship a report endpoint with citations and a trace."

Endpoints:
  POST /research            -> starts a run, returns {run_id} immediately
  GET  /research/{run_id}   -> status, and the validated report once done
  GET  /research/{run_id}/trace -> full step-level trace log

THEORY on the two things this file is most careful about:

1. "POST /research returns a run id immediately. GET /research/{id}
   returns status and then the finished report. Stream progress events
   so the interface is not a four-minute spinner." We approximate the
   "not a spinner" requirement with a background task (FastAPI's
   BackgroundTasks) so the HTTP request returns instantly and the graph
   runs async to it; the client polls GET /research/{id} for status.
   (A true server-sent-events stream is a natural upgrade — noted in
   the README as a follow-on, not faked here.)

2. "Every claim in the report carries a citation, and each citation is
   validated against a real finding record before the report is
   returned." This is NOT optional and NOT decorative — see
   `_validate_citations()`. A report whose citations don't all resolve
   to real Finding ids is a bug we catch here, in code, rather than
   trusting the Writer prompt to have behaved.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel

from schemas import RunState
from storage.redis_store import (
    get_or_create,
    run_and_persist,
    load_state,
    save_state,
    list_recent_runs,
    ping,
)

app = FastAPI(title="Multi-Agent Research Assistant")

# ---------------------------------------------------------------------------
# SECURITY: optional API-key auth + per-IP rate limiting.
#
# THEORY: everything here is a guardrail on COST, not on secrets. A public
# deployment without these lets any visitor mint unbounded Gemini tokens and
# Tavily searches on your API keys. Both are opt-in-by-default-off so local
# development and the test suite stay zero-config; setting API_AUTH_TOKEN
# in the environment turns protection on for a deployment.
# ---------------------------------------------------------------------------

API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "").strip()

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# THEORY: sliding window over per-IP deques. A background-free approach —
# the active window prunes itself on each request — keeps this honest
# without a separate sweeper thread. In multi-worker deployments this
# becomes per-worker (Redis-based limiting would be the next step), which
# is still a strict improvement over none.
_rate_lock = threading.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, retry_after_seconds)."""
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        bucket = _rate_buckets[ip]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])) + 1
            return False, max(retry_after, 1)
        bucket.append(now)
        return True, 0


def enforce_public_guards(request: Request) -> None:
    """
    Shared dependency for run-mutating endpoints: rate limit first (cheap,
    pre-auth, don't leak whether an endpoint exists), then bearer-token
    check when API_AUTH_TOKEN is configured. 429 carries Retry-After.
    """
    allowed, retry_after = _check_rate_limit(_client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many research runs from this address. Please wait before retrying.",
            headers={"Retry-After": str(retry_after)},
        )
    if API_AUTH_TOKEN:
        auth = request.headers.get("Authorization", "")
        scheme, token = get_authorization_scheme_param(auth)
        if scheme.lower() != "bearer" or token != API_AUTH_TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: supply 'Authorization: Bearer <API_AUTH_TOKEN>'.",
                headers={"WWW-Authenticate": "Bearer"},
            )


# THEORY: BackgroundTasks gives no uniqueness guarantee — a client that
# retries POST /research (or opens two tabs) while a run is mid-flight
# would launch two identical graph executions and double-spend tokens.
# The Redis idempotency in get_or_create only covers re-entry AFTER a
# run finishes; this in-flight set covers re-entry DURING one. FastAPI
# runs sync endpoints on a threadpool, so the guard needs a real lock.
_INFLIGHT_RUNS: set[str] = set()
_inflight_lock = threading.Lock()


class ResearchRequest(BaseModel):
    question: str
    run_id: str | None = None  # allow client-supplied id for retry-safety


class ResearchAck(BaseModel):
    run_id: str
    status: str


def _validate_citations(state: RunState) -> list[str]:
    """
    Returns a list of problems found (empty list = all good). This is the
    concrete implementation of "each citation is validated against a real
    finding record before the report is returned."
    """
    if state.report is None:
        return ["no report present"]

    all_finding_ids = {f.id for r in state.results for f in r.findings}
    all_finding_by_id = {f.id: f for r in state.results for f in r.findings}

    problems = []
    for c in state.report.citations:
        finding = all_finding_by_id.get(c.finding_id)
        if finding is None:
            problems.append(f"citation references unknown finding_id={c.finding_id}")
            continue
        if finding.source.url.rstrip("/") != c.source_url.rstrip("/"):
            problems.append(
                f"citation source_url mismatch for finding_id={c.finding_id}: "
                f"citation says {c.source_url}, finding says {finding.source.url}"
            )
    return problems


def _run_in_background(run_id: str):
    with _inflight_lock:
        if run_id in _INFLIGHT_RUNS:
            return  # an identical execution is already running
        _INFLIGHT_RUNS.add(run_id)
    try:
        run_and_persist(run_id)
    except Exception as e:
        # THEORY: a background task exception disappears silently by
        # default in FastAPI. We persist the failure into the state itself
        # so GET /research/{id} can surface it, instead of the run
        # appearing to hang forever.
        state = load_state(run_id)
        if state is not None:
            state.status = "failed"
            state.finished_at = datetime.now(timezone.utc)
            state.trace.append(
                {
                    "node": "api",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "note": f"background run crashed: {e}",
                }
            )
            save_state(state)
    finally:
        with _inflight_lock:
            _INFLIGHT_RUNS.discard(run_id)


@app.post("/research", response_model=ResearchAck,
          dependencies=[Depends(enforce_public_guards)])
def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    state = get_or_create(req.run_id, req.question)
    if state.status not in ("done", "failed", "failed_citation_validation"):
        background_tasks.add_task(_run_in_background, state.run_id)
    return ResearchAck(run_id=state.run_id, status=state.status)


@app.post("/research/{run_id}/resume", response_model=ResearchAck,
          dependencies=[Depends(enforce_public_guards)])
def resume_research(run_id: str, background_tasks: BackgroundTasks):
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if state.status == "done":
        return ResearchAck(run_id=state.run_id, status=state.status)

    # THEORY: the wall-clock budget protects a single execution attempt,
    # not the run's entire lifetime. A run that failed at minute 2 and is
    # resumed at minute 30 must not inherit a dead clock — otherwise the
    # researcher skips every sub-question and the "resume" produces an
    # empty report (observed live during E2E verification). Explicit user
    # intent to continue = a fresh attempt = restart the clock.
    state.budget.started_at = datetime.now(timezone.utc)
    state.status = "planning" if state.plan is None else "researching"
    save_state(state)
    background_tasks.add_task(_run_in_background, state.run_id)
    return ResearchAck(run_id=state.run_id, status=state.status)


@app.get("/research/recent")
def get_recent_runs():
    runs = list_recent_runs(limit=15)
    result = []
    for r in runs:
        findings_count = sum(len(res.findings) for res in r.results) if r.results else 0
        # Measured end-to-end latency for terminal runs, live elapsed for
        # in-flight ones — makes the README's latency claims checkable.
        started_at = r.budget.started_at if r.budget and r.budget.started_at else None
        if started_at and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        duration_seconds = None
        if started_at:
            end = r.finished_at or datetime.now(timezone.utc)
            duration_seconds = round((end - started_at).total_seconds(), 1)
        result.append({
            "run_id": r.run_id,
            "question": r.question,
            "status": r.status,
            "tokens_used": r.budget.tokens_used if r.budget else 0,
            "searches_used": r.budget.searches_used if r.budget else 0,
            "findings_count": findings_count,
            "sub_questions_count": len(r.plan.sub_questions) if r.plan else 0,
            "started_at": started_at.isoformat() if started_at else None,
            "duration_seconds": duration_seconds,
            "has_report": r.report is not None,
        })
    return {"runs": result}


@app.get("/research/{run_id}")
def get_research(run_id: str):
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    response = {
        "run_id": state.run_id,
        "question": state.question,
        "status": state.status,
        "budget": state.budget.model_dump(),
        "plan": state.plan.model_dump() if state.plan else None,
        "results": [r.model_dump() for r in state.results] if state.results else [],
        "revision_count": state.revision_count,
        "trace": state.trace,
    }

    if state.status == "done" and state.report is not None:
        problems = _validate_citations(state)
        if problems:
            # THEORY: we do not silently ship a report with bad citations.
            # Surface it as a distinct status so it's impossible to miss
            # in a demo or an interview walkthrough.
            response["status"] = "failed_citation_validation"
            response["validation_problems"] = problems
        else:
            response["report"] = state.report.model_dump()

    return response


@app.get("/research/{run_id}/trace")
def get_trace(run_id: str):
    """
    THEORY: "That trace endpoint is what makes this project interview
    proof — when they ask how you would debug a bad answer, you open the
    trace instead of describing one." This just returns RunState.trace
    verbatim: node name, timestamp, note, cumulative tokens at that point.
    """
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {"run_id": run_id, "trace": state.trace}


@app.post("/research/{run_id}/corrupt-citation")
def corrupt_citation(run_id: str):
    """
    Diagnostic sandbox endpoint: deliberately mutates the citation finding_id in Redis
    to demonstrate citation validation catches fabricated/hallucinated citations in real-time.
    """
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if state.report is None or not state.report.citations:
        raise HTTPException(status_code=400, detail="Run has no report citations to corrupt")

    # Corrupt first citation finding_id
    state.report.citations[0].finding_id = "hallucinated-citation-id-666"
    save_state(state)
    
    problems = _validate_citations(state)
    return {
        "run_id": run_id,
        "status": "failed_citation_validation",
        "validation_problems": problems,
        "note": "Corrupted citation finding_id in Redis. Validated that code caught the hallucination immediately."
    }


@app.get("/health")
def health():
    # A server that answers while Redis is down is a lie for this app:
    # every state read/write would 500 mid-request. Surface the dependency.
    return {"ok": True, "redis": ping()}


@app.get("/research/{run_id}/stream")
async def stream_research(run_id: str):
    """
    Server-Sent Events stream of run state transitions.

    THEORY: replaces the frontend's blind 1.5s polling with push. The
    generator tail-polls the authoritative store (Redis) at 1s — cheap
    GETs of one key, no pub/sub fan-out to get wrong — and emits an
    `event: state` frame only when the persisted status actually changes,
    plus the final full state. SSE (not WebSockets) because the data
    flows one way, it traverses proxies cleanly, and EventSource just
    reconnects for free. Heartbeat comments keep intermediaries from
    reaping idle connections.
    """
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def event_stream():
        last_status = None
        last_payload = ""
        # Terminal statuses also include failed_citation_validation, which
        # only exists in API responses, not persisted state — so "done"
        # is emitted once with the full payload and the loop exits.
        terminal = {"done", "failed"}
        tick = 0
        while True:
            s = load_state(run_id)
            if s is None:
                yield "event: gone\ndata: {}\n\n"
                return
            payload = s.model_dump_json()
            if s.status != last_status or (s.status in terminal and payload != last_payload):
                last_status = s.status
                last_payload = payload
                yield f"event: state\ndata: {payload}\n\n"
                if s.status in terminal and s.report is not None:
                    return
                if s.status == "failed":
                    return
            tick += 1
            if tick % 15 == 0:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # don't let nginx buffer the stream
        },
    )


# Mount Static Files for the Frontend UI
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

