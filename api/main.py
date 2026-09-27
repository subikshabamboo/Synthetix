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
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from schemas import RunState
from storage.redis_store import (
    get_or_create,
    run_and_persist,
    load_state,
    save_state,
    list_recent_runs,
)

app = FastAPI(title="Multi-Agent Research Assistant")


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
            state.trace.append({"node": "api", "note": f"background run crashed: {e}"})
            save_state(state)


@app.post("/research", response_model=ResearchAck)
def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    state = get_or_create(req.run_id, req.question)
    if state.status not in ("done", "failed", "failed_citation_validation"):
        background_tasks.add_task(_run_in_background, state.run_id)
    return ResearchAck(run_id=state.run_id, status=state.status)


@app.post("/research/{run_id}/resume", response_model=ResearchAck)
def resume_research(run_id: str, background_tasks: BackgroundTasks):
    state = load_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if state.status == "done":
        return ResearchAck(run_id=state.run_id, status=state.status)
    
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
        result.append({
            "run_id": r.run_id,
            "question": r.question,
            "status": r.status,
            "tokens_used": r.budget.tokens_used if r.budget else 0,
            "searches_used": r.budget.searches_used if r.budget else 0,
            "findings_count": findings_count,
            "sub_questions_count": len(r.plan.sub_questions) if r.plan else 0,
            "started_at": r.budget.started_at.isoformat() if r.budget and r.budget.started_at else None,
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
    return {"ok": True}


# Mount Static Files for the Frontend UI
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

