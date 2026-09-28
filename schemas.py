"""
schemas.py
==========
THEORY: Why this file is the most important one in the project.

A LangGraph node is just a function: state_in -> state_out. If "state" is a
loose dict of strings, every hand-off between agents is a guess: does the
Researcher put its findings under "results" or "findings"? Did the Planner
return 3 sub-questions or a paragraph describing 3 sub-questions?

Pydantic models turn that guess into a contract that fails LOUDLY (a
ValidationError) instead of SILENTLY (a KeyError three nodes later, or a
regex that quietly extracts nothing). This is the difference between
"a specialist" (prompt + tool set + output schema) and "a personality"
(a system prompt telling the model to role-play an expert) that your
project brief warns about.

Every agent in this project:
  1. Receives a typed Pydantic object (or plain args)
  2. Calls Claude with a prompt that asks for JSON matching a schema
  3. Parses the JSON response INTO that schema (validation happens here)
  4. Returns the typed object

If step 3 fails, that's a real, catchable error — not corrupted state
silently propagating downstream.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
import uuid


# ---------------------------------------------------------------------------
# PHASE 1 core contracts
# ---------------------------------------------------------------------------

class SubQuestion(BaseModel):
    """One decomposed piece of the user's research question."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    question: str
    rationale: str = Field(
        description="Why this sub-question matters to the overall research goal"
    )


class ResearchPlan(BaseModel):
    """Output of the Planner agent."""
    original_question: str
    sub_questions: list[SubQuestion]
    max_sub_questions: int = 5  # budget enforced at construction, see Phase 2

    def model_post_init(self, __context) -> None:
        # THEORY: "enforce budgets in the graph, not in prompts" (your PDF, Phase 2).
        # A prompt asking the model to "please only produce 5" is not a budget —
        # the model can and will ignore it. A budget is code that truncates or
        # rejects. We put a cheap version of that enforcement right here in the
        # schema itself, so it's impossible to construct an out-of-budget plan.
        if len(self.sub_questions) > self.max_sub_questions:
            self.sub_questions = self.sub_questions[: self.max_sub_questions]


class SourceRecord(BaseModel):
    """
    THEORY: "Findings without provenance are unusable in the writer step."
    This is the atomic unit of evidence. Every fact the Writer ever cites
    must trace back to one of these. No exceptions — if a finding doesn't
    have this, the report endpoint (Phase 5) will refuse to cite it.
    """
    url: str
    title: str | None = None
    snippet: str = Field(description="The exact extracted text supporting the claim")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Finding(BaseModel):
    """Output unit of the Researcher agent for ONE sub-question."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sub_question_id: str
    claim: str = Field(description="A single, specific, checkable statement")
    source: SourceRecord
    confidence: Literal["high", "medium", "low"] = "medium"


class SearchStatus(BaseModel):
    """
    THEORY (Phase 3): 'Handle each failure path explicitly: no results,
    rate limited, paywalled, timed out. Each returns a structured status
    the supervisor can act on rather than an exception that kills the run.'

    This is why SearchStatus exists as its own type instead of just
    raising exceptions everywhere. A failed search is DATA, not a crash.
    """
    ok: bool
    reason: Literal["ok", "no_results", "rate_limited", "paywalled", "timeout", "error"] = "ok"
    detail: str | None = None


class ResearchResult(BaseModel):
    """Everything the Researcher produced for one sub-question, including failure info."""
    sub_question_id: str
    status: SearchStatus
    findings: list[Finding] = []


class Citation(BaseModel):
    """A claim in the final report tied back to a validated Finding."""
    claim: str
    source_url: str
    finding_id: str  # references a Finding actually produced during the run


class ReportSection(BaseModel):
    """One titled section of the synthesized report body."""
    heading: str
    body: str


class Report(BaseModel):
    """Output of the Writer agent."""
    question: str
    summary: str
    # THEORY: sections/conclusion are optional-with-defaults so older runs
    # persisted in Redis (and minimal writer calls) still validate. The
    # frontend Studio renders them when present — before they were added
    # here, the model was asked for structure the schema then threw away.
    sections: list[ReportSection] = []
    conclusion: str | None = None
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Shared graph state (used starting Phase 2, defined now so schemas.py is
# the single file you import everywhere)
# ---------------------------------------------------------------------------

class RunBudget(BaseModel):
    """
    THEORY: budgets live in the state object and get decremented by the
    graph, never "suggested" via prompt text.
    """
    max_sub_questions: int = 5
    max_searches_per_sub_question: int = 3
    max_total_tokens: int = 150_000
    wall_clock_seconds: int = 240

    searches_used: int = 0
    tokens_used: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def searches_exhausted(self) -> bool:
        return self.searches_used >= (self.max_searches_per_sub_question * self.max_sub_questions)

    def tokens_exhausted(self) -> bool:
        return self.tokens_used >= self.max_total_tokens

    def time_exhausted(self) -> bool:
        now = datetime.now(timezone.utc)
        started = self.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (now - started).total_seconds() >= self.wall_clock_seconds


class RunState(BaseModel):
    """
    THEORY: this is "the shared state object" your PDF calls the single
    source of truth. Every LangGraph node reads a slice of this and writes
    a slice back. Nothing lives outside it — that's what makes Redis
    persistence (Phase 4) a one-line "serialize this object" operation.
    """
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: str
    status: Literal["planning", "researching", "reviewing", "writing", "done", "failed"] = "planning"
    plan: ResearchPlan | None = None
    results: list[ResearchResult] = []
    report: Report | None = None
    budget: RunBudget = Field(default_factory=RunBudget)
    trace: list[dict] = []  # Phase 5: step-level log, see storage/trace.py later
    revision_count: int = 0  # THEORY (Phase 2): "allow the reviewer to send
    # work back exactly once." This counter is what makes that a hard rule
    # instead of a hope — the Supervisor checks it before looping back.
    # Terminal timestamp (set when status becomes done/failed). Measured
    # end-to-end duration = finished_at - budget.started_at; exposed via
    # the API so the "35-55s" README claim is checkable, not folklore.
    finished_at: datetime | None = None
