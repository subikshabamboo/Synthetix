"""
smoke_test_phase1.py
=====================
THEORY: This is Phase 1's "Done when" check, run manually (no LangGraph,
no Redis, no FastAPI yet — those are Phases 2-5). It wires Planner ->
Researcher (stub search) -> Writer by hand, in a straight line, so you
can see the full data flow and inspect every intermediate object before
any orchestration framework is in the picture.

Run:  python smoke_test_phase1.py "your research question"

If the Planner produces vague sub-questions, or the Writer cites things
not in the findings, THIS is where you catch and fix it — before Phase 2
adds a supervisor on top that would otherwise mask the same bug.
"""

import sys
import json
from agents.planner import plan
from agents.researcher import research_sub_question
from agents.writer import write_report


def run(question: str):
    print(f"\n=== PLANNING ===\nQuestion: {question}\n")
    research_plan, plan_tokens = plan(question)
    for sq in research_plan.sub_questions:
        print(f"  [{sq.id}] {sq.question}\n      why: {sq.rationale}")

    print(f"\n=== RESEARCHING ({len(research_plan.sub_questions)} sub-questions) ===")
    all_findings = []
    total_tokens = plan_tokens
    for sq in research_plan.sub_questions:
        result, toks = research_sub_question(sq.id, sq.question)
        total_tokens += toks
        print(f"  [{sq.id}] status={result.status.reason} findings={len(result.findings)}")
        all_findings.extend(result.findings)

    print(f"\n=== WRITING ===")
    report, write_tokens = write_report(question, all_findings)
    total_tokens += write_tokens

    print(f"\n--- REPORT ---\n{report.summary}\n")
    print("--- CITATIONS ---")
    for c in report.citations:
        print(f"  - {c.claim}\n    -> {c.source_url}")

    print(f"\n(total tokens used this run: {total_tokens})")
    return report


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What makes multi-agent LLM systems hard to put into production?"
    run(q)
