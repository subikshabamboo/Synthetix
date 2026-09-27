"""
agents/writer.py
=================
THEORY: "Writer turns findings into a report with citations."

Two important design decisions here, both defensive:

1. The Writer NEVER invents a finding_id. We hand it the exact list of
   findings with their ids, and its ONLY job is to select and phrase —
   not to introduce new "facts". This is what makes Phase 5's citation
   validation ("each citation is validated against a real finding record
   before the report is returned") actually meaningful instead of
   theater: the model was already constrained to only reference real ids.

2. If there are zero findings, we don't call the LLM at all — we return
   an explicit "insufficient evidence" report. A model asked to write a
   report from nothing will happily hallucinate one.
"""

from schemas import Finding, Report, Citation
from llm import call_structured

SYSTEM_PROMPT = """You are a rigorous report-writing function. You will be given a \
research question and a list of verified findings, each with an exact finding_id, a claim, \
and a source_url.

Your task:
1. Write an executive summary that synthesizes the findings into a clear, direct answer.
2. Structure detailed sections if appropriate.
3. Produce a citations list: for each claim you cite, you MUST copy the EXACT finding_id and the EXACT matching source_url from the findings list provided below.
4. Invariant: NEVER invent a finding_id. NEVER pair a finding_id with a different source_url than the one listed for that finding_id.
"""


def write_report(question: str, findings: list[Finding]) -> tuple[Report, int]:
    if not findings:
        return (
            Report(
                question=question,
                summary=(
                    "Insufficient evidence was gathered to answer this question. "
                    "No search results produced usable findings."
                ),
                citations=[],
            ),
            0,
        )

    findings_block = "\n".join(
        f"- finding_id: {f.id}\n"
        f"  claim: {f.claim}\n"
        f"  source_url: {f.source.url}\n"
        f"  confidence: {f.confidence}"
        for f in findings
    )

    user_prompt = (
        f"Research question: {question}\n\n"
        f"Available Findings (use only these finding_id and source_url pairs):\n{findings_block}\n\n"
        f"Produce a Report JSON matching the Report schema. Ensure every citation references a real finding_id with its exact source_url."
    )
    result = call_structured(SYSTEM_PROMPT, user_prompt, Report)
    return result.parsed, result.total_tokens


if __name__ == "__main__":
    import json
    from schemas import SourceRecord

    demo_findings = [
        Finding(
            sub_question_id="sq1",
            claim="Redis AOF persistence offers better durability than RDB snapshots at the cost of larger file size.",
            source=SourceRecord(
                url="https://redis.io/docs/management/persistence/",
                title="Redis Persistence",
                snippet="AOF logs every write operation...",
            ),
            confidence="high",
        )
    ]
    r, toks = write_report("How should I persist agent run state?", demo_findings)
    print(json.dumps(r.model_dump(), indent=2, default=str))
    print(f"\ntokens used: {toks}")
