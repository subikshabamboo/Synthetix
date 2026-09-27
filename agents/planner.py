"""
agents/planner.py
==================
THEORY: "Planner takes a research question and returns a structured plan
of sub-questions."

Notice the shape of this file: it is a prompt (data) + one function that
calls the LLM and returns a typed object. There is no "personality" —
no "You are Dr. Research, a world-renowned expert..." roleplay. The
brief specifically calls that out as an anti-pattern: "Skip the you are
a world class researcher framing and define Pydantic models for every
hand-off instead." The model's job is fully specified by the schema it
must fill in, not by flavor text.
"""

from schemas import ResearchPlan
from llm import call_structured

SYSTEM_PROMPT = """You are a research planning function. Given a research \
question, decompose it into a small number of specific, independently \
answerable sub-questions that together cover the original question.

Rules:
- Each sub-question must be answerable via web search (avoid pure opinion).
- Sub-questions should not overlap heavily.
- Prefer 3-5 sub-questions; never exceed 5.
- Each sub-question needs a short rationale: why it matters to the answer.
"""


def plan(question: str, max_sub_questions: int = 5) -> tuple[ResearchPlan, int]:
    """
    Returns (plan, tokens_used). Tokens are returned so the caller (the
    graph node, in Phase 2) can update RunBudget.tokens_used itself —
    this function stays a pure-ish function with no hidden side effects
    on shared state.
    """
    user_prompt = (
        f"Research question: {question}\n\n"
        f"Produce at most {max_sub_questions} sub-questions as JSON matching "
        f"the ResearchPlan schema. Set original_question to the question above."
    )
    result = call_structured(SYSTEM_PROMPT, user_prompt, ResearchPlan)
    result.parsed.max_sub_questions = max_sub_questions
    # re-run the truncation guard now that max is set for this call
    result.parsed.sub_questions = result.parsed.sub_questions[:max_sub_questions]
    return result.parsed, result.total_tokens


if __name__ == "__main__":
    # THEORY (Phase 1 instruction): "Test each agent alone before wiring
    # them together. If the planner produces vague sub-questions in
    # isolation, no orchestration will rescue it." This block is exactly
    # that manual smoke test — run `python agents/planner.py` directly.
    import json
    p, toks = plan("What are the tradeoffs of using Redis vs Postgres for agent run state?")
    print(json.dumps(p.model_dump(), indent=2, default=str))
    print(f"\ntokens used: {toks}")
