"""
llm.py
======
THEORY: Why one shared wrapper instead of calling genai.Client()
inside every agent file.

1. Token accounting (RunBudget.tokens_used) has to happen in exactly one
   place, or you will double-count or miss calls. Centralizing the API
   call is what makes the budget enforcement in Phase 2 possible at all.
2. "Ask for JSON, parse into a Pydantic model" is a repeated pattern across
   Planner / Researcher / Writer. Writing it once avoids three slightly
   different, slightly buggy copies.
3. If you swap models later (e.g. gemini-2.5-flash for the fast extraction
   step, gemini-2.5-pro for the Writer), you change it in one place.
"""

from __future__ import annotations
import json
import os
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or "gemini-flash-latest"

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key.startswith("AIzaSy_YOUR_GEMINI_API_KEY") or api_key == "AIzaSy...":
            raise ValueError(
                "GEMINI_API_KEY is not set or contains a placeholder. "
                "Please set a valid Google Gemini API key (starting with AI...) in your .env file or environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


T = TypeVar("T", bound=BaseModel)


class LLMCallResult:
    """Wraps a parsed object together with token usage, so callers can
    update RunBudget.tokens_used without re-touching the raw API response."""

    def __init__(self, parsed: BaseModel, input_tokens: int, output_tokens: int):
        self.parsed = parsed
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def call_structured(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_tokens: int = 8192,
) -> LLMCallResult:
    """
    Calls Google Gemini API, demands JSON matching `schema`, and validates it.

    Uses Gemini's native structured JSON schema mode with fallback parsing
    and token usage extraction.
    """
    from google.genai import types

    client = _get_client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=max_tokens,
    )

    models_to_try = [MODEL]
    for alt in ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    last_err = None
    response = None
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
                break
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "503" in err_str or "unavailable" in err_str or "429" in err_str or "rate" in err_str or "demand" in err_str:
                    import time
                    time.sleep(2 ** attempt + 1)
                    continue
                elif "404" in err_str or "not_found" in err_str:
                    # Model not available, try next candidate model
                    break
                else:
                    raise LLMStructuredOutputError(f"Gemini API call failed: {e}") from e
        if response is not None:
            break

    if response is None and last_err is not None:
        raise LLMStructuredOutputError(f"Gemini API call failed after retries: {last_err}") from last_err

    raw_text = (response.text or "").strip()

    # Defensive cleanup: strip code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        parsed = schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise LLMStructuredOutputError(
            f"Model output failed to parse/validate as {schema.__name__}: {e}\n"
            f"Raw output was:\n{raw_text[:1000]}"
        ) from e

    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

    return LLMCallResult(
        parsed=parsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class LLMStructuredOutputError(Exception):
    pass
