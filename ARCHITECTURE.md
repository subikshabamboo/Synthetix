# Synthetix — Architecture Deep-Dive

*Why every file exists, how every piece works, and the design decisions behind them.*

Read this top to bottom and you can explain the entire system from memory — because the whole thing hangs off five ideas.

---

## 0. The Five Ideas That Explain Everything

Every file, every function, every weird-looking choice in this repo is an instance of one of these:

1. **Type the hand-offs, not the chat.** Multi-agent LLM systems fail not because the model is dumb but because state passes between agents as loose dicts and gets corrupted three nodes downstream. Every seam here is a validated Pydantic contract. (`schemas.py`)
2. **Budgets are code, not prompts.** "Please be efficient" is not a budget. A budget is an `if` statement that refuses to execute. Token/search/wall-clock limits live in `RunBudget` and are checked with plain Python. (`schemas.py`, `graph.py`)
3. **Failures are data, not exceptions.** A paywalled page or a 503 from Gemini is *information the supervisor should reason about*, so it's modeled as `SearchStatus` and flows through the graph as state. Crashes still get caught — but as structured error results, not run-killers. (`tools/search.py`, `agents/researcher.py`, `graph.py`)
4. **Persist after every step; make re-entry idempotent.** If the process dies at minute 3 of a 4-minute run, the completed nodes' output must already be durable. Redis snapshotting after every node + idempotent entry points = zero duplicate token spend. (`storage/redis_store.py`, `api/main.py`)
5. **Trust but verify — in code.** Never trust an LLM's claims about its own output. The writer's citations are programmatically re-validated against the actual finding records before a report is shown. Bad citations surface as `failed_citation_validation`, never as a quietly wrong report. (`api/main.py`)

That's it. Now the files, in dependency order.

> **Map-reduce note:** the researcher phase is a genuine fan-out — one
> `Send("research_worker", ...)` per sub-question runs concurrently, reducers
> merge the deltas, and the supervisor is the join that books spend once and
> re-fans only missing sub-questions. Details in §7.

---

## 1. `schemas.py` — The Constitution

**Why this file exists:** in an agent pipeline, the schema IS the architecture. Every agent receives and returns these objects; the Redis store serializes exactly one of them (`RunState`); the API validates against them. If you change this file, the whole system re-derives.

### The contracts

| Model | Role | Key detail |
|---|---|---|
| `SubQuestion` | One decomposed piece of the user's question | auto-generated 8-char id, `rationale` forced (why it matters) |
| `ResearchPlan` | Planner output | `max_sub_questions` enforced in `model_post_init` — see below |
| `SourceRecord` | Atomic evidence: url, title, snippet, `retrieved_at` | every fact must trace to one of these |
| `Finding` | Claim + its `SourceRecord` + confidence (`high/medium/low`) + `sub_question_id` | the unit the writer cites |
| `SearchStatus` | `ok` flag + machine-readable `reason` (`ok / no_results / rate_limited / paywalled / timeout / error`) | Idea #3 in its purest form |
| `ResearchResult` | Everything the researcher produced for ONE sub-question, including failure info | status + findings live together |
| `Citation` | claim + `source_url` + `finding_id` | `finding_id` must resolve to a real `Finding` |
| `ReportSection` / `Report` | Synthesized answer: summary, sections, conclusion, citations | sections were added because the UI renders them — see §10 |
| `RunBudget` | The hard limits: 5 sub-questions, 3 searches/SQ, 150k tokens, 240s wall clock | plus the `*_exhausted()` checks |
| `RunState` | The single shared state object that flows through the whole graph | status (incl. `auditing`), plan, results, report, budget, trace, `conflicts`, `revision_count`, `finished_at` |

### Design decisions inside

- **Truncation guard in `model_post_init`**: if the planner LLM returns 9 sub-questions despite the prompt, the plan is *silently truncated to 5 at construction*. An out-of-budget `ResearchPlan` cannot exist. That's Idea #2 applied to schema construction.
- **`RunBudget.time_exhausted()` handles naive datetimes**: older persisted states had no `tzinfo`; comparing naive vs aware raises `TypeError`. The check normalizes to UTC before comparing. (Real bug class, fixed in code, not in hope.)
- **`revision_count`** is what turns "the reviewer may send work back once" from a hope into a rule — the supervisor checks the counter before looping.
- **`finished_at`** is stamped only at terminal transitions (`done`/`failed`). Measured end-to-end latency = `finished_at − budget.started_at`. This is how the README's latency claims became checkable data instead of folklore.
- **`trace: list[dict]`** looks loose next to all the strict models — deliberate. Trace entries are an audit log, not a contract; over-typing logs makes them brittle for zero benefit.

---

## 2. `llm.py` — The Single Choke Point

**Why one wrapper instead of calling the SDK inside every agent:**

1. **Token accounting has to happen in exactly one place**, or you double-count or miss calls. `call_structured` returns `LLMCallResult(parsed, input_tokens, output_tokens)` — callers update the budget themselves, the wrapper never touches shared state.
2. **"Ask for JSON, parse into a Pydantic model"** is the repeated pattern across planner/researcher/writer. Written once, tested once.
3. **Model changes happen in one place.**

### How a call works

```
call_structured(system_prompt, user_prompt, SchemaClass)
  → builds GenerateContentConfig (system_instruction, response_mime_type="application/json",
    response_schema=SchemaClass, max_output_tokens=8192)
  → tries models_to_try in order
  → strips accidental markdown fences
  → json.loads + SchemaClass.model_validate  ← validation happens HERE
  → extracts usage_metadata token counts
  → returns LLMCallResult
```

### The retry ladder (this was rebuilt after a live incident)

- **Primary model: 4 attempts**, exponential backoff **2s → 4s → 8s → 16s** (~30s worst case). The original 3 attempts ~3s apart lost every race against real Gemini "high demand" 503 storms — the pipeline died mid-run on a healthy day.
- **Alternate models (`gemini-3.8-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`): 2 quick attempts each, 2s apart.** Models have independent capacity pools, so fallback works; but bounded (worst case ≈ 42s/call) so retries can't eat the run's 240s budget.
- **404 (model retired) → skip that model instantly**, no sleeping. This list was verified against the live API — 2.x models are gone.
- **Anything else (bad key, malformed request) → raise immediately.** Non-availability errors must fail loudly; fallback would only mask a config bug.
- **Parse failure → `LLMStructuredOutputError`** including a snippet of raw output, so debugging shows you *what* the model actually said.

---

## 3. `agents/planner.py` — Decomposition

A system prompt (pure spec: "3–5 answerable, non-overlapping sub-questions, each with a rationale") + one function.

**Design decisions:**
- No "You are Dr. Research" roleplay — the model's job is fully specified by the schema it must fill. Personality prompts produce prose; schemas produce data.
- `plan()` returns `(ResearchPlan, tokens)` — tokens go back to the *caller*, which decides how to book them. The agent stays a pure-ish function with no hidden side effects on shared state.
- After the call, the agent re-applies `max_sub_questions` (belt-and-suspenders on top of the schema-level truncation guard).
- `if __name__ == "__main__"` smoke test exists because each agent must be testable **alone** before wiring into the graph — if the planner produces vague sub-questions in isolation, no orchestration rescues it.

---

## 4. `agents/researcher.py` — The Seam

The most important structural move in the codebase is here: the search backend is a **swappable function parameter**.

```python
def research_sub_question(sub_question_id, question_text, search_backend=None):
    backend = search_backend or _default_backend()   # real Tavily, lazily imported
```

- Phase 1 was built against `stub_search` (canned results) so the LLM extraction logic was validated with **zero network flakiness**. Phase 3 swapped in `tools.search.tavily_search_backend` and the researcher barely changed. That's the payoff of designing the seam early.
- The backend's `try/except` converts any raising backend into `SearchStatus(ok=False, reason="error")` — a buggy tool is contained data, not a crashed run.
- **`_FindingsOnly`** is a private schema: the LLM only produces `findings`. Status is computed by code, because "did the search succeed" is infrastructure truth the model cannot know.
- **Globally unique finding IDs, enforced not asked:** the model naturally writes local ids (`1`, `2`…) per sub-question, which collide when findings merge (`all_finding_by_id` would silently overwrite). The code overwrites whatever the model said with `f"{sub_question_id}_f{idx}"` — deterministic, unique, debuggable. Same story for `sub_question_id`: enforced, never trusted.
- Extraction prompt rule that matters: *"If nothing in the sources answers the sub-question, return an empty findings list — do not guess."* Empty is a valid, expected answer.

---

## 5. `agents/writer.py` — Constrained Synthesis

**Two defensive decisions:**

1. **The writer never invents finding ids.** It receives the exact findings list (ids, claims, source urls) and its ONLY job is to select and phrase. The prompt states the invariant twice: never invent an id, never pair an id with a different url. This is what makes Phase 5's citation validation *meaningful* rather than theater — the model was already constrained to reference real ids, and the validator checks anyway.
2. **Zero findings → no LLM call at all.** It returns an explicit "Insufficient evidence" report with 0 tokens. A model asked to write a report from nothing will happily hallucinate one; we don't ask.

`sections` + `conclusion` exist on `Report` because the frontend renders titled sections; when the schema lacked them, the model dutifully produced structure that was then thrown away at validation. Optional-with-defaults keeps old persisted runs valid.

---

## 6. `tools/search.py` — Live Retrieval with a Failure Taxonomy

Three requirements, three functions:

**1. `fetch_and_extract(url)` — don't trust snippets.** Search snippets are truncated and often misleading, so each result's page is downloaded (httpx, 8s timeout, redirects followed) and run through **trafilatura** for readable-text extraction. Any failure (403/402 → paywalled, timeout, connect error, anything else) returns `None` — the caller decides that's not fatal.

**2. `_dedup_and_cap(results)` — diversity guardrails.** Exact-URL dedup, max **2 results per domain**, max **5 total**. Without the domain cap one site (usually a single SEO-heavy domain) dominates the evidence base and the report becomes an echo.

**3. `tavily_search_backend(query)` — every failure path is explicit.** It matches the researcher's seam shape but returns `(SearchStatus, results)` because the search layer must report *why* it failed, not just *that* it failed:

| Situation | Returned status |
|---|---|
| client raises with rate/429 | `rate_limited` |
| client raises with timeout | `timeout` |
| any other client error | `error` |
| zero results | `no_results` |
| all pages unextractable | `paywalled` |
| otherwise | `ok` + enriched results |

Enrichment: full text (first 1500 chars) or the Tavily snippet (first 800) as fallback; a page that yields nothing extractable is *skipped*, never fed to the LLM as empty context.

---

## 7. `graph.py` — Orchestration

### The two-state trick

LangGraph's `StateGraph` is built around plain dict state for partial updates (each node returns only changed keys). But our contracts are Pydantic. So:

- `GraphState` is a `TypedDict` — the **wire format** the graph merges.
- Every node converts dict → `RunState.model_validate(g)` at entry and `state.model_dump()` at exit.

Slightly annoying, deliberately so: type safety *inside* nodes, LangGraph-native merging *between* them.

### The node contract (discipline that makes parallelism safe)

Every node returns a **DELTA** — only the keys it changed, and for reducer
channels only the NEW entries. `results`, `trace`, and the worker-spend
counters are `Annotated[..., operator.add]` channels: concurrent writers
append, and returning a full snapshot through an `add`-reducer would append
the whole old list to itself (the classic LangGraph duplication bug).

### The nodes

- **`node_planner`** — calls the planner, books tokens, sets status `researching`.
- **`route_after_planner`** — emits one `Send("research_worker", ...)` per
  sub-question (**map**). If the budget is already dead, routes a single
  `Send` to the supervisor instead — "dead budget ⇒ writer with documented
  gaps" stays true even before any research runs.
- **`node_research_worker(payload)`** — one worker = one sub-question:
  - skips sub-questions that already succeeded (**idempotency across revision passes** — a re-fanned worker never re-spends on covered ground),
  - wraps its research call in `try/except` → a crash becomes `SearchStatus(ok=False, reason="error")` (per-SQ fault isolation — one bad LLM call used to kill whole runs; now it's one structured gap among N healthy workers),
  - returns its own result + `_worker_tokens`/`_worker_searches` deltas + one trace entry. Workers execute **concurrently** (search + extraction are I/O-bound): wall-clock ≈ the slowest sub-question, not the sum.
- **`node_supervisor`** — the **JOIN**. **Zero LLM calls, zero tools.** Pure set algebra:
  - books the workers' cumulative `_worker_tokens`/`_worker_searches` into the authoritative budget **once, in one deterministic place**;
  - `covered` = results with `status.ok` **and** non-empty findings,
  - `missing` = plan ids − covered ids,
  - `budget_dead` = tokens/time/searches exhausted,
  - if missing AND not budget_dead AND `revision_count < 1` → sets `_refan` to **only the missing sub-questions** (not the whole plan), counter incremented;
  - else → status `auditing` (logging *why*: full coverage vs proceeding-with-gaps).
- **`node_refan`** — re-emits workers for just the missing sub-questions (**revision pass**); the `refan → research_worker` edge re-enters the same map step.
- **`node_auditor`** — **contradiction detection, pure logic, zero LLM.** Pairwise check over findings from DIFFERENT sub-questions (same-SQ pairs share one source context and are skipped):
  - **negation pairs** with ≥ 0.5 lexical overlap ("X is safe" vs "X is not safe"),
  - **numeric disagreement**: no pair of extracted numbers within 15% of each other, with ≥ 0.4 overlap ("1000 GW" vs "740 GW"),
  - **opposing comparatives** between the same entities ("A faster than B" vs "A slower than B", overlap-gated),
  - hedges ("may", "roughly", "~") are stripped first, so softening a claim can't dodge the check — and two denials agree, not conflict.
  
  Deterministic rules cost zero tokens, cannot hallucinate conflicts, and are unit-testable with plain asserts. Conflicting findings still flow to the writer; each conflict is recorded in state, surfaced by GET `/research/{id}` and rendered as a banner in the studio so the reader weighs the disagreement with both citations in view.
- **`node_writer`** — merges all findings (**deduped by finding id** — resumed runs can hold results from more than one pass), writes the report, status `done`.

### The shape

```
START → planner ──Send×N──→ research_worker ×N (parallel) ──→ supervisor (join)
                 ▲                                              │
                 └── refan: ONLY missing SQs (max 1 revision) ──┤
                                                                ├──(gaps remain)──→ auditor
                                                                └──(full coverage)─→ auditor

auditor (pairwise contradiction check) → writer → END
```

One supervised loop, ever. No unbounded reflection. Status flow:
`planning → researching → auditing → writing → done`.

---

## 8. `storage/redis_store.py` — Durable State, Idempotent Everything

**The key insight: persistence never touches graph.py.** `run_and_persist` wraps `graph.stream(..., stream_mode="values")` and writes the full `RunState` to Redis after **every node yields**. Kill the process mid-run, and the last completed node is already durable — re-entry resumes exactly there without re-spending tokens.

- **`get_or_create(run_id, question)`** — the idempotency front door: an existing run_id returns existing state (a client retrying a timed-out POST cannot fork duplicate planning/research/writing).
- **`run_and_persist`** — done-runs return immediately (no-op re-entry); terminal states stamp `finished_at` exactly once.
- **`save_state`** — a Redis **pipeline** (atomic): `SET run:{id}` + `ZADD runs:index {id → started_at epoch}`. Durable (no TTL — this store IS the history and the resume source).
- **`list_recent_runs`** — one `ZREVRANGE` on the sorted-set index: **O(log N + limit)** instead of scanning and parsing every key. Self-healing: dangling index members are pruned; runs missing from the index (legacy, or written around `save_state`) are lazily re-indexed on first read and the result is re-sorted.
- **`ping()`** — powers `/health`. A server answering while Redis is down is a lie for this app.

---

## 9. `api/main.py` — The Boundary

### Endpoints

| Endpoint | Method | Guard | Purpose |
|---|---|---|---|
| `/research` | POST | rate-limit + optional bearer | start a run, return `run_id` immediately |
| `/research/{id}/resume` | POST | rate-limit + optional bearer | resume failed run, **fresh wall-clock budget** |
| `/research/{id}` | GET | open | status, plan, results, trace, validated report |
| `/research/{id}/trace` | GET | open | the audit log verbatim |
| `/research/{id}/stream` | GET | open | **SSE** push of state transitions |
| `/research/recent` | GET | open | indexed recent runs + `duration_seconds` |
| `/research/{id}/corrupt-citation` | POST | open (demo) | mutate a citation to show validation catches it |
| `/health` | GET | open | ok + Redis dependency state |

### The security layer (opt-in, off by default)

- **Rate limiting is always on**: per-IP sliding window (`time.monotonic()` deques, self-pruning per request), default 10 run-starts / 60s. Over the limit → `429` with `Retry-After`. Deliberately fires *before* auth so probing doesn't leak endpoint behavior.
- **Auth arms with one env var**: set `API_AUTH_TOKEN` and mutating endpoints require `Authorization: Bearer …` (401 + `WWW-Authenticate: Bearer` otherwise). Reads stay public — this guards **cost** (your Gemini/Tavily quota), not secrecy.
- Both are `Depends(enforce_public_guards)` on the two mutating endpoints only. Local dev and tests stay zero-config.

### Concurrency correctness

- `_INFLIGHT_RUNS` (set + real lock, because FastAPI runs sync endpoints on a threadpool): a client retrying mid-run — or two tabs — cannot fork a second identical graph execution. Redis idempotency covers re-entry *after* completion; this covers *during*.
- Background task exceptions are caught and **persisted into the run state** (`failed` + trace note + `finished_at`) — otherwise FastAPI swallows them and the run looks like it's hanging forever.

### Citation validation (Idea #5, the anti-hallucination guarantee)

On every `done` read, `_validate_citations` rebuilds the finding map and checks every citation:
- unknown `finding_id` → problem,
- `source_url` mismatch (trailing-slash-insensitive) → problem.

Problems ⇒ the response ships as `failed_citation_validation` with the problem list — **never** a quietly wrong report. The `/corrupt-citation` endpoint deliberately breaks a citation so you can *watch the guard fire*.

### SSE (`/research/{id}/stream`)

An async generator tail-polls the authoritative store (Redis) at 1s — cheap single-key GETs — and emits an `event: state` frame **only when the persisted status changes**, then closes on terminal status. Heartbeats every ~15s keep proxies from reaping idle connections; `X-Accel-Buffering: no` defeats nginx buffering. SSE over WebSockets because data flows one way, it traverses proxies cleanly, and `EventSource` reconnects for free.

### Resume restarts the clock

`resume` sets `budget.started_at = now`. The wall-clock budget protects *one execution attempt*, not the run's lifetime — a run that failed at minute 2 and resumed at minute 30 must not inherit a dead clock (observed live: it made the researcher skip every sub-question and produce an empty report).

---

## 10. `frontend/` — The Studio

- **`index.html`**: hero + presets; 5-step pipeline stepper; 4 live telemetry meters (tokens, searches, findings, evidence reviews); left panel (plan + extracted evidence); right panel (report + citations table); saved-runs history; live activity timeline; provenance modal. The logo is inline SVG — a molecule/network mark on the brand gradient — crisp at any size, zero image assets. CSS/JS links are versioned (`?v=2`) for cache-busting.
- **`app.js`**:
  - **One live-update channel, ever.** `followRun()` opens `EventSource(/stream)`; `state` frames drive the exact same `renderRunState()` the poller used; hard SSE failure (or no support) falls back to the 1.5s polling loop. Both paths converge on identical rendering, and terminal states stop the channel and refresh history.
  - **`renderRunState`** updates stepper classes (`completed/active/pending/failed/validation`), telemetry bars, plan cards, finding cards (confidence chips, verbatim quote, source link), report (summary → sections → conclusion → citations table), and the trace timeline.
  - **Provenance inspector**: clicking any citation opens a modal with the finding's claim, the **verbatim source quote**, and the resolved URL — the "show me where this came from" moment. Unknown ids render as a validation anomaly.
  - **Hygiene**: every injected string goes through `escapeHtml`; export buttons produce Markdown/JSON; resume and corrupt-citation are one click each.
- **`styles.css`**: design tokens in `:root` (slate palette, orange→amber CTA gradient, mono for data), so the brand is one variable away.

---

## 11. `tests/` — Why the Suite Is Shaped Like This

**Hermetic by construction.** Every LLM call goes through `call_structured`; tests patch it (or patch agent functions at module boundaries) with `LLMCallResult` fakes. Search tests stub backends or httpx. Therefore the suite needs **no API keys**, runs in ~10–15s, and CI spends zero tokens. That property is what made the CI workflow trivial.

| File | What it locks down |
|---|---|
| `test_schemas.py` | truncation guard, evidence records, `SearchStatus` vocabulary, budget exhaustion math, `RunState` defaults |
| `test_phase1.py` | each agent **alone** (planner/researcher/writer), including the zero-findings → no-LLM path |
| `test_phase2.py` | supervisor routing (full coverage / gap→one loop / budget-dead→proceed) and a full mocked graph run with exact token accounting |
| `test_phase3.py` | dedup/cap math, fetch success vs paywall/error, Tavily no-results vs rate-limited |
| `test_phase4.py` | save/load roundtrip, `get_or_create` idempotency, done-run no-op, full persisted execution |
| `test_phase5.py` | endpoints + citation validator (success AND detection of forged id / mismatched url) + static frontend serving |
| `test_deliberate_breaks.py` | the suite's crown jewels: **deliberately broken scenarios** — killed budget, gibberish search results, corrupted citation in Redis, idempotent re-entry — because a pipeline is only as good as its worst-day behavior |
| `test_resilience.py` | the live-incident regressions: 503 backoff sleep sequence, model-fallback attempt counts, fail-fast on non-availability, per-SQ crash isolation, no double-start, health, schema roundtrip |
| `test_production.py` | index ordering + stale pruning, `finished_at` stamping (idempotent + durable) + `duration_seconds`, auth (401/401/200, reads open, off by default), rate limiting (burst→429+Retry-After, sliding window), SSE (404, single terminal frame, transition frames) |

Testing-note: `TestClient` runs FastAPI background tasks **synchronously** — so every POST-happy test mocks `run_and_persist`, otherwise a "successful" POST in a test would execute the real graph (real spend). Also, shared dev Redis means assertions are written relative (order, membership) rather than absolute positions.

---

## 12. `.github/workflows/ci.yml` — Portable Trust

On every push to `main` and every PR: a real **Redis 7 service container** (health-checked), Python 3.12 with pip caching, `pip install -r requirements.txt`, `pytest -v`, then a smoke import of the FastAPI app. Placeholder keys are set **because the suite is hermetic** — `llm.py` asserts a non-placeholder key at client-build time, so placeholders + mocked calls = the pipeline's full logic proven, zero spend. The README badge is wired to this workflow.

---

## 13. One Run, End to End (the story you tell)

1. **POST /research** {"question": "…"} → rate-limit check → (optional) auth check → `get_or_create` seeds `RunState` in Redis → background task queued → `{run_id, status:"planning"}` returns **instantly**.
2. **Planner node** → Gemini structured output → `ResearchPlan` (≤5 sub-questions, truncated in code) → tokens booked → snapshot to Redis + index → SSE `state` frame pushes to any listener.
3. **Researcher node** → per sub-question: budget check → Tavily search (dedup/cap) → trafilatura extraction (snippet fallback) → Gemini extraction into `_FindingsOnly` → ids enforced `{sq_id}_f{idx}` → `ResearchResult` appended → snapshot after each. Any crash here = structured `error` result, run continues.
4. **Supervisor node** (pure logic) → missing coverage + revision budget left → one retry loop; otherwise → writer with the gap documented in the trace.
5. **Writer node** → findings (ids + urls) → Gemini `Report` (summary/sections/conclusion/citations) → `done` → snapshot, `finished_at` stamped.
6. **GET /research/{id}** → citation validation against real finding records → report **or** `failed_citation_validation` with the problem list.
7. **UI** → SSE frames move the stepper in step with the pipeline; telemetry bars track tokens/searches/`finished_at−started_at`; every citation is one click from its verbatim provenance.

---

## 14. Failures We Actually Hit (and how they shaped the code)

These are the stories that prove the design — each one turned into a regression test.

1. **Finding-ID collisions across sub-questions.** Local indices (`1`, `2`…) from each sub-question silently overwrote each other in the citation map → false URL mismatches. Fix: deterministic global ids at extraction time (`{sq}_f{idx}`), enforced in code.
2. **Naive vs aware datetimes from old Redis blobs** → `TypeError` on sort. Fix: normalization in `_get_run_timestamp` / `time_exhausted`.
3. **Paywall/timeout scraping crashed runs.** Fix: the whole failure taxonomy in `SearchStatus`; extraction failures degrade to snippets; unusable sources are skipped.
4. **Gemini 503 "high demand" storm killed runs mid-research** (observed live): 3 attempts × ~3s lost every race. Fix: real exponential backoff, bounded model fallback, fail-fast on non-availability — all sleep-tested.
5. **One sub-question's LLM crash killed the whole graph** despite `SearchStatus` existing for exactly this. Fix: per-SQ `try/except` in `node_researcher` → structured error result; supervisor spends its revision on it; writer still ships.
6. **Resume had a dead clock**: resumed runs inherited the original `started_at`, skipped all research, produced empty reports. Fix: resume = fresh attempt = restart the wall clock.
7. **Report sections silently dropped**: the model produced structure the schema discarded. Fix: `ReportSection`/`conclusion` on the contract; optional for old persisted runs.
8. **Duplicate concurrent runs**: a mid-flight retry forked a second identical execution. Fix: in-flight set + lock (Redis idempotency alone can't see "currently running").
9. **A retired model name (404) in the fallback list** — discovered live, when the API told us 2.x is gone. Fix: alternates pinned to the verified current lineup; 404s skip instantly.

---

## 15. Trade-offs We Chose Knowingly

| Chose | Gave up | Why |
|---|---|---|
| Pydantic contracts at every seam | Free-form agent "creativity" | Deterministic failures in tests beat corrupted state in prod |
| Redis snapshot per node | A few ms of I/O per step | Crash-resume + idempotency + live UI all read the same store |
| Deterministic supervisor (no LLM) | "Smarter" routing | Testable with asserts; budgets enforced as code; no runaway loops |
| Parallel worker fan-out (Send API) | Simple sequential loop | Wall-clock ≈ slowest sub-question, not the sum; delta-only node contract keeps concurrency safe by construction |
| Zero-LLM contradiction auditor | An LLM "contradiction detector" | Zero tokens, deterministic, cannot hallucinate conflicts, unit-testable |
| 1 supervised revision, max (re-fans only missing SQs) | Unlimited self-improvement | Hard ceiling on spend; diminishing returns after one pass |
| SSE + 1s store tail-poll | Redis pub/sub fan-out | One writer (the snapshot), one reader pattern, no subscription lifecycle bugs |
| In-memory per-IP rate limiting | Cross-worker exactness | Zero deps, honest sliding window; Redis-based limiting is the documented next step for multi-worker |
| No TTL on run keys | Automatic pruning | The store IS the history/resume/UI list; pruning is a retention policy, not a side effect |

---

## 16. Runbook

```bash
cd research_assistant
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                 # fill GEMINI_API_KEY, TAVILY_API_KEY
docker run -d --name research-redis -p 6379:6379 redis:7-alpine
.venv\Scripts\pytest -v                                # 70 tests, no API keys needed
.venv\Scripts\uvicorn api.main:app --port 8000 --reload
# open http://localhost:8000

# production guards (optional):
#   set API_AUTH_TOKEN=...        → mutating endpoints need Bearer auth
#   set RATE_LIMIT_REQUESTS / RATE_LIMIT_WINDOW_SECONDS → tune the limiter
```
