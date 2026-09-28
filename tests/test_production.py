"""
tests/test_production.py
========================
Tests for the production-readiness layer:

1. Sorted-set run index: list_recent_runs is O(log N + limit) via
   ZREVRANGE instead of scanning the whole keyspace; ordering is
   newest-first; the index self-heals when keys vanish.
2. finished_at: terminal runs carry a measured end-to-end duration.
3. Auth: when API_AUTH_TOKEN is set, run-mutating endpoints demand a
   bearer token (401 without, 200 with); reads stay open.
4. Rate limiting: per-IP sliding window returns 429 with Retry-After.
5. SSE: /research/{id}/stream pushes state frames and terminates on
   completion.
"""

import json
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app, _rate_buckets, _rate_lock, enforce_public_guards
from schemas import (
    Citation,
    Finding,
    Report,
    ResearchPlan,
    ResearchResult,
    RunState,
    SearchStatus,
    SourceRecord,
    SubQuestion,
)
from storage.redis_store import INDEX_KEY, _r, list_recent_runs, save_state

client = TestClient(app)


# ---------------------------------------------------------------------------
# Sorted-set index
# ---------------------------------------------------------------------------

def test_list_recent_runs_ordered_newest_first():
    """
    The ZSET index must return runs newest-first without a keyspace scan.
    Asserts RELATIVE order over the whole index (the dev store may hold
    hundreds of runs from earlier sessions — position-slicing a page of
    15 would make the test depend on that pollution).
    """
    from datetime import datetime, timedelta, timezone

    older = RunState(question="older run")
    newer = RunState(question="newer run")
    older.budget.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    newer.budget.started_at = datetime.now(timezone.utc) + timedelta(hours=1)
    save_state(older)
    save_state(newer)

    total = _r().zcard(INDEX_KEY)
    runs = list_recent_runs(limit=total)
    ids = [r.run_id for r in runs]
    assert newer.run_id in ids and older.run_id in ids
    assert ids.index(newer.run_id) < ids.index(older.run_id)


def test_index_drops_stale_members_on_read():
    """A run deleted out-of-band must be pruned from the index, not crash reads."""
    ghost = RunState(question="ghost run")
    save_state(ghost)
    # delete the data key out-of-band, leaving a dangling index entry
    from storage.redis_store import _r, _key

    _r().delete(_key(ghost.run_id))

    runs = list_recent_runs(limit=50)  # must not raise
    assert ghost.run_id not in [r.run_id for r in runs]
    # and the dangling member was removed
    assert ghost.run_id not in set(_r().zrange(INDEX_KEY, 0, -1))


# ---------------------------------------------------------------------------
# Measured duration
# ---------------------------------------------------------------------------

def test_run_and_persist_stamps_finished_at():
    from storage.redis_store import get_or_create, run_and_persist, load_state
    from datetime import datetime, timezone

    mock_plan = ResearchPlan(
        original_question="Duration test",
        sub_questions=[SubQuestion(id="sq1", question="Q1", rationale="R1")],
    )
    mock_report = Report(question="Duration test", summary="S", citations=[
        Citation(claim="C", source_url="https://x.com", finding_id="f1")
    ])
    mock_finding = Finding(
        id="f1", sub_question_id="sq1", claim="C",
        source=SourceRecord(url="https://x.com", snippet="S"),
    )

    with patch("graph.run_planner", return_value=(mock_plan, 10)), \
         patch("graph.research_sub_question", return_value=(
             ResearchResult(sub_question_id="sq1", status=SearchStatus(ok=True), findings=[mock_finding]), 20)), \
         patch("graph.write_report", return_value=(mock_report, 30)):

        state = get_or_create(None, "Duration test")
        final = run_and_persist(state.run_id)

    assert final.status == "done"
    assert final.finished_at is not None
    duration = (final.finished_at - final.budget.started_at).total_seconds()
    assert duration >= 0

    # durable
    persisted = load_state(state.run_id)
    assert persisted.finished_at is not None

    # idempotent: re-entry must not move the timestamp
    from storage.redis_store import run_and_persist as rp
    again = rp(state.run_id)
    assert again.finished_at == final.finished_at


def test_recent_runs_exposes_duration_seconds():
    state = RunState(question="duration in list")
    state.status = "done"
    from datetime import datetime, timedelta, timezone

    # Stamp started_at an hour in the future so the run outranks every
    # pre-existing run in the shared dev store and lands in the newest-15
    # window; finished_at keeps the measured duration at 30s.
    start = datetime.now(timezone.utc) + timedelta(hours=1)
    state.budget.started_at = start
    state.finished_at = start + timedelta(seconds=30)
    save_state(state)

    resp = client.get("/research/recent")
    assert resp.status_code == 200
    match = [r for r in resp.json()["runs"] if r["run_id"] == state.run_id]
    assert match, "run missing from /research/recent"
    assert match[0]["duration_seconds"] is not None
    assert 29 <= match[0]["duration_seconds"] <= 31


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_on():
    """Temporarily enable API-key auth, then restore the off state."""
    old = api_main.API_AUTH_TOKEN
    api_main.API_AUTH_TOKEN = "secret-test-token"
    yield "secret-test-token"
    api_main.API_AUTH_TOKEN = old


def test_auth_blocks_unauthenticated_run_start(auth_on):
    resp = client.post("/research", json={"question": "no token"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_auth_rejects_wrong_token(auth_on):
    resp = client.post(
        "/research",
        json={"question": "bad token"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_auth_allows_valid_bearer_token(auth_on):
    """
    Valid bearer token passes the guard. Background execution is mocked
    out because TestClient runs FastAPI background tasks synchronously —
    without the mock, a successful POST would execute the real graph
    (live LLM/search spend) inside the test process.
    """
    with patch.object(api_main, "run_and_persist", lambda rid: None):
        resp = client.post(
            "/research",
            json={"question": "authed question"},
            headers={"Authorization": "Bearer secret-test-token"},
        )
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_auth_reads_stay_open(auth_on):
    """GET endpoints stay public: the guard protects cost, not secrecy."""
    assert client.get("/health").status_code == 200
    assert client.get("/research/recent").status_code == 200


def test_auth_off_by_default():
    """Zero-config local dev: POST works with no token configured."""
    assert api_main.API_AUTH_TOKEN == ""
    with patch.object(api_main, "run_and_persist", lambda rid: None):
        resp = client.post("/research", json={"question": "open local dev"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@pytest.fixture()
def tight_limit():
    """Tiny window for tests, and an isolated bucket per test."""
    old_reqs = api_main.RATE_LIMIT_REQUESTS
    old_win = api_main.RATE_LIMIT_WINDOW_SECONDS
    api_main.RATE_LIMIT_REQUESTS = 3
    api_main.RATE_LIMIT_WINDOW_SECONDS = 60
    with _rate_lock:
        _rate_buckets.clear()
    yield
    api_main.RATE_LIMIT_REQUESTS = old_reqs
    api_main.RATE_LIMIT_WINDOW_SECONDS = old_win
    with _rate_lock:
        _rate_buckets.clear()


def test_rate_limit_allows_burst_then_429_with_retry_after(tight_limit):
    """Burst over the limit -> 429 with Retry-After; all execution mocked."""
    with patch.object(api_main, "run_and_persist", lambda rid: None):
        statuses = []
        # TestClient shares one source IP ("testclient"), so the 4th request bursts
        for i in range(4):
            r = client.post("/research", json={"question": f"burst {i}"})
            statuses.append(r.status_code)
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429
    retry_after = int(client.post("/research", json={"question": "again"}).headers["Retry-After"])
    assert retry_after >= 1


def test_rate_limit_window_slides(tight_limit):
    """Entries older than the window no longer count."""
    from collections import deque

    ip = "testclient"
    ancient = time.monotonic() - (api_main.RATE_LIMIT_WINDOW_SECONDS + 10)
    with _rate_lock:
        _rate_buckets[ip] = deque([ancient, ancient, ancient])

    with patch.object(api_main, "run_and_persist", lambda rid: None):
        resp = client.post("/research", json={"question": "window moved on"})
    assert resp.status_code == 200  # ancient entries pruned, allowed


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

def _seed_running_run() -> RunState:
    state = RunState(question="SSE test")
    state.status = "planning"
    save_state(state)
    return state


def test_sse_404_for_unknown_run():
    assert client.get("/research/nonexistent-id/stream").status_code == 404


def test_sse_streams_done_run_and_terminates():
    """A terminal run yields exactly one state frame, then the stream ends."""
    state = _seed_running_run()
    state.status = "done"
    state.report = Report(question="SSE test", summary="S", citations=[
        Citation(claim="C", source_url="https://x.com", finding_id="f1")
    ])
    save_state(state)

    with client.stream("GET", f"/research/{state.run_id}/stream") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    assert len(events) == 1
    assert events[0]["status"] == "done"
    assert events[0]["run_id"] == state.run_id


def test_sse_emits_status_transitions_then_stops():
    """Seed planning, flip to done mid-stream, expect two frames and close."""
    state = _seed_running_run()

    def flip_later():
        time.sleep(0.3)
        s = RunState.model_validate_json(
            __import__("storage.redis_store", fromlist=["load_state"])
            .load_state(state.run_id)
            .model_dump_json()
        )
        s.status = "done"
        s.report = Report(question="SSE test", summary="S", citations=[])
        save_state(s)

    t = threading.Thread(target=flip_later, daemon=True)
    t.start()

    events = []
    with client.stream("GET", f"/research/{state.run_id}/stream") as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    t.join()

    statuses = [e["status"] for e in events]
    assert statuses == ["planning", "done"]
