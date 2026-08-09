"""
API contract tests for the interception proxy.

Rewritten for the JSON SSE protocol. The previous version asserted the old
hardcoded string format ("[INTERCEPTION: SEMANTIC ENTROPY ...") and a greeting
bypass that returned a canned "Security Gateway Active" banner; both are gone.
Events are now JSON objects carrying real per-token telemetry.

Run: python -m pytest test_phase3_interception.py -v
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

import main
from test_stage2_granite_live import _model_available

BASE = "http://test"


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=main.app), base_url=BASE)


async def _events(resp) -> list:
    body = b""
    async for chunk in resp.aiter_bytes():
        body += chunk
    return [
        json.loads(line[6:])
        for line in body.decode().split("\n\n")
        if line.startswith("data: ")
    ]


# --------------------------------------------------------------------------- #
# Endpoints that need no model
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_health_reports_measured_state():
    async with _client() as ac:
        body = (await ac.get("/health")).json()
    assert body["circuit_breaker_tau"] > 0
    assert body["event_log_events"] == 460
    assert body["event_log_cases"] == 150
    # The old endpoint returned a literal interception_latency_ms of 11.4.
    assert "interception_latency_ms" not in body


@pytest.mark.asyncio
async def test_graphs_describes_only_the_process_that_exists():
    async with _client() as ac:
        body = (await ac.get("/graphs")).json()
    assert set(body) == {"o2c"}, "only one process graph exists in the event log"
    g = body["o2c"]
    assert g["collection"] == "celonis_ground_truth"
    assert g["vector_count"] == 460
    # These four were advertised with invented vector counts and never existed.
    assert not {"p2p", "ap_audit", "supply_chain"} & set(body)


@pytest.mark.asyncio
async def test_metrics_are_derived_from_the_event_log():
    import celonis_metrics as cm
    async with _client() as ac:
        body = (await ac.get("/metrics")).json()
    assert body["mean_cycle_days"] == cm.process_profile()["mean_cycle_days"]
    assert body["declared_avg_compliance_cycle_time_days"] == 10.4


@pytest.mark.asyncio
async def test_recover_answers_only_with_grounded_figures():
    import re
    import celonis_metrics as cm
    async with _client() as ac:
        body = (await ac.post("/recover", json={"query": "mean compliance cycle time"})).json()
    assert body["status"] == "RECOVERED"
    for n in re.findall(r"\d+\.?\d*", body["verified_ground_truth"].replace(",", "")):
        assert cm.is_grounded_number(float(n)), f"recovery stated ungrounded {n}"


# --------------------------------------------------------------------------- #
# Streaming, which needs the weights
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def loaded():
    if not _model_available():
        pytest.skip("Granite model not available via Ollama")
    if main._state["runner"] is None:
        main._load_models()
    return main._state


@pytest.mark.asyncio
async def test_stream_emits_json_events_with_real_logprobs(loaded):
    async with _client() as ac:
        async with ac.stream("GET", "/stream?query=What is the mean compliance cycle time?") as r:
            assert r.status_code == 200
            events = await _events(r)

    assert events, "stream produced nothing"
    assert all("kind" in e for e in events)
    assert events[-1]["kind"] in {"done", "recovery"}

    tokens = [e for e in events if e["kind"] == "token"]
    if tokens:
        lps = [e["logprob"] for e in tokens]
        assert all(lp <= 0 for lp in lps)
        assert len(set(lps)) > 1, "logprobs constant - not real model output"
        assert set(round(lp, 2) for lp in lps) != {-0.05}, "old hardcoded value"


@pytest.mark.asyncio
async def test_unprotected_stream_never_intercepts(loaded):
    async with _client() as ac:
        async with ac.stream("GET", "/unprotected_stream?query=What is the mean compliance cycle time?") as r:
            events = await _events(r)
    assert not [e for e in events if e["kind"] == "intercept"], "baseline must not intervene"
    assert events[-1]["kind"] == "done"
    assert events[-1]["intercepted"] is False


@pytest.mark.asyncio
async def test_answerable_query_ends_grounded(loaded):
    """
    /stream decodes ungrounded first so the proxy is the only variable against
    the baseline, which means an answerable query may also trip and then be
    repaired. What must hold is the OUTCOME: whatever path it took, the answer
    the user is left with states only figures the event log supports.
    """
    import re
    import celonis_metrics as cm

    async with _client() as ac:
        async with ac.stream(
            "GET", "/stream?query=Give the exact mean cycle time for Compliance Review."
        ) as r:
            events = await _events(r)

    final = events[-1]
    assert final["kind"] in {"done", "recovery"}, f"stream ended on {final['kind']}"

    if final["kind"] == "recovery":
        assert final["all_figures_grounded"], "recovery answer still contains an ungrounded figure"
    answer = final.get("text") or "".join(
        e["text"] for e in events if e["kind"] in {"token", "recovery_token"}
    )
    for n in re.findall(r"\d+\.?\d*", answer.replace(",", "")):
        assert cm.is_grounded_number(float(n)), f"final answer states ungrounded {n}: {answer}"
