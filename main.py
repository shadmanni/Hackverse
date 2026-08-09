"""
Sentinel-RAG interception proxy.

The split-screen demo is a genuine A/B on one model:

  /unprotected_stream - Granite with NO retrieved context and NO interception.
                        It is asked for figures the event log never gave it, so
                        it fabricates on its own. Nothing is scripted.
  /stream             - the same Granite, given retrieved Celonis context and
                        audited per token. Breaching the threshold terminates
                        decoding mid-sentence.

Both sides emit one JSON object per SSE event so the terminal can plot real
log-probabilities instead of animating a fixed string.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

import celonis_metrics as cm
import semantic_divergence as sd
from entropy_engine import EntropyEngine
from sentinel_stream import SentinelStream

load_dotenv()

# Calibrated against real granite3.3:8b log-probabilities (calibrate_tau.py,
# re-run 2026-08-09 after that script was repaired - it had been iterating an
# async generator with a plain `for` since the move to Ollama, so it could not
# execute, and the previous tau was inherited from the transformers/granite-2b
# runner it was measured on).
#
# Measured on the live model, 6 answerable vs 6 unanswerable prompts:
#   tau=0.65 run=2 -> 1 false positive, 4/6 detected   <- the previous setting
#   tau=0.85 run=2 -> 0 false positives, 3/6 detected  <- this one
# A false positive halts a correct answer on stage; a miss is picked up by the
# numeric-grounding layer, which is deterministic and independent of tau. That
# asymmetry is why the safer threshold wins. Six prompts per class is a small
# sample - report these as counts, never as a percentage.
TAU = float(os.getenv("SENTINEL_TAU", "0.85"))
WINDOW = int(os.getenv("SENTINEL_WINDOW", "5"))
MAX_NEW_TOKENS = int(os.getenv("SENTINEL_MAX_TOKENS", "120"))

_state: Dict[str, Any] = {"runner": None, "retriever": None, "load_ms": None}


def _load_models() -> None:
    """
    Loaded once at startup. Previously the SentenceTransformer, CrossEncoder and
    Milvus client were constructed lazily INSIDE the async generator on the first
    request, which blocked the event loop for ~25 s and made the first query take
    30 s. Nothing heavy may be constructed on the request path.
    """
    t0 = time.perf_counter()
    try:
        # Ollama serves Granite as GGUF through llama.cpp's Metal backend. The
        # previous transformers/MPS runner held granite-3.3-2b in bf16 at 4.7 GB
        # on a 16 GB machine and decoded at 260-480 ms/token; a quantized 8B fits
        # in about the same memory with ~4x the parameters. Streaming logprobs
        # with top-k are available per chunk, which is the hard requirement.
        from ollama_runner import OllamaRunner
        _state["runner"] = OllamaRunner()
    except Exception as err:
        print(f"[Sentinel] Granite unavailable: {err}")

    try:
        from phase3_rag_retriever import SentinelRAGRetriever
        _state["retriever"] = SentinelRAGRetriever()
    except Exception as err:
        print(f"[Sentinel] Retriever unavailable: {err}")

    # Force the weights into VRAM before the first query rather than during it.
    # Constructing OllamaRunner only checks /api/tags, which does not load
    # anything - so without this the model loaded on the first token the judges
    # were watching for. Measured: 10.9 s cold versus 0.8 s warm on the same
    # query. One generation of one token is enough to trigger the load, and
    # keep_alive=-1 in the runner stops it being evicted again.
    if _state["runner"]:
        try:
            async def _warm():
                async for _ in _state["runner"].stream("ok", max_new_tokens=1):
                    break
            asyncio.run(_warm())
        except Exception as err:
            print(f"[Sentinel] warm-up generation failed: {err}")

    _state["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[Sentinel] Warm in {_state['load_ms']} ms")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_running_loop().run_in_executor(None, _load_models)
    yield


app = FastAPI(title="Sentinel-RAG Interception Proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse(kind: str, **fields) -> str:
    return f"data: {json.dumps({'kind': kind, **fields})}\n\n"


# Decoding used to run on a worker thread behind a global lock, drained through
# an asyncio.Queue: the transformers runner held the weights in this process, so
# iterating it inline stalled the event loop and two requests could interleave
# forward passes on one MPS context. Ollama owns the model in its own process,
# which makes generation network I/O - awaited directly, with no thread, no
# queue and no lock, and nothing left to serialise on this side.


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health_check():
    """Reports measured state. The former 11.4 ms figure was a literal."""
    prof = cm.process_profile()
    # Imported here, not at module scope: it pulls pymilvus and
    # sentence-transformers, which would land on `import main` and slow every
    # test that only touches an endpoint.
    from phase2_ingestion_pipeline import PROVENANCE_PATH
    # Which engine redacted the vectors being SERVED, read from what the ingest
    # recorded. Reporting this process's own PII_ENGINE was wrong in a way that
    # always understated the system: DPK pins transformers to 4.57.6 through
    # flair, so it deliberately lives in .venv-dpk and the server can never
    # import it - /health said "regex-fallback" no matter what built the store.
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text())
    except (OSError, ValueError):
        provenance = {}
    return {
        "status": "NOMINAL" if _state["runner"] else "DEGRADED_NO_MODEL",
        "model": os.getenv("OLLAMA_MODEL", "granite3.3:8b"),
        "backend": "ollama",
        "model_loaded": _state["runner"] is not None,
        "retriever_loaded": _state["retriever"] is not None,
        # The UI used to assert "Data Prep Kit" unconditionally. An unverifiable
        # claim in the trust panel of a hallucination firewall is the one place
        # it cannot sit, so this names the engine or says it does not know.
        "pii_engine": provenance.get("pii_engine", "unknown-not-ingested"),
        "vector_store": "milvus-lite" if getattr(
            _state["retriever"], "has_milvus", False) else "in-memory-fallback",
        "warmup_ms": _state["load_ms"],
        "circuit_breaker_tau": TAU,
        "event_log_cases": prof["total_cases"],
        "event_log_events": prof["total_events"],
    }


@app.get("/graphs")
async def list_graphs():
    """
    The single process graph actually present in the event log.

    The previous version advertised four collections (celonis_p2p_chunks,
    celonis_o2c_chunks, ...) with invented vector counts. None existed: the only
    Milvus collection is celonis_ground_truth, and the data is Order-to-Cash.
    """
    prof = cm.process_profile()
    return {
        "o2c": {
            "name": prof["process"],
            "collection": "celonis_ground_truth",
            "source_system": prof["source_system"],
            "vector_count": prof["total_events"],
            "cases": prof["total_cases"],
            "pii_masked": True,
            "mean_cycle_days": prof["mean_cycle_days"],
            "declared_compliance_cycle_days": prof["declared_avg_compliance_cycle_time_days"],
            "activities": {k: v["event_count"] for k, v in prof["by_activity"].items()},
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Ground-truth aggregates, computed from the event log on request."""
    return JSONResponse(cm.process_profile())


@app.get("/stream")
async def stream_tokens(query: str = Query(None), mode: str = Query("ab")):
    """
    mode=deployed (default) - Sentinel as it would actually be deployed:
        retrieved context and the grounding instruction go to the model, and the
        audit runs on top. Answerable questions stream through untouched;
        fabrications are still caught, because the audit is downstream of the
        prompt and does not care how the model was asked.

    mode=ab - the controlled experiment: the protected panel decodes from the
        SAME context-free prompt as /unprotected_stream, so exactly one variable
        differs between the two panels - the proxy itself. Retrieval is deferred
        to the recovery pass.

    `ab` is the default because it is the architecture CLAUDE.md describes and
    the only one that can demonstrate it: the right panel must be somewhere it
    CAN hallucinate before entropy can spike, the breach can be logged, and the
    self-healing agent can be triggered. Under `deployed` the model is handed the
    figure in its context and simply answers, so neither detection layer ever
    fires - a flat graph and an empty recovery panel, which reads to an audience
    as a product that does nothing.

    This default was previously `deployed`, because `ab` had been observed
    tripping on all four answerable demo prompts - every one recovering to a
    correct answer, but a detector that fires on every input demonstrates no
    discrimination. That observation predates the grounding work (46 audited
    defects, then figure_unit), which removed the false positives at their
    source rather than by avoiding the configuration that exposed them; see
    test_grounding_regression. Re-measure with demo_run_of_show.py before
    trusting either number.
    """
    # `graph` is no longer a parameter: the log contains exactly one process.
    # FastAPI ignores the extra query param older clients still send.
    q = query or "What is the mean compliance cycle time?"
    grounded = mode != "ab"

    async def gen():
        if not _state["runner"]:
            yield sse("error", message="Granite model not loaded")
            return
        stream = SentinelStream(
            runner=_state["runner"],
            retriever=_state["retriever"],
            tau=TAU,
            window_size=WINDOW,
            max_new_tokens=MAX_NEW_TOKENS,
            grounded_prompt=grounded,
        )
        try:
            async for ev in stream.run(q):
                yield sse(ev.kind, text=ev.text, **ev.payload)
        except Exception as err:
            yield sse("error", message=str(err))

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/unprotected_stream")
async def stream_unprotected(query: str = Query(None)):
    """
    The baseline: identical model, no retrieved context, no interception.

    Nothing here is scripted. Denied the event log, Granite fills the gap from
    parametric memory - which is the behaviour the whole project exists to stop.
    """
    q = query or "What is the mean compliance cycle time?"

    async def gen():
        runner = _state["runner"]
        if not runner:
            yield sse("error", message="Granite model not loaded")
            return
        engine = EntropyEngine(threshold_tau=TAU, window_size=WINDOW)
        prompt = runner.build_prompt(q, context=None, grounded=False)
        t0 = time.perf_counter()
        count = 0
        try:
            async for step in runner.stream(prompt, max_new_tokens=MAX_NEW_TOKENS):
                # Scored identically to the protected side, but never acted upon,
                # so the terminal can show what would have been caught.
                _, uncertainty, variance = engine.evaluate_token(
                    step.text, logprob=step.logprob, top_probs=step.top_probs
                )
                # Same divergence measure as the protected path. Both panels have
                # to be scored the same way or the split-screen graph compares
                # two different quantities and proves nothing.
                divergence = sd.analyse(step.top_tokens, step.top_probs)
                count += 1
                yield sse(
                    "token",
                    text=step.text,
                    logprob=round(step.logprob, 4),
                    probability=round(step.prob, 4),
                    uncertainty=round(uncertainty, 4),
                    rolling_variance=round(variance, 4),
                    shannon_entropy=round(engine.compute_shannon_entropy(step.top_probs), 4),
                    semantic_entropy=round(divergence.semantic_entropy, 4),
                    figure_at_stake=divergence.divergent,
                    index=step.index,
                )
            yield sse(
                "done",
                text="",
                tokens=count,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                intercepted=False,
            )
        except Exception as err:
            yield sse("error", message=str(err))

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/recover")
async def recover(payload: Dict[str, Any]):
    """
    Autonomous recovery. Answers strictly from the event log, so the repaired
    response cannot itself contain a figure the data does not support.
    """
    query = payload.get("query", "")
    answer = cm.ground_truth_answer(query)
    chunks = []
    if _state["retriever"]:
        try:
            chunks = _state["retriever"].format_granite_context(query).get("chunks", [])[:3]
        except Exception:
            pass
    return JSONResponse({
        "status": "RECOVERED",
        "strategy": "deterministic_event_log_lookup",
        "verified_ground_truth": answer,
        "source": "mock_celonis_data_large.json",
        "supporting_chunks": chunks,
        "all_figures_grounded": True,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
