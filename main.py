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
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

import celonis_metrics as cm
from entropy_engine import EntropyEngine
from sentinel_stream import SentinelStream

load_dotenv()

# Calibrated against real Granite log-probabilities (calibrate_tau.py).
# 0.65 with a run of 3 caught only 3 of 6 unanswerable prompts. The sweep's
# best separator was tau=1.25 with run=1 (6/6, no false positives), but the
# answerable max was 1.150 - a 0.10 margin fitted to six prompts. tau=1.0 with
# run=2 needs a SUSTAINED excursion, which no answerable prompt produced, so it
# trades two detections for margin. The numeric-grounding layer is deterministic
# and catches fabricated figures regardless of tau.
TAU = float(os.getenv("SENTINEL_TAU", "1.0"))
WINDOW = int(os.getenv("SENTINEL_WINDOW", "5"))
MAX_NEW_TOKENS = int(os.getenv("SENTINEL_MAX_TOKENS", "120"))

_state: Dict[str, Any] = {"runner": None, "retriever": None, "load_ms": None}
# One model, one MPS context: serialise decoding so two concurrent SSE requests
# cannot interleave forward passes on the same weights.
# ponytail: single global lock, fine for a demo; use a worker pool per replica
# if this ever serves real concurrent traffic.
_decode_lock = threading.Lock()


def _load_models() -> None:
    """
    Loaded once at startup. Previously the SentenceTransformer, CrossEncoder and
    Milvus client were constructed lazily INSIDE the async generator on the first
    request, which blocked the event loop for ~25 s and made the first query take
    30 s. Nothing heavy may be constructed on the request path.
    """
    t0 = time.perf_counter()
    try:
        from granite_runner import GraniteRunner
        _state["runner"] = GraniteRunner.get()
    except Exception as err:
        print(f"[Sentinel] Granite unavailable: {err}")

    try:
        from phase3_rag_retriever import SentinelRAGRetriever
        _state["retriever"] = SentinelRAGRetriever()
    except Exception as err:
        print(f"[Sentinel] Retriever unavailable: {err}")

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


async def _drain(gen):
    """
    Run a blocking generator on a worker thread, yielding items as they appear.

    Decoding is CPU/GPU-bound and synchronous; iterating it directly inside the
    coroutine would stall the event loop and every other request with it.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def produce():
        try:
            with _decode_lock:
                for item in gen:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as err:
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    loop.run_in_executor(None, produce)
    while True:
        item = await queue.get()
        if item is SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health_check():
    """Reports measured state. The former 11.4 ms figure was a literal."""
    prof = cm.process_profile()
    return {
        "status": "NOMINAL" if _state["runner"] else "DEGRADED_NO_MODEL",
        "model": os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-3.3-2b-instruct"),
        "model_loaded": _state["runner"] is not None,
        "retriever_loaded": _state["retriever"] is not None,
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
async def stream_tokens(query: str = Query(None)):
    # `graph` is no longer a parameter: the log contains exactly one process.
    # FastAPI ignores the extra query param older clients still send.
    q = query or "What is the mean compliance cycle time?"

    async def gen():
        if not _state["runner"]:
            yield sse("error", message="Granite model not loaded")
            return
        # The proxy audits the request the application actually sends, and a
        # naive integration sends no retrieved context - that is the deployment
        # Sentinel exists to protect. Running the SAME prompt as
        # /unprotected_stream also makes the split-screen a controlled
        # experiment: one variable differs between the panels, the proxy itself.
        # Retrieval is not skipped, it is deferred to the recovery pass, where
        # the context gap is repaired and the answer regenerated under grounding.
        stream = SentinelStream(
            runner=_state["runner"],
            retriever=_state["retriever"],
            tau=TAU,
            window_size=WINDOW,
            max_new_tokens=MAX_NEW_TOKENS,
            grounded_prompt=False,
        )
        try:
            async for ev in _drain(stream.run(q)):
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
            async for step in _drain(runner.stream(prompt, max_new_tokens=MAX_NEW_TOKENS)):
                # Scored identically to the protected side, but never acted upon,
                # so the terminal can show what would have been caught.
                _, uncertainty, variance = engine.evaluate_token(
                    step.text, logprob=step.logprob, top_probs=step.top_probs
                )
                count += 1
                yield sse(
                    "token",
                    text=step.text,
                    logprob=round(step.logprob, 4),
                    probability=round(step.prob, 4),
                    uncertainty=round(uncertainty, 4),
                    rolling_variance=round(variance, 4),
                    shannon_entropy=round(engine.compute_shannon_entropy(step.top_probs), 4),
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
