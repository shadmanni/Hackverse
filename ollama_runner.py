"""
Granite decoding via Ollama, with real streaming log-probabilities.

Why this replaced the transformers path
---------------------------------------
The transformers runner loaded granite-3.3-2b in bf16 at 4.7 GB resident on a
16 GB machine, decoding at 260-480 ms/token through PyTorch MPS. Ollama serves
GGUF through llama.cpp's Metal backend: a quantized 8B lands in roughly the same
memory as the 2B did unquantized, so the model gets ~4x the parameters at no
memory cost, and the C++ backend avoids the MPS dispatch overhead.

The requirement that decided it is streaming logprobs. The entropy engine needs
log P(token) AND the top-k distribution PER TOKEN, DURING generation - an audit
that arrives after generation is exactly the post-hoc evaluation this project
exists to replace. Verified against Ollama 0.31.1: /api/chat with stream=true,
logprobs=true, top_logprobs=5 returns a logprobs entry on every chunk, each
carrying the chosen token's logprob plus five alternatives.

Decoding is asynchronous. Ollama runs the model in its own process, so from
here generation is network I/O, not compute: awaiting it leaves the event loop
free instead of parking a worker thread on a blocking socket read.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
TOP_K = 5
# -1 pins the weights in VRAM until Ollama is told otherwise. Overridable for
# anyone who has to share the GPU; the demo machine should leave it alone.
#
# int, not str: /api/chat accepts a number of seconds or a duration string like
# "5m", and rejects the bare string "-1" with a 400 that fails EVERY generation,
# not just the warm-up. Read from the env it is always a string, so cast here.
KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))


@dataclass
class TokenStep:
    """One decoding step, with the model's real distribution attached."""
    text: str
    logprob: float           # log P(chosen token | context)
    top_probs: List[float]   # renormalised top-k probabilities, for Shannon entropy
    top_tokens: List[str]
    index: int

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


class OllamaRunner:
    """Streams Granite from a local Ollama daemon, one TokenStep per token."""

    def __init__(self, model: str = MODEL_ID, host: str = OLLAMA_HOST, timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._check()

    def _check(self) -> None:
        """
        Fail loudly at construction if the model is missing.

        Deliberately not a silent fall-back to whatever else is pulled: tau is
        calibrated per model (calibrate_tau.py), so quietly answering from a
        different one would run the circuit breaker at a threshold that was
        never measured for it, and the first symptom would be false positives
        mid-demo with no visible cause.
        """
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=10).json()
        except Exception as err:
            raise RuntimeError(
                f"Ollama unreachable at {self.host}. Is the daemon running? ({err})"
            ) from err
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(
                f"Model {self.model!r} not pulled. Run: ollama pull {self.model}\n"
                f"Available: {sorted(names) or 'none'}"
            )
        print(f"[OllamaRunner] Ready: {self.model} via {self.host}")

    def build_prompt(
        self, query: str, context: Optional[str] = None, grounded: bool = True
    ) -> List[Dict[str, str]]:
        """
        Chat messages, not a formatted string: Ollama applies the model's own
        chat template server-side, so hand-rolling one here would double-apply it.

        :param grounded: with context and the grounding instruction (the protected
            path), or without either (the unprotected baseline).

            The baseline is not a different model and not a script. It is this
            same Granite, denied the event log and told to answer directly - so
            it answers from parametric memory and invents plausible enterprise
            figures on its own. That is the behaviour being demonstrated, and it
            has to be genuine for the comparison to mean anything.
        """
        if grounded:
            system = (
                "You are a Celonis process-mining analyst. Answer ONLY from the provided "
                "event-log context. Every figure you state must appear in the context. "
                "Prefer the pre-computed aggregates when the question asks for one; do not "
                "recompute them from individual cases. If the context does not contain the "
                "answer, say you cannot verify it. "
                # Answer, then stop. Without this the model appends a citation -
                # "This information is provided in the RETRIEVED EVENT CHUNKS
                # section under..." - and that trailing clause is legitimately
                # high-entropy: there are many ways to word it and the model is
                # genuinely unsure which section name to use. It tripped the
                # entropy layer at token 30 on an answer that was correct and
                # complete at token 12. The uncertainty was real; it just was not
                # about the fact. Suppressing the filler removes the
                # false-positive surface at its source rather than by raising tau.
                "Answer in one short sentence stating the figure. Do not explain "
                "where in the context you found it."
            )
            user = f"Context from Celonis event log:\n{context}\n\nQuestion: {query}"
        else:
            # Deliberately the naive integration: a system prompt that asserts
            # database access and demands concrete figures, with no retrieved
            # context behind it. This is how these deployments are actually
            # written, and it is what turns a well-aligned model into a confident
            # fabricator - the model has no data, but has been told it does.
            system = (
                "You are the Celonis EMS analytics copilot with direct query access to "
                "the live enterprise event log and all process graphs. You always have "
                "the data. Never say you lack access and never ask for clarification. "
                "Answer in one or two sentences, always citing specific numeric figures "
                "such as cycle times in days, percentages, and dollar amounts."
            )
            user = query
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def stream(
        self,
        prompt: Any,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_k: int = TOP_K,
    ) -> AsyncIterator[TokenStep]:
        body = {
            "model": self.model,
            "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}],
            "stream": True,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {"num_predict": max_new_tokens, "temperature": temperature},
            # Never evict the model. Ollama's default is 5 minutes, which is
            # exactly the length of a conversation with a judge: measured cold,
            # the same query that answers in 0.8 s took 10.9 s, all of it
            # reloading 5.71 GB into VRAM. The warm-up in main.py pays that cost
            # once at startup and this keeps it paid.
            "keep_alive": KEEP_ALIVE,
        }
        index = 0
        # A client per call, not a shared one: an AsyncClient binds to the event
        # loop that created it, and the test suite runs each case in a fresh loop.
        # ponytail: reuse a client per loop if connection setup ever shows up in
        # the token-latency numbers; against localhost it does not.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.host}/api/chat", json=body) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break

                    for entry in chunk.get("logprobs") or []:
                        text = entry.get("token", "")
                        if not text:
                            continue
                        logprob = float(entry.get("logprob", 0.0))
                        alts = entry.get("top_logprobs") or []
                        probs = [math.exp(float(a["logprob"])) for a in alts if "logprob" in a]
                        total = sum(probs)
                        # Renormalise the truncated head so it is a proper
                        # distribution for H(P). This understates true entropy -
                        # the discarded tail is exactly where an uncertain model
                        # puts its mass - so the measure is conservative.
                        top_probs = (
                            sorted((p / total for p in probs), reverse=True)
                            if total > 0
                            else [math.exp(logprob)]
                        )

                        yield TokenStep(
                            text=text,
                            logprob=logprob,
                            top_probs=top_probs,
                            top_tokens=[a.get("token", "") for a in alts],
                            index=index,
                        )
                        index += 1

    def unconditioned_logprob(self, token_text: str, partial: str) -> Optional[float]:
        """Not implemented: Ollama exposes no scoring endpoint for a forced token."""
        return None


async def _demo() -> None:
    r = OllamaRunner()
    steps = [s async for s in r.stream(r.build_prompt("Name three colours."), max_new_tokens=24)]
    assert steps, "no tokens returned"
    lps = [s.logprob for s in steps]
    assert all(lp <= 0.0 for lp in lps), "log-probabilities must be <= 0"
    assert len(set(round(lp, 4) for lp in lps)) > 1, "logprobs constant - not real output"
    for s in steps[:10]:
        assert abs(sum(s.top_probs) - 1.0) < 1e-4, "top_probs is not a distribution"
        print(f"  {s.index:>3} {s.text!r:<14} logprob={s.logprob:>8.4f} p={s.prob:.4f} k={len(s.top_probs)}")
    print(f"\nOK: {len(steps)} tokens, logprob range [{min(lps):.3f}, {max(lps):.3f}]")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_demo())
