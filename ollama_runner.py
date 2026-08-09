"""
Granite decoding via Ollama, with real streaming log-probabilities.

Drop-in for GraniteRunner: same build_prompt/stream contract, same TokenStep,
so sentinel_stream.py and main.py need no changes beyond which runner they
construct.

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
exists to replace. Verified against Ollama 0.31.1: /api/generate with
stream=true, logprobs=true, top_logprobs=5 returns a logprobs entry on every
chunk, each carrying the chosen token's logprob plus five alternatives.
"""

import json
import math
import os
from typing import Iterator, List, Optional

import requests

from granite_runner import TokenStep

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
TOP_K = 5


class OllamaRunner:
    """Same interface as GraniteRunner, backed by Ollama."""

    def __init__(self, model: str = MODEL_ID, host: str = OLLAMA_HOST, timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._check()

    def _check(self) -> None:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
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

    # --- identical contract to GraniteRunner -------------------------------

    def build_prompt(self, query: str, context: Optional[str] = None, grounded: bool = True) -> str:
        """
        Returns a JSON envelope rather than a formatted string.

        Ollama applies the model's own chat template server-side via /api/chat,
        so hand-rolling one here would double-apply it. stream() unpacks this.
        """
        if grounded:
            system = (
                "You are a Celonis process-mining analyst. Answer ONLY from the provided "
                "event-log context. Every figure you state must appear in the context. "
                "Prefer the pre-computed aggregates when the question asks for one; do not "
                "recompute them from individual cases. If the context does not contain the "
                "answer, say you cannot verify it."
            )
            user = f"Context from Celonis event log:\n{context}\n\nQuestion: {query}"
        else:
            # The naive integration: asserts database access it does not have.
            system = (
                "You are the Celonis EMS analytics copilot with direct query access to "
                "the live enterprise event log and all process graphs. You always have "
                "the data. Never say you lack access and never ask for clarification. "
                "Answer in one or two sentences, always citing specific numeric figures "
                "such as cycle times in days, percentages, and dollar amounts."
            )
            user = query
        return json.dumps({"system": system, "user": user})

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_k: int = TOP_K,
    ) -> Iterator[TokenStep]:
        envelope = json.loads(prompt)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": envelope["system"]},
                {"role": "user", "content": envelope["user"]},
            ],
            "stream": True,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {"num_predict": max_new_tokens, "temperature": temperature},
        }
        resp = requests.post(
            f"{self.host}/api/chat", json=body, stream=True, timeout=self.timeout
        )
        resp.raise_for_status()

        index = 0
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if chunk.get("done"):
                break

            entries = chunk.get("logprobs") or []
            if not entries:
                continue
            for entry in entries:
                text = entry.get("token", "")
                if not text:
                    continue
                logprob = float(entry.get("logprob", 0.0))
                alts = entry.get("top_logprobs") or []
                probs = [math.exp(float(a["logprob"])) for a in alts if "logprob" in a]
                total = sum(probs)
                # Renormalise the truncated head so it is a proper distribution
                # for H(P), matching what the transformers runner produced.
                top_probs = sorted((p / total for p in probs), reverse=True) if total > 0 else [math.exp(logprob)]

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


def demo() -> None:
    r = OllamaRunner()
    steps = list(r.stream(r.build_prompt("Name three colours."), max_new_tokens=24))
    assert steps, "no tokens returned"
    lps = [s.logprob for s in steps]
    assert all(lp <= 0.0 for lp in lps), "log-probabilities must be <= 0"
    assert len(set(round(lp, 4) for lp in lps)) > 1, "logprobs constant - not real output"
    for s in steps[:10]:
        assert abs(sum(s.top_probs) - 1.0) < 1e-4, "top_probs is not a distribution"
        print(f"  {s.index:>3} {s.text!r:<14} logprob={s.logprob:>8.4f} p={s.prob:.4f} k={len(s.top_probs)}")
    print(f"\nOK: {len(steps)} tokens, logprob range [{min(lps):.3f}, {max(lps):.3f}]")


if __name__ == "__main__":
    demo()
