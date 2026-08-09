"""
Ollama decoding via the openai.AsyncOpenAI client, with real per-token log-probabilities.

Single-stream design: only the primary model (granite3-dense:8b) is called per request.
The K-Path Sampler was removed — Ollama is serial by default so concurrent requests
queue behind the primary and roughly double decode latency with no benefit.
"""

import json
import math
import os

import requests
from typing import AsyncIterator, List, Optional, Dict, Any

from granite_runner import TokenStep

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "granite3-dense:8b")
TOP_K = 5


class OllamaRunner:
    """Ollama AsyncOpenAI runner — single primary stream, no K-path sampler."""

    def __init__(self, model: str = MODEL_ID, host: str = OLLAMA_HOST):
        from openai import AsyncOpenAI

        self.host = host.rstrip("/")
        self.client = AsyncOpenAI(base_url=f"{self.host}/v1", api_key="ollama")
        
        self.primary_model = "granite3-dense:8b"
        self.sampler_model = "granite3-dense:2b"

        # Proactively check available models in local Ollama
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                # If 8b is not found but 2b is, fall back to 2b for primary to prevent crashing
                if "granite3-dense:8b" not in models and "granite3-dense:latest" not in models:
                    if "granite3-dense:2b" in models:
                        print("[OllamaRunner] granite3-dense:8b not found. Falling back to granite3-dense:2b for primary.")
                        self.primary_model = "granite3-dense:2b"
        except Exception as err:
            print(f"[OllamaRunner] Could not query Ollama models: {err}. Defaulting to granite3-dense:8b.")

    def build_prompt(self, query: str, context: Optional[str] = None, grounded: bool = True) -> List[Dict[str, str]]:
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
            system = (
                "You are the Celonis EMS analytics copilot with direct query access to "
                "the live enterprise event log and all process graphs. You always have "
                "the data. Never say you lack access and never ask for clarification. "
                "Answer in one or two sentences, always citing specific numeric figures "
                "such as cycle times in days, percentages, and dollar amounts."
            )
            user = query

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    async def stream(
        self,
        prompt: Any,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_k: int = TOP_K,
    ) -> AsyncIterator[TokenStep]:
        """
        Stream tokens directly from the primary model with no secondary sampler.

        The K-Path Sampler (concurrent 2b request) was removed: Ollama is serial
        by default, so a second concurrent request queues behind the primary and
        creates back-pressure that roughly doubles decode latency. The sampler
        output was discarded anyway (async for _ in stream: pass), so removing it
        costs nothing and cuts first-token latency significantly.
        """
        # Resolve messages format (JSON envelope from legacy build_prompt or plain list)
        if isinstance(prompt, str):
            try:
                envelope = json.loads(prompt)
                messages = [
                    {"role": "system", "content": envelope.get("system", "")},
                    {"role": "user", "content": envelope.get("user", "")}
                ]
            except Exception:
                messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        primary_stream = await self.client.chat.completions.create(
            model=self.primary_model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
            stream=True,
            logprobs=True,
            top_logprobs=top_k,
        )

        step_idx = 0
        async for chunk in primary_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            logprobs_data = choice.logprobs

            text = (choice.delta.content or "")
            if not text and not logprobs_data:
                continue

            logprob_val = 0.0
            top_probs = [1.0]
            top_tokens = [text]

            if logprobs_data and logprobs_data.content:
                lp_item = logprobs_data.content[0]
                text = lp_item.token
                logprob_val = lp_item.logprob
                if lp_item.top_logprobs:
                    raw_probs = [math.exp(t.logprob) for t in lp_item.top_logprobs]
                    total = sum(raw_probs)
                    top_probs = [p / total for p in raw_probs] if total > 0 else raw_probs
                    top_tokens = [t.token for t in lp_item.top_logprobs]

            yield TokenStep(
                text=text,
                logprob=logprob_val,
                top_probs=top_probs,
                top_tokens=top_tokens,
                index=step_idx,
            )
            step_idx += 1

    def unconditioned_logprob(self, token_text: str, partial: str) -> Optional[float]:
        return None
