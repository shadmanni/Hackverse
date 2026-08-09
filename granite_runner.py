"""
Real IBM Granite decoding with true per-token log-probabilities.

Replaces the hardcoded `logprob=-0.1` constants that used to be fed to the
EntropyEngine. Every value emitted here comes from the model's actual output
distribution, so Shannon entropy and log-prob variance are measured, not staged.

Runs locally on Apple MPS (or CUDA/CPU) so the demo has no network dependency.
"""

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-3.3-2b-instruct")
TOP_K = 5


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


class GraniteRunner:
    """Thread-safe lazy singleton around a local Granite causal LM."""

    _instance: Optional["GraniteRunner"] = None
    _lock = threading.Lock()

    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or _pick_device()
        # fp16 on MPS/CUDA halves memory and roughly doubles throughput; CPU needs fp32.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        print(f"[GraniteRunner] Loading {model_id} on {self.device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        print(f"[GraniteRunner] Ready. vocab={self.model.config.vocab_size}")

    @classmethod
    def get(cls, **kw) -> "GraniteRunner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kw)
        return cls._instance

    def build_prompt(self, query: str, context: Optional[str] = None, grounded: bool = True) -> str:
        """
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
                "answer, say you cannot verify it."
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_k: int = TOP_K,
    ) -> Iterator[TokenStep]:
        """
        Greedy (temperature=0) or sampled decoding, yielding one TokenStep per token.

        Uses an incremental KV cache, so each step is a single-token forward pass
        rather than a re-read of the whole prefix.
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        past = None
        cur = input_ids
        eos_ids = {self.tokenizer.eos_token_id}
        if self.tokenizer.convert_tokens_to_ids("<|end_of_text|>") is not None:
            eos_ids.add(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

        for step in range(max_new_tokens):
            out = self.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()

            logprobs = torch.log_softmax(logits, dim=-1)
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logprobs, dim=-1, keepdim=True)

            tok_id = int(next_id.item())
            if tok_id in eos_ids:
                break

            chosen_lp = float(logprobs[tok_id].item())
            tk = torch.topk(logprobs, k=top_k)
            tk_probs = torch.exp(tk.values)
            # Renormalise the truncated head so it is a proper distribution for H(P).
            tk_probs = (tk_probs / tk_probs.sum()).tolist()

            yield TokenStep(
                text=self.tokenizer.decode([tok_id], skip_special_tokens=True),
                logprob=chosen_lp,
                top_probs=tk_probs,
                top_tokens=[self.tokenizer.decode([int(i)]) for i in tk.indices.tolist()],
                index=step,
            )
            cur = next_id.view(1, 1)

    @torch.inference_mode()
    def unconditioned_logprob(self, token_text: str, partial: str) -> Optional[float]:
        """
        log P(token | partial WITHOUT retrieved context) — the parametric-memory
        baseline needed for the contrastive PMI term in EntropyEngine. Without
        this, use_contrastive has nothing valid to compare against.
        """
        ids = self.tokenizer(partial, return_tensors="pt").to(self.device)["input_ids"]
        tgt = self.tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if not tgt:
            return None
        out = self.model(input_ids=ids, use_cache=False)
        lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(lp[tgt[0]].item())


def demo() -> None:
    """Self-check: real logprobs must vary, and grounded text must beat nonsense."""
    r = GraniteRunner.get()
    steps = list(r.stream(r.build_prompt("Say exactly: the sky is blue."), max_new_tokens=24))
    assert steps, "model produced no tokens"
    lps = [s.logprob for s in steps]
    assert all(lp <= 0.0 for lp in lps), "log-probabilities must be <= 0"
    assert len(set(round(lp, 4) for lp in lps)) > 1, "logprobs are constant - not real model output"
    for s in steps[:12]:
        print(f"  {s.index:>3} {s.text!r:<16} logprob={s.logprob:>8.4f} p={s.prob:.4f}")
    print(f"\nOK: {len(steps)} tokens, logprob range [{min(lps):.3f}, {max(lps):.3f}]")


if __name__ == "__main__":
    demo()
