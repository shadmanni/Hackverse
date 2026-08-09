import math
import hashlib
import json
import os
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import openai

AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.json"

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# ===========================================================================
# 1. EMA_Firewall Class
# ===========================================================================
class EMA_Firewall:
    """
    Manages the running Exponential Moving Average (EMA) state of semantic
    entropy values to smooth out single-token syntactic variances.
    """
    def __init__(self, alpha: float = 0.4, threshold: float = 0.65):
        self.alpha = alpha
        self.threshold = threshold
        self.s_t: float = 0.0
        self.initialized = False

    def update(self, raw_se: float) -> float:
        """
        Applies: S_t = (alpha * raw_se) + ((1 - alpha) * S_{t-1})
        """
        if not self.initialized:
            self.s_t = raw_se
            self.initialized = True
        else:
            self.s_t = (self.alpha * raw_se) + ((1.0 - self.alpha) * self.s_t)
        return self.s_t

    def check_breach(self) -> bool:
        return self.s_t >= self.threshold


# ===========================================================================
# 2. Semantic_Entropy_Engine Class
# ===========================================================================
class Semantic_Entropy_Engine:
    """
    Manages local CrossEncoder entailment clustering and AsyncOpenAI K-path sampling via Ollama.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama"
    ):
        self.model_name = model_name
        self.cross_encoder = None
        if CrossEncoder is not None:
            try:
                print(f"[Semantic_Entropy_Engine] Initializing local cross-encoder: {model_name}")
                self.cross_encoder = CrossEncoder(model_name)
            except Exception as e:
                print(f"[Semantic_Entropy_Engine] Error loading CrossEncoder: {e}")
        
        # Configure AsyncOpenAI Client for Ollama
        self.base_url = base_url
        self.api_key = api_key
        self.openai_client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self.primary_model = "granite3-dense:8b"
        self.sampler_model = "granite3-dense:2b"

    async def cluster_and_compute_entropy(self, continuations: List[str]) -> float:
        """
        Uses NLI entailment to cluster K continuations and compute cluster-level Shannon Entropy.
        """
        K = len(continuations)
        if K <= 1:
            return 0.0

        clusters: List[List[int]] = []
        for i in range(K):
            placed = False
            for cluster in clusters:
                rep_idx = cluster[0]
                agreement = await self._check_agreement(continuations[i], continuations[rep_idx])
                if agreement:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        se = 0.0
        for cluster in clusters:
            p_c = len(cluster) / K
            se -= p_c * math.log2(p_c)

        return se

    async def _check_agreement(self, text_a: str, text_b: str) -> bool:
        """
        Helper using CrossEncoder to check NLI non-contradiction between text_a and text_b.
        Runs in worker thread via asyncio.to_thread to keep main loop non-blocking.
        """
        if self.cross_encoder is None:
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if not words_a or not words_b:
                return False
            overlap = len(words_a.intersection(words_b)) / max(len(words_a), len(words_b))
            return overlap >= 0.65

        try:
            import numpy as np
            pairs = [(text_a, text_b), (text_b, text_a)]
            scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)
            if len(scores.shape) > 1 and scores.shape[1] >= 2:
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
                contra_a_b = float(probs[0][0])
                contra_b_a = float(probs[1][0])
                return contra_a_b < 0.20 and contra_b_a < 0.20
            else:
                return float(scores[0]) > 0.0 and float(scores[1]) > 0.0
        except Exception:
            return len(set(text_a.lower().split()) & set(text_b.lower().split())) >= 2

    async def _fetch_single_continuation(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> str:
        """Fetches a single continuation using AsyncOpenAI client targeting granite3-dense:2b."""
        try:
            resp = await self.openai_client.chat.completions.create(
                model=self.sampler_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            if resp.choices:
                return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Semantic_Entropy_Engine] Ollama sampling error: {e}")
        return f"{prompt} continuation variant."

    async def generate_k_paths(self, prompt: str, K: int = 3, max_tokens: int = 10) -> List[str]:
        """
        Asynchronously generates K=3 continuations inside asyncio.gather() using granite3-dense:2b at temperature=0.7.
        """
        tasks = [
            self._fetch_single_continuation(
                prompt=f"Complete in 10 words or less: {prompt}",
                max_tokens=max_tokens,
                temperature=0.7,
                seed=42 + i
            )
            for i in range(K)
        ]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]


# ===========================================================================
# 3. Compliance Logger Utility
# ===========================================================================
def log_compliance_breach(query: str, ema_score: float):
    """
    Appends a hashed query record to audit_log.json for ZK compliance.
    """
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_sha256": query_hash,
        "action": "INTERCEPTED_AND_HALTED",
        "ema_score": round(ema_score, 4)
    }
    try:
        logs = []
        if AUDIT_LOG_PATH.exists():
            with open(AUDIT_LOG_PATH, "r") as f:
                logs = json.load(f)
        logs.append(record)
        with open(AUDIT_LOG_PATH, "w") as f:
            json.dump(logs, indent=2, fp=f)
    except Exception as e:
        print(f"[Compliance Logger] Error writing audit log: {e}")
