import math
import hashlib
import json
import os
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

# Try importing dependencies; fall back to mock implementations if libraries are missing.
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from ibm_watsonx_ai.foundation_models import Model
except ImportError:
    Model = None

AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.json"

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
    Manages the local CrossEncoder entailment model to cluster continuations
    semantically, calculate Shannon entropy, and maintain watsonx clients.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        watsonx_credentials: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None
    ):
        self.model_name = model_name
        self.cross_encoder = None
        if CrossEncoder is not None:
            try:
                print(f"[Semantic_Entropy_Engine] Initializing local cross-encoder: {model_name}")
                self.cross_encoder = CrossEncoder(model_name)
            except Exception as e:
                print(f"[Semantic_Entropy_Engine] Error loading CrossEncoder: {e}. Falling back to cosine similarity rules.")
        
        # Initialize Ollama client parameters for Granite models
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.primary_model = "granite3-dense:8b"
        self.sampler_model = "granite3-dense:2b"

    def cluster_and_compute_entropy(self, continuations: List[str]) -> float:
        """
        Uses NLI entailment to cluster K continuations and compute cluster-level Shannon Entropy.
        Paths expressing the same fact (strong bi-directional entailment/agreement) belong to the same cluster.
        """
        K = len(continuations)
        if K <= 1:
            return 0.0

        clusters: List[List[int]] = []
        for i in range(K):
            placed = False
            for cluster in clusters:
                rep_idx = cluster[0]
                agreement = self._check_agreement(continuations[i], continuations[rep_idx])
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

    def _check_agreement(self, text_a: str, text_b: str) -> bool:
        """
        Helper using the CrossEncoder to test if text_a and text_b entail each other.
        If CrossEncoder is not available, defaults to a word overlap threshold.
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
            scores = self.cross_encoder.predict(pairs)
            if len(scores.shape) > 1 and scores.shape[1] >= 2:
                # Calculate softmax probabilities across [contradiction, entailment, neutral]
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
                
                # Check contradiction probabilities
                contra_a_b = float(probs[0][0])
                contra_b_a = float(probs[1][0])
                
                # Agreement means they do NOT contradict each other (neither is contradiction)
                return contra_a_b < 0.20 and contra_b_a < 0.20
            else:
                return float(scores[0]) > 0.0 and float(scores[1]) > 0.0
        except Exception:
            return len(set(text_a.lower().split()) & set(text_b.lower().split())) >= 2



    async def _fetch_single_ollama_continuation(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> str:
        """Helper to fetch a single completion path via Ollama's OpenAI-compatible HTTP API."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.ollama_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.ollama_api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.sampler_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "seed": seed
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            # Fallback to requests if httpx is not present or errors out
            try:
                import requests
                def _sync_post():
                    r = requests.post(
                        f"{self.ollama_base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.ollama_api_key}", "Content-Type": "application/json"},
                        json={
                            "model": self.sampler_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "seed": seed
                        },
                        timeout=5.0
                    )
                    if r.status_code == 200:
                        return r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    return ""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _sync_post)
            except Exception:
                pass
        return f"{prompt} continuation variant."

    async def generate_k_paths(self, prompt: str, K: int = 3, max_tokens: int = 10) -> List[str]:
        """
        Asynchronously generates K=3 continuations using Ollama (granite3-dense:2b) in parallel.
        """
        tasks = [
            self._fetch_single_ollama_continuation(
                prompt=f"Complete this response in 10 words or less: {prompt}",
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
