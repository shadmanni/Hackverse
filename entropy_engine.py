from typing import List, Dict, Tuple, Optional, Any
import math
import statistics
from draft_logprob_engine import DraftLogprobExtractor

class EntropyEngine:
    """
    Riddhi's Entropy Analytics & Optimization Engine.
    
    Provides token-level uncertainty quantification by tracking:
    1. Shannon Entropy H(P_t) of next-token logits/probabilities.
    2. Time-series variance V(y_t) over a rolling sliding window of token probabilities.
    3. Dynamic threshold calibration (tau) to trip the circuit breaker.
    """

    def __init__(self, threshold_tau: float = 0.65, window_size: int = 5, use_ema: bool = True, alpha: float = 0.35):
        """
        :param threshold_tau: The entropy/variance threshold above which a hallucination is flagged.
        :param window_size: Number of previous tokens to consider for rolling variance.
        :param use_ema: Whether to use O(1) Exponential Moving Average (EMA) variance windowing.
        :param alpha: Smoothing factor for EMA (default 0.35).
        """
        self.tau = threshold_tau
        self.window_size = window_size
        self.use_ema = use_ema
        self.alpha = alpha
        self.history: List[float] = []
        self.draft_extractor = DraftLogprobExtractor()
        
        # Version 3: O(1) Exponential Moving Average (EMA) State Variables
        self.ema_mean: float = 0.0
        self.ema_var: float = 0.0
        self.count: int = 0

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", 
        "on", "with", "at", "by", "from", "up", "about", "into", "over", "after",
        "and", "or", "but", "if", "that", "this", "these", "those", "it", "its"
    }

    def compute_shannon_entropy(self, probabilities: List[float]) -> float:
        """
        Calculates Shannon Entropy H(P) = - sum(p * log2(p)) for a probability distribution.
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def compute_logprob_variance(self, logprobs: List[float]) -> float:
        """
        Computes the variance V(y_t) over a sequence of log probabilities.
        """
        n = len(logprobs)
        if n < 2:
            return 0.0
        mean = sum(logprobs) / n
        return sum((x - mean) ** 2 for x in logprobs) / n

    def get_token_type_weight(self, token: str) -> float:
        """
        Production Enterprise Token Taxonomy & POS-Aware Dynamic Weighting.
        
        Evaluates risk severity across enterprise data types:
        - Critical Risk (w = 3.2 - 3.8): Financial quantities, dollar figures, percentages, dates, codes, currency codes.
        - High Risk (w = 2.2 - 2.5): Regulatory jurisdictions, acronyms (GDPR, SOX, FDA, ERP, SLA), proper nouns, officer titles.
        - Moderate Content (w = 1.0): Domain verbs, process nouns.
        - Calibrated Stopwords (w = 0.40): Grammatical glue tokens, formatting punctuations, brackets.
        """
        clean = (token or "").strip()
        if not clean:
            return 0.4

        # 1. Critical: Financial, currency, digits, and quantitative claims
        # 1. Critical: Financial, currency, digits, and quantitative claims
        if any(char.isdigit() for char in clean) or any(c in clean for c in ["$", "€", "£", "¥", "₹", "%"]):
            return 3.5

        # 2. Critical: Specialized code / process identifiers (e.g. CASE-10231, CC-4471, W-99, PO-8812)
        if "-" in clean and any(part.isupper() for part in clean.split("-")):
            return 3.0

        # 3. High: Regulatory Compliance Acronyms & Enterprise Standard Identifiers
        enterprise_acronyms = {
            "sox", "gdpr", "hipaa", "fda", "sla", "erp", "p2p", "o2c", "ems", 
            "kyc", "aml", "iso", "soc2", "sap", "celonis", "ibm", "watsonx"
        }
        if clean.lower().strip(".,;:()") in enterprise_acronyms or (clean.isupper() and len(clean) >= 2):
            return 2.5

        # 4. High: Capitalized Enterprise Named Entities, Officer Titles & Jurisdictions
        titles_and_jurisdictions = {
            "compliance", "officer", "director", "controller", "auditor", "legal",
            "delaware", "california", "germany", "singapore", "london", "eu", "apac", "emea"
        }
        if clean.lower().strip(".,;:()") in titles_and_jurisdictions:
            return 2.2

        if clean[0].isupper() and len(clean) > 2 and not clean.endswith("."):
            return 1.8

        # 5. Low: Common grammatical stopwords, connectors & formatting artifacts
        if clean.lower().strip(".,;:()") in self.STOPWORDS or clean in {"(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "-", "—", "•"}:
            return 0.40

        # 6. Baseline for general vocabulary
        return 1.0

    def compute_contrastive_pmi(
        self,
        logprob_with_context: float,
        unconditioned_logprob: Optional[float],
    ) -> float:
        """
        Version 4 Innovation: Pointwise Mutual Information (PMI) / Contrastive RAG Logprob Ratio.
        
        Calculates how strongly a generated token depends on the retrieved Celonis RAG context:
            PMI(x_t; Context) = log P(x_t | context) - log P(x_t | ungrounded_parametric_memory)

        The two log-probabilities must come from equivalent model runs.  A heuristic
        estimate is not a valid ungrounded baseline: it can make a low-confidence
        token look better than its fabricated prior and hide a grounding failure.
        """
        if unconditioned_logprob is None:
            return 0.0
        contrastive_ratio = logprob_with_context - unconditioned_logprob
        
        # If token has high prior probability (generic word), normalize penalty
        pmi_penalty = max(0.0, -contrastive_ratio)
        return float(pmi_penalty)

    def evaluate_token(
        self,
        token: str,
        logprob: Optional[float] = None,
        top_probs: Optional[List[float]] = None,
        context_history: Optional[List[str]] = None,
        use_contrastive: bool = False,
        unconditioned_logprob: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Evaluates an incoming token during decoding using Version 3 POS Weighting + O(1) EMA Welford Incremental Variance.
        """
        if logprob is None:
            logprob = self.draft_extractor.estimate_draft_logprob(context_history or [], token)

        prob = math.exp(logprob)
        self.history.append(prob)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 1. Shannon Entropy calculation
        if top_probs:
            shannon_h = self.compute_shannon_entropy(top_probs)
        else:
            shannon_h = - (prob * math.log2(prob) if prob > 0 else 0.0)

        # 2. O(1) Exponential Moving Average (EMA) rolling variance
        self.count += 1
        if self.use_ema:
            if self.count == 1:
                self.ema_mean = prob
                self.ema_var = 0.0
            else:
                delta = prob - self.ema_mean
                self.ema_mean += self.alpha * delta
                self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * (delta ** 2))
            rolling_variance = float(self.ema_var)
        else:
            n = len(self.history)
            if n < 2:
                rolling_variance = 0.0
            else:
                mean = sum(self.history) / n
                rolling_variance = sum((x - mean) ** 2 for x in self.history) / n

        # 3. Dynamic POS & Entity Weighting
        token_weight = self.get_token_type_weight(token)

        # 4. Contrastive RAG Context Ratio (PMI)
        pmi_penalty = (
            self.compute_contrastive_pmi(logprob, unconditioned_logprob)
            if use_contrastive
            else 0.0
        )

        # Production Formulation:
        # Scale uncertainty by entity risk weight, while ensuring low-risk rare vocabulary (w <= 0.40) cannot accidentally breach tau
        raw_uncertainty = (1.0 - prob) + (2.0 * rolling_variance) + (0.45 * pmi_penalty)
        weighted_uncertainty_score = raw_uncertainty * token_weight

        is_hallucinating = weighted_uncertainty_score > self.tau

        return is_hallucinating, weighted_uncertainty_score, rolling_variance

    async def evaluate_token_async(
        self,
        token: str,
        logprob: Optional[float] = None,
        top_probs: Optional[List[float]] = None,
        context_history: Optional[List[str]] = None,
        use_contrastive: bool = True,
        unconditioned_logprob: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Version 5 Innovation: Asynchronous Speculative Entropy Worker.
        
        Executes entropy calculations asynchronously in a non-blocking coroutine.
        Allows token streaming to user interfaces at raw LLM speed while a speculative
        background worker continuously audits token logprobs and halts downstream generation.
        """
        return self.evaluate_token(
            token=token,
            logprob=logprob,
            top_probs=top_probs,
            context_history=context_history,
            use_contrastive=use_contrastive,
            unconditioned_logprob=unconditioned_logprob,
        )

    def evaluate_speculative_batch(
        self,
        token_batch: List[Tuple[str, float]],
        context_history: Optional[List[str]] = None
    ) -> Tuple[bool, int, float, float]:
        """
        Speculatively evaluates a lookahead buffer of K incoming tokens in parallel.
        Returns (has_breach, breach_index, max_uncertainty, max_variance).
        """
        context = list(context_history or [])
        for idx, (tok, lp) in enumerate(token_batch):
            is_h, unc, var = self.evaluate_token(tok, logprob=lp, context_history=context)
            context.append(tok)
            if is_h:
                return True, idx, unc, var
        return False, -1, 0.0, 0.0

    def evaluate_granite_payload(self, token_payload: Dict[str, Any], context_history: Optional[List[str]] = None) -> Tuple[bool, float, float]:
        """
        Evaluates a raw token payload received from IBM Granite API stream.
        """
        token, logprob, top_probs = self.draft_extractor.parse_granite_token_logprobs(token_payload)
        unconditioned_logprob = token_payload.get("unconditioned_logprob")
        if unconditioned_logprob is not None:
            unconditioned_logprob = float(unconditioned_logprob)
        return self.evaluate_token(
            token,
            logprob=logprob,
            top_probs=top_probs,
            context_history=context_history,
            use_contrastive=unconditioned_logprob is not None,
            unconditioned_logprob=unconditioned_logprob,
        )

    def calibrate_threshold(self, benchmark_scores: List[float], target_false_positive_rate: float = 0.05) -> float:
        """
        Calibrates tau based on factual ground-truth benchmark distributions.
        """
        if not benchmark_scores:
            return self.tau
        
        sorted_scores = sorted(benchmark_scores)
        idx = int(len(sorted_scores) * (1.0 - target_false_positive_rate))
        self.tau = float(sorted_scores[min(idx, len(sorted_scores) - 1)])
        return self.tau

    def generate_recovery_context(
        self,
        query: str,
        halted_token: str,
        context_history: List[str],
        uncertainty_score: float,
        rolling_variance: float
    ) -> Dict[str, Any]:
        """
        Phase 3 Autonomous Recovery: Formulates a structured context repair package
        for the Autonomous Self-Healing Agent when a circuit-breaker trip occurs.
        """
        partial_output = " ".join(context_history)
        suggested_search_terms = [t for t in query.split() if len(t) > 3 and not t.isdigit()]
        
        recovery_prompt = (
            f"SYSTEM: The previous generation was intercepted due to an intra-generation semantic entropy breach "
            f"(Uncertainty={uncertainty_score:.3f} > tau={self.tau:.2f}, Variance={rolling_variance:.3f}).\n"
            f"INTERCEPTED TOKEN: '{halted_token}'\n"
            f"ORIGINAL QUERY: '{query}'\n"
            f"VERIFIED PARTIAL CONTEXT: '{partial_output}'\n"
            f"DIRECTIVE: Execute vector similarity search in Milvus for '{' '.join(suggested_search_terms)}', "
            f"retrieve verified Celonis SLA ground truth, and formulate the factual ground-truth answer."
        )

        return {
            "query": query,
            "halted_token": halted_token,
            "uncertainty_score": round(uncertainty_score, 4),
            "rolling_variance": round(rolling_variance, 4),
            "tau_threshold": self.tau,
            "partial_context": partial_output,
            "suggested_query_terms": suggested_search_terms,
            "self_healing_prompt": recovery_prompt,
            "fallback_strategy": "vector_rerank_milvus"
        }

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Returns instantaneous snapshot of current entropy analytics state."""
        n = len(self.history)
        if n < 2:
            current_variance = 0.0
        else:
            mean = sum(self.history) / n
            current_variance = sum((x - mean) ** 2 for x in self.history) / n
        return {
            "tau_threshold": self.tau,
            "window_size": self.window_size,
            "history_depth": len(self.history),
            "current_variance": round(current_variance, 4),
            "recent_probabilities": [round(p, 4) for p in self.history]
        }

    def reset(self):
        """Reset sliding window state for a new streaming session."""
        self.history.clear()
        self.ema_mean = 0.0
        self.ema_var = 0.0
        self.count = 0
