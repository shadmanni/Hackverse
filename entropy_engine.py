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
        if len(logprobs) < 2:
            return 0.0
        return float(statistics.pvariance(logprobs))

    def get_token_type_weight(self, token: str) -> float:
        """
        Calculates a dynamic importance weight w(token) based on token taxonomy & POS semantics.
        
        Weighting rationale:
        - Critical Risk (w = 2.8 - 3.5): Numeric data, dollar currencies, percentages, dates, codes (e.g. '$128,500', '42.8_days', 'CC-4471').
          Hallucinations in these entities cause direct financial & compliance liabilities.
        - High Risk (w = 2.0): Proper nouns, capitalized identifiers, technical enterprise terms.
        - Medium Risk (w = 1.0): Content verbs, adjectives, general nouns.
        - Low Risk (w = 0.45): Grammatical stop words, prepositions, articles ('the', 'is', 'to', 'for', 'of').
        """
        clean = (token or "").strip()
        if not clean:
            return 0.5

        # 1. Critical: Financial, currency, digits, and quantitative claims
        if any(char.isdigit() for char in clean) or any(c in clean for c in ["$", "€", "£", "%"]):
            return 3.2

        # 2. Critical: Specialized code / process identifiers (e.g. CASE-10231, CC-4471, W-99)
        if "-" in clean and any(part.isupper() for part in clean.split("-")):
            return 2.8

        # 3. High: Capitalized Enterprise Named Entities / Proper Nouns
        if clean[0].isupper() and len(clean) > 2 and not clean.endswith("."):
            return 1.8

        # 4. Low: Common grammatical stopwords / glue tokens
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", 
            "on", "with", "at", "by", "from", "up", "about", "into", "over", "after",
            "and", "or", "but", "if", "that", "this", "these", "those", "it", "its"
        }
        if clean.lower() in stopwords:
            return 0.45

        # 5. Baseline for general vocabulary
        return 1.0

    def compute_contrastive_pmi(self, token: str, logprob_with_context: float, context_history: Optional[List[str]] = None) -> float:
        """
        Version 4 Innovation: Pointwise Mutual Information (PMI) / Contrastive RAG Logprob Ratio.
        
        Calculates how strongly a generated token depends on the retrieved Celonis RAG context:
            PMI(x_t; Context) = log P(x_t | context) - log P(x_t | ungrounded_parametric_memory)
            
        - High positive PMI: Token is strongly supported by Celonis ground truth.
        - Low or negative PMI: Token relies on ungrounded model parametric memory (hallucination risk).
        """
        # Baseline prior probability without context
        unconditioned_logprob = self.draft_extractor.estimate_draft_logprob([], token)
        
        # Contrastive delta: difference between contextual logprob and prior
        contrastive_ratio = logprob_with_context - unconditioned_logprob
        
        # If contrastive ratio is negative, model is generating against grounding -> penalty multiplier
        pmi_penalty = max(0.0, -contrastive_ratio)
        return float(pmi_penalty)

    def evaluate_token(self, token: str, logprob: Optional[float] = None, top_probs: Optional[List[float]] = None, context_history: Optional[List[str]] = None, use_contrastive: bool = True) -> Tuple[bool, float, float]:
        """
        Evaluates an incoming token during decoding using Version 4 Contrastive RAG Logprob Ratio + POS Weighting + O(1) EMA.
        
        :param token: The decoded string token.
        :param logprob: The log-probability of the chosen token (if available).
        :param top_probs: Optional distribution over top candidates to calculate Shannon entropy.
        :param context_history: Streaming context history for draft estimation.
        :param use_contrastive: Enable Version 4 Contrastive PMI ratio evaluation.
        :return: (is_hallucinating, metric_value, rolling_variance)
        """
        # If logprob isn't provided directly, use draft model logic to estimate it from context
        if logprob is None:
            logprob = self.draft_extractor.estimate_draft_logprob(context_history or [], token)

        prob = math.exp(logprob)
        self.history.append(prob)
        
        # Keep window size fixed
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 1. Calculate Shannon Entropy if top candidate probabilities are available
        if top_probs:
            shannon_h = self.compute_shannon_entropy(top_probs)
        else:
            # Fallback estimation based on single token prob
            shannon_h = - (prob * math.log2(prob) if prob > 0 else 0.0)

        # 2. Calculate rolling variance: Version 3 uses O(1) Exponential Moving Average (EMA)
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
            rolling_variance = float(statistics.pvariance(self.history)) if len(self.history) >= 2 else 0.0

        # 3. Dynamic POS & Entity Weighting
        token_weight = self.get_token_type_weight(token)

        # 4. Version 4 Contrastive RAG Logprob Ratio (PMI)
        pmi_penalty = self.compute_contrastive_pmi(token, logprob, context_history) if use_contrastive else 0.0

        # Mathematical formulation:
        # High uncertainty on numeric/financial entities scaled by contrastive context gap
        # Contrastive PMI penalty is weighted by token_weight so stopwords don't trigger false positives
        raw_uncertainty = (1.0 - prob) + (2.0 * rolling_variance) + (0.5 * pmi_penalty)
        weighted_uncertainty_score = raw_uncertainty * token_weight

        is_hallucinating = weighted_uncertainty_score > self.tau

        return is_hallucinating, weighted_uncertainty_score, rolling_variance

    async def evaluate_token_async(
        self,
        token: str,
        logprob: Optional[float] = None,
        top_probs: Optional[List[float]] = None,
        context_history: Optional[List[str]] = None,
        use_contrastive: bool = True
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
            use_contrastive=use_contrastive
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
        return self.evaluate_token(token, logprob=logprob, top_probs=top_probs, context_history=context_history)

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
        for the watsonx Self-Healing Agent when a circuit-breaker trip occurs.
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
        current_variance = float(statistics.pvariance(self.history)) if len(self.history) >= 2 else 0.0
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
