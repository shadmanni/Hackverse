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

    def __init__(self, threshold_tau: float = 0.65, window_size: int = 5):
        """
        :param threshold_tau: The entropy/variance threshold above which a hallucination is flagged.
        :param window_size: Number of previous tokens to consider for rolling variance.
        """
        self.tau = threshold_tau
        self.window_size = window_size
        self.history: List[float] = []
        self.draft_extractor = DraftLogprobExtractor()

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

    def evaluate_token(self, token: str, logprob: Optional[float] = None, top_probs: Optional[List[float]] = None, context_history: Optional[List[str]] = None) -> Tuple[bool, float, float]:
        """
        Evaluates an incoming token during decoding using Token-Type & POS-Aware Entropy Weighting.
        
        :param token: The decoded string token.
        :param logprob: The log-probability of the chosen token (if available).
        :param top_probs: Optional distribution over top candidates to calculate Shannon entropy.
        :param context_history: Streaming context history for draft estimation.
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

        # 2. Calculate rolling variance over the sliding window
        rolling_variance = float(statistics.pvariance(self.history)) if len(self.history) >= 2 else 0.0

        # 3. Dynamic POS & Entity Weighting
        token_weight = self.get_token_type_weight(token)

        # Mathematical formulation:
        # High uncertainty on numeric/financial entities (high weight) scales the score exponentially,
        # while grammatical variations on stopwords (low weight) avoid triggering false positives.
        raw_uncertainty = (1.0 - prob) + (2.0 * rolling_variance)
        weighted_uncertainty_score = raw_uncertainty * token_weight

        is_hallucinating = weighted_uncertainty_score > self.tau

        return is_hallucinating, weighted_uncertainty_score, rolling_variance

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
