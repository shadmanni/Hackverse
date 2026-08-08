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

    def evaluate_token(self, token: str, logprob: Optional[float] = None, top_probs: Optional[List[float]] = None, context_history: Optional[List[str]] = None) -> Tuple[bool, float, float]:
        """
        Evaluates an incoming token during decoding.
        
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

        # Combine metric score (higher score = higher uncertainty/hallucination risk)
        # Low probability or high variance inflates the uncertainty score
        uncertainty_score = (1.0 - prob) + (2.0 * rolling_variance)

        is_hallucinating = uncertainty_score > self.tau

        return is_hallucinating, uncertainty_score, rolling_variance

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

    def reset(self):
        """Reset sliding window state for a new streaming session."""
        self.history.clear()
