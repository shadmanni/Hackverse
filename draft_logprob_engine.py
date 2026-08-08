import math
import statistics
from typing import Dict, List, Optional, Tuple, Any

class DraftLogprobExtractor:
    """
    Riddhi's Draft Model & Log-Probability Extraction Engine.
    
    Extracts, normalizes, and estimates real-time token log-probabilities 
    p(x_t | x_<t) from IBM Granite API context windows or parallel draft models.
    """

    def __init__(self, default_top_k: int = 5):
        self.default_top_k = default_top_k

    def parse_granite_token_logprobs(self, token_data: Dict[str, Any]) -> Tuple[str, float, List[float]]:
        """
        Parses raw token data from Granite API response payload.
        Expected format:
        {
            "token": "word",
            "logprob": -0.15,
            "top_logprobs": {"word": -0.15, "term": -2.3, "item": -3.8}
        }
        """
        token = token_data.get("token", "")
        logprob = float(token_data.get("logprob", 0.0))
        
        top_logprobs_dict = token_data.get("top_logprobs", {})
        if top_logprobs_dict:
            # Convert logprobs to probabilities for Shannon Entropy calculation
            probs = [math.exp(lp) for lp in top_logprobs_dict.values()]
            # Normalize probabilities to sum to 1.0
            sum_p = sum(probs)
            if sum_p > 0:
                probs = [p / sum_p for p in probs]
        else:
            probs = [math.exp(logprob)]

        return token, logprob, probs

    def estimate_draft_logprob(self, context_tokens: List[str], target_token: str) -> float:
        """
        Parallel draft model estimation for fallback when raw logprobs are missing from stream.
        Estimates log-probability based on language structure / context length heuristic.
        """
        if not target_token:
            return 0.0
        
        # Heuristic draft model: numerical digits / ungrounded terms after complex context have lower logprobs
        if any(char.isdigit() for char in target_token) and len(context_tokens) >= 5:
            # Lower confidence on numeric claims without explicit RAG backing
            return -2.45
        elif target_token.startswith("$") or "cost" in target_token.lower():
            return -2.10
        else:
            # High confidence baseline for common structural tokens
            return -0.12

    def calculate_context_entropy(self, logprobs: List[float]) -> float:
        """
        Calculates time-series log-probability variance V(y_t) over context window tokens.
        """
        if len(logprobs) < 2:
            return 0.0
        return float(statistics.pvariance(logprobs))
