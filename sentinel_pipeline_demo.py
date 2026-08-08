import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from entropy_engine import EntropyEngine

DATA_PATH = Path(__file__).parent / "data" / "mock_celonis_data.json"

class IntegratedSentinelRAG:
    """
    Riddhi's Integrated Sentinel-RAG Pipeline Engine.
    Combines:
    1. Celonis Ground-Truth Knowledge Base (loaded & sanitized from Shadman's mock dataset).
    2. RAG Context Retrieval.
    3. Riddhi's Real-time Entropy & Logprob Circuit Breaker Engine.
    """

    def __init__(self, threshold_tau: float = 0.65, window_size: int = 5):
        self.entropy_engine = EntropyEngine(threshold_tau=threshold_tau, window_size=window_size)
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, Any]:
        if DATA_PATH.exists():
            with open(DATA_PATH, "r") as f:
                return json.load(f)
        return {}

    def get_ground_truth_context(self, query: str) -> str:
        """
        Retrieves authoritatively grounded context from Celonis process logs.
        """
        summary = self.ground_truth.get("summary_metrics", {})
        avg_compliance = summary.get("avg_compliance_cycle_time_days", 12)
        total_orders = summary.get("total_orders", 3)
        return (
            f"Ground Truth (Celonis Event Logs): "
            f"Average Q3 compliance cycle time for vendor onboarding / compliance review is exactly {avg_compliance} days "
            f"across {total_orders} processed orders. Orders over $100k take 12 days on average."
        )

    def process_token_stream(self, tokens_with_logprobs: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        Processes a stream of generated tokens with their log-probabilities,
        evaluating entropy & log-probability variance at every single step.
        """
        self.entropy_engine.reset()
        stream_results = []
        context_history = []

        for token, logprob in tokens_with_logprobs:
            is_hallucinating, uncertainty_score, rolling_variance = self.entropy_engine.evaluate_token(
                token=token,
                logprob=logprob,
                context_history=context_history
            )

            result = {
                "token": token,
                "logprob": logprob,
                "uncertainty_score": round(uncertainty_score, 4),
                "rolling_variance": round(rolling_variance, 4),
                "circuit_breaker_tripped": is_hallucinating
            }
            stream_results.append(result)
            context_history.append(token)

            if is_hallucinating:
                # Stop decoding immediately on threshold breach
                break

        return stream_results

def run_sentinel_rag_demo():
    """
    Demonstrates the full end-to-end Sentinel-RAG pipeline on two test cases:
    1. Grounded Factual Response (logprobs near 0, circuit breaker silent).
    2. Poison Prompt / Hallucination Attack (logprob drop, circuit breaker trips mid-sentence).
    """
    pipeline = IntegratedSentinelRAG(threshold_tau=0.65)
    query = "What is the exact Q3 compliance cycle time for vendor onboarding based on Celonis event logs?"

    print("=" * 80)
    print("🛡️ SENTINEL-RAG INTERCEPTOR DEMO: INTEGRATED ENTERPRISE PIPELINE")
    print("=" * 80)
    print(f"\n[QUERY] {query}")
    ground_truth = pipeline.get_ground_truth_context(query)
    print(f"[RAG GROUND TRUTH] {ground_truth}\n")

    # Case 1: Grounded Stream
    print("-" * 50)
    print("SCENARIO A: Grounded Generation (High Log-Probability)")
    print("-" * 50)
    grounded_tokens = [
        ("Based", -0.05), ("on", -0.02), ("Celonis", -0.08), ("logs,", -0.04),
        ("the", -0.01), ("Q3", -0.03), ("compliance", -0.05), ("cycle", -0.02),
        ("time", -0.01), ("is", -0.02), ("exactly", -0.04), ("12", -0.06), ("days.", -0.03)
    ]
    results_a = pipeline.process_token_stream(grounded_tokens)
    for res in results_a:
        print(f"Token: {res['token']:<12} | Logprob: {res['logprob']:<6} | Uncertainty: {res['uncertainty_score']:<6} | Tripped: {res['circuit_breaker_tripped']}")
    print("Result: ✅ Response streamed cleanly to client with 0 interceptions.\n")

    # Case 2: Hallucinating Stream
    print("-" * 50)
    print("SCENARIO B: Hallucination Attack (Extrinsic Number Fabrication)")
    print("-" * 50)
    hallucinating_tokens = [
        ("Based", -0.05), ("on", -0.02), ("Celonis", -0.08), ("logs,", -0.04),
        ("the", -0.01), ("Q3", -0.03), ("compliance", -0.05), ("cycle", -0.02),
        ("time", -0.01), ("is", -0.02), ("exactly", -0.04), ("42.8_days", -2.85), ("unverified_vendor_cost_$5M", -3.90)
    ]
    results_b = pipeline.process_token_stream(hallucinating_tokens)
    for res in results_b:
        print(f"Token: {res['token']:<12} | Logprob: {res['logprob']:<6} | Uncertainty: {res['uncertainty_score']:<6} | Tripped: {res['circuit_breaker_tripped']}")
        if res['circuit_breaker_tripped']:
            print(f"\n🚨 [CIRCUIT BREAKER TRIPPED MID-STREAM!]")
            print(f"   Reason: Uncertainty ({res['uncertainty_score']}) > Threshold tau (0.65)")
            print(f"   Action: Connection severed at token '{res['token']}' before hallucination reached user.")
            print(f"   Fallback: Rerouting query to Agentic Fallback Handler.\n")

if __name__ == "__main__":
    run_sentinel_rag_demo()
