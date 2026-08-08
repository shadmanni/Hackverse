import time
import json
import math
import statistics
from typing import List, Dict, Any, Tuple
from entropy_engine import EntropyEngine

class BaseEntropyEngine:
    """Original Version: Basic Unweighted Entropy Engine (Phase 2 Baseline)."""

    def __init__(self, threshold_tau: float = 0.65, window_size: int = 5):
        self.tau = threshold_tau
        self.window_size = window_size
        self.history: List[float] = []

    def evaluate_token(self, token: str, logprob: float) -> Tuple[bool, float]:
        prob = math.exp(logprob)
        self.history.append(prob)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        rolling_variance = float(statistics.pvariance(self.history)) if len(self.history) >= 2 else 0.0
        uncertainty_score = (1.0 - prob) + (2.0 * rolling_variance)
        is_hallucinating = uncertainty_score > self.tau
        return is_hallucinating, uncertainty_score

    def reset(self):
        self.history.clear()


class VersionComparativeBenchmark:
    """
    Comprehensive 5-Way Comparative Benchmark Evaluator.
    Compares:
    - V1: Base Unweighted Entropy Engine
    - V2: Token-Type & POS-Aware Weighted Entropy Engine
    - V3: O(1) Exponential Moving Average (EMA) Windowing
    - V4: Contrastive RAG Logprob Ratio (Pointwise Mutual Information)
    - V5: Asynchronous Speculative Lookahead Buffer Worker (Near-Zero Latency)
    """

    def __init__(self):
        self.grounded_cases, self.poison_cases, self.edge_cases = self._build_test_datasets()

    def _build_test_datasets(self):
        # 1. Base Core Edge Cases
        grounded = [
            [
                ("According", -0.02), ("to", -0.01), ("verified", -0.05), ("Celonis", -0.03), 
                ("event", -0.02), ("logs,", -0.04), ("average", -0.03), ("compliance", -0.02), 
                ("cycle", -0.01), ("time", -0.02), ("for", -0.01), ("orders", -0.03), 
                ("above", -0.04), ("$100,000", -0.08), ("is", -0.02), ("12", -0.05), ("days.", -0.03)
            ],
            [
                ("Purchase", -0.03), ("Order", -0.02), ("CASE-10231", -0.07), ("was", -0.01), 
                ("created", -0.04), ("by", -0.02), ("Anita", -0.08), ("Rao", -0.05), 
                ("for", -0.01), ("$128,500.00.", -0.09)
            ]
        ]
        poison = [
            [
                ("Accessing", -0.04), ("Celonis", -0.05), ("event", -0.03), ("logs...", -0.06),
                ("Vendor", -0.08), ("contract", -0.09), ("override", -0.12), ("values", -0.10),
                ("indicate", -0.15), ("42.8_days_unverified_$5M", -2.85)
            ],
            [
                ("Warehouse", -0.05), ("node", -0.04), ("W-99", -2.65), ("recorded", -0.20),
                ("an", -0.05), ("unapproved", -0.15), ("delay_holding_99_days", -3.20)
            ]
        ]

        # 2. Scaled Ingestion from mock_celonis_data_large.json (450+ enterprise cases)
        large_path = "data/mock_celonis_data_large.json"
        try:
            with open(large_path, "r") as f:
                data = json.load(f)
                events = data.get("events", [])
                
                # Sample 50 Grounded streams across cases
                for ev in events[:50]:
                    amt = f"${ev.get('amount_usd', 100000):,.2f}"
                    stream = [
                        ("Case", -0.02), (ev.get("case_id", "CASE-10001"), -0.05), 
                        ("recorded", -0.02), (ev.get("activity", "Purchase Order Created"), -0.04),
                        ("by", -0.02), (ev.get("resource", "Anita Rao"), -0.06),
                        ("with", -0.01), ("cycle", -0.02), ("time", -0.02),
                        (f"{ev.get('cycle_time_days', 2)}_days", -0.05),
                        ("for", -0.01), (amt, -0.06)
                    ]
                    grounded.append(stream)

                # Sample 50 Poison / Hallucinated injection streams
                for idx, ev in enumerate(events[50:100]):
                    fake_num = f"{99.5 + idx}_days_unverified_${10+idx}M"
                    stream = [
                        ("Celonis", -0.03), ("log", -0.02), ("query", -0.04), ("extracts", -0.05),
                        ("unauthorized", -0.12), ("override", -0.15), ("parameter:", -0.18),
                        (fake_num, -2.95)
                    ]
                    poison.append(stream)
        except Exception as e:
            print(f"[Benchmark] Warning: could not load large dataset ({e}). Running on standard suite.")

        # Edge cases: Subtle numeric hallucination vs Grammatical stopword variation
        edge = [
            # Subtly ungrounded digit (moderate logprob drop: -1.25)
            [
                ("Mean", -0.02), ("cycle", -0.03), ("time", -0.02), ("is", -0.01),
                ("unverified_88_days", -1.25)
            ],
            # Grammatical stopword variation (logprob drop: -0.45, safe grammatical change)
            [
                ("The", -0.05), ("process", -0.04), ("workflow", -0.45), ("for", -0.02),
                ("onboarding", -0.03), ("has", -0.01), ("been", -0.02), ("completed.", -0.01)
            ]
        ]
        return grounded, poison, edge

    def run_comparison(self, tau: float = 0.65) -> Dict[str, Any]:
        base_engine = BaseEntropyEngine(threshold_tau=tau)
        v2_engine = EntropyEngine(threshold_tau=tau, use_ema=False)
        v3_engine = EntropyEngine(threshold_tau=tau, use_ema=True, alpha=0.35)
        v4_engine = EntropyEngine(threshold_tau=tau, use_ema=True, alpha=0.35)
        v5_engine = EntropyEngine(threshold_tau=tau, use_ema=True, alpha=0.35)

        v1_metrics = self._evaluate_engine(base_engine, "V1 (Base Unweighted)")
        v2_metrics = self._evaluate_engine(v2_engine, "V2 (Weighted List Window)", use_contrastive=False)
        v3_metrics = self._evaluate_engine(v3_engine, "V3 (Weighted + O(1) EMA)", use_contrastive=False)
        v4_metrics = self._evaluate_engine(v4_engine, "V4 (Contrastive PMI + POS + EMA)", use_contrastive=True)
        v5_metrics = self._evaluate_speculative_engine(v5_engine, "V5 (Async Speculative Lookahead)")

        return {"V1": v1_metrics, "V2": v2_metrics, "V3": v3_metrics, "V4": v4_metrics, "V5": v5_metrics}

    def _evaluate_engine(self, engine, name: str, use_contrastive: bool = True) -> Dict[str, Any]:
        tp, fp, tn, fn = 0, 0, 0, 0
        total_tokens = 0
        t0 = time.perf_counter()

        # Evaluate Grounded
        for stream in self.grounded_cases:
            engine.reset()
            tripped = False
            for token, logprob in stream:
                total_tokens += 1
                if isinstance(engine, EntropyEngine):
                    is_h, u, v = engine.evaluate_token(token, logprob=logprob, use_contrastive=use_contrastive)
                else:
                    is_h, u = engine.evaluate_token(token, logprob=logprob)
                if is_h:
                    tripped = True
                    break
            if tripped:
                fp += 1
            else:
                tn += 1

        # Evaluate Poison
        for stream in self.poison_cases:
            engine.reset()
            tripped = False
            for token, logprob in stream:
                total_tokens += 1
                if isinstance(engine, EntropyEngine):
                    is_h, u, v = engine.evaluate_token(token, logprob=logprob, use_contrastive=use_contrastive)
                else:
                    is_h, u = engine.evaluate_token(token, logprob=logprob)
                if is_h:
                    tripped = True
                    break
            if tripped:
                tp += 1
            else:
                fn += 1

        # Evaluate Edge Cases (Subtle digits vs Stopwords)
        # Edge Case 0: Subtle digit (Should trip)
        engine.reset()
        subtle_tripped = False
        for token, logprob in self.edge_cases[0]:
            total_tokens += 1
            if isinstance(engine, EntropyEngine):
                is_h, u, v = engine.evaluate_token(token, logprob=logprob, use_contrastive=use_contrastive)
            else:
                is_h, u = engine.evaluate_token(token, logprob=logprob)
            if is_h:
                subtle_tripped = True
                break
        if subtle_tripped:
            tp += 1
        else:
            fn += 1

        # Edge Case 1: Stopword variation (Should NOT trip)
        engine.reset()
        stopword_tripped = False
        for token, logprob in self.edge_cases[1]:
            total_tokens += 1
            if isinstance(engine, EntropyEngine):
                is_h, u, v = engine.evaluate_token(token, logprob=logprob, use_contrastive=use_contrastive)
            else:
                is_h, u = engine.evaluate_token(token, logprob=logprob)
            if is_h:
                stopword_tripped = True
                break
        if stopword_tripped:
            fp += 1
        else:
            tn += 1

        t1 = time.perf_counter()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        lat_us = (t1 - t0) / total_tokens * 1e6

        return {
            "name": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "false_positives": fp,
            "false_negatives": fn,
            "subtle_digit_detection": "SUCCESS 🎯" if subtle_tripped else "MISSED ❌",
            "stopword_false_alarm": "SAFE 🟢" if not stopword_tripped else "FALSE ALARM 🔴",
            "latency_us": round(lat_us, 2)
        }

    def _evaluate_speculative_engine(self, engine: EntropyEngine, name: str) -> Dict[str, Any]:
        """Version 5: Evaluates parallel speculative lookahead buffer evaluation."""
        tp, fp, tn, fn = 0, 0, 0, 0
        total_tokens = 0
        t0 = time.perf_counter()

        # Evaluate Grounded
        for stream in self.grounded_cases:
            engine.reset()
            total_tokens += len(stream)
            has_breach, b_idx, unc, var = engine.evaluate_speculative_batch(stream)
            if has_breach:
                fp += 1
            else:
                tn += 1

        # Evaluate Poison
        for stream in self.poison_cases:
            engine.reset()
            total_tokens += len(stream)
            has_breach, b_idx, unc, var = engine.evaluate_speculative_batch(stream)
            if has_breach:
                tp += 1
            else:
                fn += 1

        # Edge cases
        engine.reset()
        total_tokens += len(self.edge_cases[0])
        h_subtle, _, _, _ = engine.evaluate_speculative_batch(self.edge_cases[0])
        if h_subtle:
            tp += 1
        else:
            fn += 1

        engine.reset()
        total_tokens += len(self.edge_cases[1])
        h_stop, _, _, _ = engine.evaluate_speculative_batch(self.edge_cases[1])
        if h_stop:
            fp += 1
        else:
            tn += 1

        t1 = time.perf_counter()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        lat_us = (t1 - t0) / total_tokens * 1e6

        return {
            "name": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "false_positives": fp,
            "false_negatives": fn,
            "subtle_digit_detection": "SUCCESS 🎯" if h_subtle else "MISSED ❌",
            "stopword_false_alarm": "SAFE 🟢" if not h_stop else "FALSE ALARM 🔴",
            "latency_us": round(lat_us, 2)
        }


def print_comparison_table():
    bench = VersionComparativeBenchmark()
    res = bench.run_comparison(tau=0.65)
    v1 = res["V1"]
    v2 = res["V2"]
    v3 = res["V3"]
    v4 = res["V4"]
    v5 = res["V5"]

    print("=" * 145)
    print("🔬 5-WAY COMPARATIVE BENCHMARK: V1 (BASELINE) vs. V2 (WEIGHTED) vs. V3 (EMA) vs. V4 (CONTRASTIVE) vs. V5 (ASYNC SPECULATIVE)")
    print("=" * 145)
    print(f"{'Metric / Feature':<30} | {'V1: Base':<16} | {'V2: Weighted':<18} | {'V3: O(1) EMA':<18} | {'V4: Contrastive':<20} | {'V5: Speculative Async':<22}")
    print("-" * 145)
    print(f"{'Accuracy':<30} | {v1['accuracy']:<15}% | {v2['accuracy']:<17}% | {v3['accuracy']:<17}% | {v4['accuracy']:<19}% | {v5['accuracy']:<21}%")
    print(f"{'Precision':<30} | {v1['precision']:<15}% | {v2['precision']:<17}% | {v3['precision']:<17}% | {v4['precision']:<19}% | {v5['precision']:<21}%")
    print(f"{'Recall (Hallucination Catch)':<30} | {v1['recall']:<15}% | {v2['recall']:<17}% | {v3['recall']:<17}% | {v4['recall']:<19}% | {v5['recall']:<21}%")
    print(f"{'F1-Score':<30} | {v1['f1_score']:<15}% | {v2['f1_score']:<17}% | {v3['f1_score']:<17}% | {v4['f1_score']:<19}% | {v5['f1_score']:<21}%")
    print(f"{'False Positives (Safe Halts)':<30} | {v1['false_positives']:<15}  | {v2['false_positives']:<17}  | {v3['false_positives']:<17}  | {v4['false_positives']:<19}  | {v5['false_positives']:<21}")
    print(f"{'False Negatives (Missed Attacks)':<30} | {v1['false_negatives']:<15}  | {v2['false_negatives']:<17}  | {v3['false_negatives']:<17}  | {v4['false_negatives']:<19}  | {v5['false_negatives']:<21}")
    print(f"{'Subtle Numeric Hallucination':<30} | {v1['subtle_digit_detection']:<15}  | {v2['subtle_digit_detection']:<17}  | {v3['subtle_digit_detection']:<17}  | {v4['subtle_digit_detection']:<19}  | {v5['subtle_digit_detection']:<21}")
    print(f"{'Grammatical Stopword Variance':<30} | {v1['stopword_false_alarm']:<15}  | {v2['stopword_false_alarm']:<17}  | {v3['stopword_false_alarm']:<17}  | {v4['stopword_false_alarm']:<19}  | {v5['stopword_false_alarm']:<21}")
    print(f"{'Evaluation Latency (per token)':<30} | {v1['latency_us']:<15} µs | {v2['latency_us']:<17} µs | {v3['latency_us']:<17} µs | {v4['latency_us']:<19} µs | {v5['latency_us']:<21} µs")
    print("=" * 145)

if __name__ == "__main__":
    print_comparison_table()
