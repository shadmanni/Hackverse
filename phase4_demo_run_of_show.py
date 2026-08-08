"""
PHASE 4 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Sentinel-RAG Phase 4 Demo Run-of-Show & Pitch Benchmark Evaluator

This module provides the automated pitch demonstration run-of-show evaluator for
the final hackathon presentation.

Phase 4 Core Deliverables (Shadman):
  1. Demo Run-of-Show Automation: Execute complete Pitch Demo sequence.
  2. 100% Poison Prompt Interception Guarantee: Intercept out-of-scope forecasts,
     missing nodes (W-99), unauthorized discounts, and hallucinated entities (CC-9999).
  3. 100% Grounded Accuracy Verification: Retrieve authentic Celonis process logs
     with 0 false-positive circuit breaker trips.
  4. Dual Dataset Validation: Validate operational behavior on both standard
     demo logs (mock_celonis_data.json) and scaled enterprise logs (mock_celonis_data_large.json).
  5. Real-Time Telemetry & Report Generation: Export authoritative evaluation report
     to data/phase4_demo_run_of_show_report.json.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in python path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from phase3_rag_retriever import SentinelRAGRetriever, POISON_PROMPTS_PATH, MILVUS_DB_PATH
from entropy_engine import EntropyEngine

PHASE4_REPORT_PATH = BASE_DIR / "data" / "phase4_demo_run_of_show_report.json"

class Phase4DemoRunner:
    """
    Shadman's Phase 4 Pitch Demo & Run-of-Show Evaluator.
    """

    def __init__(self, db_path: str = MILVUS_DB_PATH, threshold_tau: float = 0.65):
        print(f"[Phase4DemoRunner] Initializing Sentinel-RAG Pipeline...")
        self.retriever = SentinelRAGRetriever(db_path=db_path)
        self.entropy_engine = EntropyEngine(threshold_tau=threshold_tau)
        self.db_path = db_path

    def run_demo_run_of_show(self, suite_path: Path = POISON_PROMPTS_PATH) -> Dict[str, Any]:
        """
        Executes the official Phase 4 Pitch Demo sequence across all test cases.
        """
        if not suite_path.exists():
            raise FileNotFoundError(f"Test suite file not found: {suite_path}")

        with open(suite_path, "r") as f:
            suite_data = json.load(f)

        print("\n" + "=" * 80)
        print(" [SENTINEL-RAG] PHASE 4 PITCH DEMO RUN-OF-SHOW EVALUATION")
        print("=" * 80)
        print(f"Dataset DB Path : {self.db_path}")
        print(f"Test Suite File : {suite_path}")
        print(f"Circuit Breaker : tau = {self.entropy_engine.tau}")
        print("=" * 80 + "\n")

        # -----------------------------------------------------------------
        # PART 1: Poison Prompt Interception (Hallucination Triggers)
        # -----------------------------------------------------------------
        print("[STAGE 1] Testing Poison Prompts (Firewall Interception Suite)...")
        poison_prompts = suite_data.get("poison_prompts", [])
        poison_results = []
        poison_intercepted = 0

        for i, item in enumerate(poison_prompts, start=1):
            t_start = time.perf_counter()
            retrieval_res = self.retriever.format_granite_context(item["prompt"])
            t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            is_intercepted = retrieval_res["is_poison"]
            if is_intercepted:
                poison_intercepted += 1

            status_symbol = "[INTERCEPTED]" if is_intercepted else "[FAILED: UNPROTECTED]"
            print(f"  [{i}/{len(poison_prompts)}] {item['id']} ({item['category']}): {status_symbol} in {t_elapsed_ms}ms")
            print(f"      Prompt: \"{item['prompt']}\"")
            print(f"      Reason: {retrieval_res['reason']}\n")

            poison_results.append({
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "expected_action": "CIRCUIT_BREAKER_TRIPPED",
                "actual_status": retrieval_res["status"],
                "is_intercepted": is_intercepted,
                "latency_ms": t_elapsed_ms,
                "interception_reason": retrieval_res["reason"]
            })

        # -----------------------------------------------------------------
        # PART 2: Grounded Queries (Verifiable Celonis EMS Metadata)
        # -----------------------------------------------------------------
        print("[STAGE 2] Testing Grounded Queries (Verifiable Celonis Process Logs)...")
        grounded_prompts = suite_data.get("grounded_prompts", [])
        grounded_results = []
        grounded_verified = 0

        for i, item in enumerate(grounded_prompts, start=1):
            t_start = time.perf_counter()
            retrieval_res = self.retriever.format_granite_context(item["prompt"])
            t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            is_verified = not retrieval_res["is_poison"]
            if is_verified:
                grounded_verified += 1

            chunks = retrieval_res.get("chunks", [])
            top_score = chunks[0]["similarity_score"] if chunks else 0.0
            top_chunk_text = chunks[0]["text"] if chunks else "N/A"

            status_symbol = "[VERIFIED]" if is_verified else "[FALSE POSITIVE INTERCEPTION]"
            print(f"  [{i}/{len(grounded_prompts)}] {item['id']} ({item['category']}): {status_symbol} in {t_elapsed_ms}ms")
            print(f"      Prompt: \"{item['prompt']}\"")
            print(f"      Top Vector Score: {top_score} | Context: {top_chunk_text[:90]}...\n")

            grounded_results.append({
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "expected_action": "STREAM_CLEANLY",
                "actual_status": retrieval_res["status"],
                "is_verified": is_verified,
                "latency_ms": t_elapsed_ms,
                "top_similarity_score": top_score,
                "retrieved_fact_snippet": top_chunk_text[:120]
            })

        # -----------------------------------------------------------------
        # PART 3: Performance Metrics & Pitch Summary Computation
        # -----------------------------------------------------------------
        total_poison = len(poison_prompts)
        total_grounded = len(grounded_prompts)
        poison_rate = (poison_intercepted / total_poison * 100) if total_poison > 0 else 0.0
        grounded_rate = (grounded_verified / total_grounded * 100) if total_grounded > 0 else 0.0

        all_latencies = [r["latency_ms"] for r in poison_results + grounded_results]
        avg_latency_ms = round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0

        is_pitch_ready = (poison_rate == 100.0) and (grounded_rate == 100.0)
        readiness_status = "READY FOR PITCH (100% ACCURACY)" if is_pitch_ready else "CALIBRATION REQUIRED"

        report = {
            "title": "Sentinel-RAG Phase 4 Pitch Demo & Run-of-Show Evaluation Report",
            "phase": "PHASE 4 - Hardening, Enterprise Polish & Final Pitch Prep",
            "lead": "Shadman Nishat (Data Pipeline & Ingestion Lead)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_poison_prompts": total_poison,
                "poison_prompts_intercepted": poison_intercepted,
                "poison_interception_rate_percent": round(poison_rate, 2),
                "total_grounded_prompts": total_grounded,
                "grounded_prompts_verified": grounded_verified,
                "grounded_accuracy_rate_percent": round(grounded_rate, 2),
                "average_pipeline_latency_ms": avg_latency_ms,
                "circuit_breaker_tau": self.entropy_engine.tau,
                "pitch_readiness_status": readiness_status
            },
            "poison_interception_details": poison_results,
            "grounded_retrieval_details": grounded_results
        }

        with open(PHASE4_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print("=" * 80)
        print("PHASE 4 DEMO RUN-OF-SHOW SUMMARY REPORT")
        print("=" * 80)
        print(f"  Poison Prompts Intercepted : {poison_intercepted} / {total_poison} ({poison_rate:.1f}%)")
        print(f"  Grounded Queries Verified  : {grounded_verified} / {total_grounded} ({grounded_rate:.1f}%)")
        print(f"  Average Interception Time  : {avg_latency_ms} ms")
        print(f"  Firewall Pitch Status      : {readiness_status}")
        print(f"  Report Exported To         : {PHASE4_REPORT_PATH}")
        print("=" * 80 + "\n")

        return report

def main():
    parser = argparse.ArgumentParser(description="Sentinel-RAG Phase 4 Pitch Demo Run-of-Show Evaluator")
    parser.add_argument("--db", type=str, default=MILVUS_DB_PATH, help="Path to Milvus DB file")
    parser.add_argument("--suite", type=str, default=str(POISON_PROMPTS_PATH), help="Path to poison_prompts.json")
    args = parser.parse_args()

    runner = Phase4DemoRunner(db_path=args.db)
    runner.run_demo_run_of_show(suite_path=Path(args.suite))

if __name__ == "__main__":
    main()
