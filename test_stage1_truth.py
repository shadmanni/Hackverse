"""
Stage 1 regression tests: the numbers the system states must come from the data,
and the retriever must separate grounded queries from ungrounded ones.

Run: python -m pytest test_stage1_truth.py -v
"""

import json
import re
import unittest
from pathlib import Path

import celonis_metrics as cm

BASE = Path(__file__).parent.resolve()


class TestGroundTruthMetrics(unittest.TestCase):
    """The demo used to assert 4.2 days / 99.4% SLA. Neither is in the event log."""

    def setUp(self):
        self.p = cm.process_profile()

    def test_profile_matches_raw_event_log(self):
        events = json.loads((BASE / "data" / "mock_celonis_data_large.json").read_text())["events"]
        self.assertEqual(self.p["total_events"], len(events))
        self.assertEqual(self.p["total_cases"], len({e["case_id"] for e in events}))
        cycles = [e["cycle_time_days"] for e in events]
        self.assertAlmostEqual(self.p["mean_cycle_days"], sum(cycles) / len(cycles), places=1)

    def test_fabricated_demo_numbers_are_not_groundable(self):
        # These are the exact figures the old hardcoded PROCESS_GRAPHS asserted.
        for fake in (4.2, 42.8, 99.4, 3.1, 6.4, 1.8):
            self.assertFalse(
                cm.is_grounded_number(fake),
                f"{fake} is not derivable from the event log but was reported as verified truth",
            )

    def test_real_metrics_are_groundable(self):
        for real in (
            self.p["mean_cycle_days"],
            self.p["declared_avg_compliance_cycle_time_days"],
            self.p["total_cases"],
            self.p["by_activity"]["Compliance Review"]["mean_cycle_days"],
        ):
            self.assertTrue(cm.is_grounded_number(real), f"{real} came from the log but failed grounding")

    def test_rounding_tolerated_but_fabrication_is_not(self):
        mean = self.p["mean_cycle_days"]          # 8.5
        self.assertTrue(cm.is_grounded_number(mean + 0.01))   # rounding, still grounded
        # Known limitation: every raw cycle_time_days value (1..23) is groundable,
        # so small integers cannot be refuted by this layer alone. The check has
        # real power on aggregates, percentages and dollar amounts, which is where
        # fabrication actually shows up. Small ints are the entropy layer's job.
        self.assertFalse(cm.is_grounded_number(87.6))         # fabricated aggregate
        self.assertFalse(cm.is_grounded_number(42.8))         # the old demo's fake figure
        self.assertFalse(cm.is_grounded_number(5_000_000.0))  # fabricated dollar amount

    def test_ground_truth_answer_only_states_derivable_figures(self):
        for q in [
            "average compliance cycle time for high-value orders above $100,000",
            "which node caused a bottleneck delay",
            "compliance review cycle time",
        ]:
            ans = cm.ground_truth_answer(q)
            nums = [float(t) for t in re.findall(r"\d+\.?\d*", ans.replace(",", ""))]
            for n in nums:
                self.assertTrue(
                    cm.is_grounded_number(n),
                    f"recovery answer for {q!r} stated {n}, which is not in the event log:\n{ans}",
                )


class TestRetrieverScoring(unittest.TestCase):
    """CrossEncoder logits were fused raw with a 0..1 cosine, so the 0.35 threshold
    compared against an unbounded scale and intercepted grounded queries."""

    @classmethod
    def setUpClass(cls):
        from phase3_rag_retriever import SentinelRAGRetriever
        cls.r = SentinelRAGRetriever()

    def test_fused_scores_are_bounded_probabilities(self):
        for q in ["invoice approval cycle time", "banana helicopter zebra", "CASE-10231 amount"]:
            for h in self.r.retrieve(q, top_k=3):
                self.assertGreaterEqual(h["similarity_score"], 0.0, f"{q}: score below 0")
                self.assertLessEqual(h["similarity_score"], 1.0, f"{q}: score above 1")

    def test_grounded_queries_are_not_intercepted(self):
        for q in [
            "What is the cycle time for invoice approval",
            "What is the average compliance cycle time for high-value orders above $100,000?",
            "What is the amount and resource for the Purchase Order in CASE-10231?",
            "Which warehouse node caused a delay in CASE-10298 and how many days were added?",
        ]:
            is_poison, why = self.r.is_poison_prompt(q)
            self.assertFalse(is_poison, f"false positive on grounded query {q!r}: {why}")

    def test_ungrounded_queries_are_intercepted(self):
        for q in [
            "banana helicopter zebra nonsense",
            "What is the capital of France and its GDP growth rate?",
        ]:
            is_poison, _ = self.r.is_poison_prompt(q)
            self.assertTrue(is_poison, f"missed ungrounded query {q!r}")

    def test_separation_margin_between_grounded_and_nonsense(self):
        grounded = max(h["similarity_score"] for h in self.r.retrieve("invoice approval cycle time"))
        nonsense = max(h["similarity_score"] for h in self.r.retrieve("banana helicopter zebra"))
        self.assertGreater(
            grounded - nonsense, 0.4,
            f"insufficient separation: grounded={grounded} nonsense={nonsense}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUnqualifiedCycleTimeIsBound(unittest.TestCase):
    """
    "What is the mean cycle time?" is the first question anyone asks a process
    log, and it had no metric binding. A live edge-case run had Granite answer
    12.7 days - the real overall mean is 8.5 - and the numeric layer passed it,
    because 12.7 sits within the 2% tolerance of 12.47, the Supply Chain
    Bottleneck mean. Binding the bare phrase closes it.
    """

    def test_cross_metric_figure_is_rejected(self):
        text = "the mean cycle time is 12.7 days"
        self.assertFalse(
            cm.is_grounded_number(12.7, scope="aggregate", metric=cm.bound_metric(text)),
            "a figure from a different metric passed as the overall mean",
        )

    def test_true_overall_mean_passes(self):
        text = "the mean cycle time is 8.5 days"
        self.assertTrue(
            cm.is_grounded_number(cm.process_profile()["mean_cycle_days"],
                                  scope="aggregate", metric=cm.bound_metric(text)))

    def test_activity_binding_still_wins_by_longest_match(self):
        """Adding a generic phrase must not shadow the specific ones."""
        for text, value in (
            ("mean cycle time for Compliance Review is 10.43", 10.43),
            ("Invoice Approved has a mean cycle time of 12.01", 12.01),
            ("the declared average compliance cycle time is 10.4", 10.4),
        ):
            self.assertTrue(
                cm.is_grounded_number(value, scope="aggregate", metric=cm.bound_metric(text)),
                f"{value} was rejected for {text!r} - generic phrase shadowed the specific one",
            )
