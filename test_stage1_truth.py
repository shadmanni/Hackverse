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

    def test_chunk_level_queries_are_well_supported(self):
        """Queries that name an entity a chunk describes must retrieve it."""
        for q in [
            "What is the cycle time for invoice approval",
            "What is the average compliance cycle time for high-value orders above $100,000?",
            "What is the amount and resource for the Purchase Order in CASE-10231?",
            "Which warehouse node caused a delay in CASE-10298 and how many days were added?",
        ]:
            weak, why = self.r.retrieval_support(q)
            self.assertFalse(weak, f"chunk-level query reported unsupported: {q!r}: {why}")

    def test_nonsense_retrieves_nothing_worth_showing(self):
        for q in [
            "banana helicopter zebra nonsense",
            "What is the capital of France and its GDP growth rate?",
        ]:
            weak, _ = self.r.retrieval_support(q)
            self.assertTrue(weak, f"nonsense query reported as supported: {q!r}")

    def test_retrieval_support_is_not_an_answerability_verdict(self):
        """
        The measured reason the keyword list could not simply be swapped for a
        score threshold. Both of these are real and both break the assumption
        that a high score means answerable and a low score means not:

          a corpus-level question no single chunk resembles scores LOW but is
          answerable from the aggregates;
          a fabricated question about real entities retrieves those entities and
          scores HIGH while the log holds neither figure it asks for.

        If this ever starts passing as a clean separation, the architecture
        argument in phase3_rag_retriever's header needs re-measuring, not the
        test deleting.
        """
        low_but_answerable, _ = self.r.retrieval_support("How many cases are in the event log?")
        high_but_unanswerable, _ = self.r.retrieval_support(
            "What was the throughput delay and inventory holding time at warehouse node W-88?")
        self.assertTrue(
            low_but_answerable or not high_but_unanswerable,
            "retrieval score now separates answerability on this fixture - the "
            "claim that it cannot is no longer supported by the data",
        )

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


class TestStatisticBinding(unittest.TestCase):
    """
    A figure can be real, belong to the right metric, and still be a lie because
    it is the WRONG STATISTIC of that metric. Set membership cannot see it.

    Expected values are written here by hand rather than read back out of
    celonis_metrics, so these fail if the underlying numbers move.
    """

    def check(self, text: str, value: float) -> bool:
        return cm.is_grounded_number(
            value, scope="aggregate",
            metric=cm.bound_metric(text), statistic=cm.bound_statistic(text),
        )

    def test_max_stated_as_the_mean_is_rejected(self):
        # Compliance Review: mean 10.43, max 16. Both real; only one is "average".
        self.assertFalse(self.check("Compliance Review took 16 days on average", 16),
                         "the activity's MAX passed as its MEAN")

    def test_each_statistic_passes_under_its_own_name(self):
        self.assertTrue(self.check("Compliance Review took 10.43 days on average", 10.43))
        self.assertTrue(self.check("the longest Compliance Review took 16 days", 16))

    def test_unnamed_statistic_falls_back_and_never_tightens(self):
        # No statistic word, so the check must stay exactly as permissive as it
        # was before this feature existed - this is the no-regression guarantee.
        for value in (10.43, 16, 122):
            text = f"Compliance Review covers {value}"
            self.assertIsNone(cm.bound_statistic(text))
            self.assertTrue(self.check(text, value))

    def test_ambiguous_statistic_does_not_bind(self):
        # Naming two statistics must not enforce one of them arbitrarily.
        self.assertIsNone(cm.bound_statistic("the mean and the longest Compliance Review"))


class TestTheSystemPassesItsOwnChecker(unittest.TestCase):
    """
    Everything Sentinel writes into the prompt, or returns as its verified
    answer, must survive the check it applies to the model.

    This is the invariant that was silently false. The deterministic recovery
    answer - generated from the event log and true by construction - was
    reported as containing ungrounded figures on four of five queries, because
    metric and statistic binding read those words out of a NEIGHBOURING clause:
    "occurs 122 times ... with a mean cycle time of 10.43" bound the count 122 to
    "mean", and the process title "Order-to-Cash (O2C) Compliance Cycle" bound
    every later figure in the sentence to the order-to-cash metric.

    A checker that rejects its own ground truth would have shown judges a
    NOT GROUNDED badge on the one answer that cannot be wrong.
    """

    def setUp(self):
        from sentinel_stream import SentinelStream
        self.s = SentinelStream(runner=None, retriever=None)

    def test_aggregate_block_is_self_consistent(self):
        """Every figure handed to the model as VERIFIED AGGREGATES must pass."""
        from sentinel_stream import SentinelStream
        block = SentinelStream._aggregate_block()
        offending, _ = self.s._scan_new_figures(block, 0, complete_only=False)
        self.assertIsNone(
            offending,
            f"the context we give the model states {offending}, which our own "
            f"grounding check rejects - the model would be penalised for quoting it",
        )

    def test_deterministic_recovery_answers_are_self_consistent(self):
        for q in (
            "Give the exact mean cycle time for Compliance Review.",
            "What is the mean compliance cycle time?",
            "How many cases are in the event log?",
            "How many orders are above $100,000?",
            "What percentage of orders were flagged as supply chain bottlenecks?",
            "What is the average approval delay?",
        ):
            answer = cm.ground_truth_answer(q)
            offending, _ = self.s._scan_new_figures(answer, 0, complete_only=False)
            self.assertIsNone(
                offending,
                f"recovery answer for {q!r} states {offending}, which the "
                f"grounding check rejects:\n  {answer}",
            )

    def test_clause_binding_does_not_cross_into_the_next_claim(self):
        """The specific attribution bug, pinned."""
        text = "'Compliance Review' occurs 122 times in the log with a mean cycle time of 10.43 days"
        i = text.index("122")
        self.assertIsNone(
            cm.bound_statistic(self.s._statistic_span(text, i, i + 3)),
            "a count was bound to 'mean' from a later clause",
        )

    def test_process_title_does_not_bind_later_figures(self):
        title = "Order-to-Cash (O2C) Compliance Cycle (Celonis EMS): 150 cases"
        self.assertIsNone(
            cm.bound_metric(self.s._clause(title)),
            "the process name bound figures to the order-to-cash metric",
        )


class TestMetricNamedAfterTheFigure(unittest.TestCase):
    """
    English names the metric after the figure at least as often as before it,
    and the span only looked backwards.

    "3 events are recorded for the Invoice Approved activity" bound to no metric,
    fell back to the whole-log check, and passed because 3 is some event's cycle
    time. The real count is 150. This was a live answer on a green-path demo
    prompt - a flat fabrication streaming clean past both layers.
    """

    def setUp(self):
        from sentinel_stream import SentinelStream
        self.s = SentinelStream(runner=None, retriever=None)

    def _caught(self, text):
        return self.s._scan_new_figures(text, 0, complete_only=False)[0] is not None

    def test_false_count_with_trailing_metric_is_caught(self):
        self.assertTrue(
            self._caught("3 events are recorded for the Invoice Approved activity."),
            "a fabricated count bound to no metric because the metric came after it",
        )

    def test_true_count_with_trailing_metric_passes(self):
        self.assertFalse(
            self._caught("There are 150 events recorded for the Invoice Approved activity."))

    def test_lookahead_is_confined_to_the_sentence(self):
        """A metric in the NEXT sentence must not bind backwards into this one."""
        self.assertFalse(
            self._caught("There are 150 events. Invoice Approved has a mean cycle time of 12.01 days."))

    def test_midstream_binding_is_unaffected(self):
        """
        Mid-decode there is no text after the figure, so the lookahead must be
        a no-op rather than a crash or a changed verdict.
        """
        partial = "The mean compliance cycle time is 10.4"
        self.assertFalse(self.s._scan_new_figures(partial, 0, complete_only=False)[0] is not None)


class TestGenericAliasLosesToSpecific(unittest.TestCase):
    """
    "cycle time" names the DIMENSION. Every activity metric in the table is a
    cycle time, so the generic phrase co-occurs with the specific one in almost
    every real sentence, and neither length nor position separates them:

        "'Supply Chain Bottleneck Flagged' is the biggest bottleneck with a
         mean cycle time of 12.47"

    "bottleneck" and "cycle time" are both ten characters and the generic sits
    nearer the figure, so it won on both rules. 12.47 is that activity's correct
    mean; it was halted against the overall 8.5, and the recovery then answered
    with 12.47 - the system contradicting itself in front of the audience.
    """

    def setUp(self):
        from sentinel_stream import SentinelStream
        self.s = SentinelStream(runner=None, retriever=None)

    def _caught(self, text):
        return self.s._scan_new_figures(text, 0, complete_only=False)[0] is not None

    def test_specific_metric_wins_over_the_dimension(self):
        for text in (
            "The activity 'Supply Chain Bottleneck Flagged' is the biggest bottleneck "
            "with a mean cycle time of 12.47 days.",
            "Across 122 high-value orders above $100,000, the mean cycle time is 9.11 days.",
            "mean cycle time for Compliance Review is 10.43 days",
            "Invoice Approved has a mean cycle time of 12.01 days.",
        ):
            self.assertFalse(self._caught(text), f"correct figure halted: {text}")

    def test_generic_still_binds_when_it_is_the_only_metric_named(self):
        """Dropping the generic tier entirely would reopen the 12.7 leak."""
        self.assertTrue(self._caught("The mean cycle time is 12.7 days."))
        self.assertFalse(self._caught("The mean cycle time is 8.5 days."))

    def test_wrong_statistic_of_the_right_metric_is_still_caught(self):
        """Specificity must not become a blanket pass for the bound metric."""
        self.assertTrue(
            self._caught("Invoice Approved has a mean cycle time of 23 days."),
            "23 is that activity's MAX stated as its mean",
        )


class TestLogWideFactsSurviveMetricBinding(unittest.TestCase):
    """
    How many cases the log holds is a fact about the LOG, not a value of
    whatever metric the sentence happens to mention.

    Binding made them mutually exclusive: "150 cases are in the order-to-cash
    process" narrowed the admissible set to order-to-cash's single declared
    value of 12.0 and halted on 150 - the correct case count. Measured live,
    that is verbatim what Granite answers to "How many cases are in the
    order-to-cash process?", so the firewall halted its own correct answer on a
    question a judge asks in the first minute.

    The exemption is withheld for mean/median/max/rate, because there the figure
    IS being claimed as that statistic of that metric.
    """

    def setUp(self):
        from sentinel_stream import SentinelStream
        self.s = SentinelStream(runner=None, retriever=None)

    def _caught(self, text):
        return self.s._scan_new_figures(text, 0, complete_only=False)[0] is not None

    def test_population_counts_pass_next_to_a_named_metric(self):
        for text in (
            "150 cases are in the order-to-cash process.",
            "There are 150 cases in the order-to-cash process.",
            "The mean cycle time across 460 events is 8.5 days.",
            "38 out of 150 cases were flagged as supply-chain bottlenecks.",
        ):
            self.assertFalse(self._caught(text), f"true log-wide figure halted: {text}")

    def test_range_bounds_pass(self):
        self.assertFalse(self._caught("Cycle times range between 1 and 23 days."))

    def test_the_exemption_does_not_swallow_a_statistic_claim(self):
        """
        The false negative this could have introduced. 150 is a real log-wide
        count, so exempting it unconditionally would let it be asserted as a
        mean of anything.
        """
        self.assertTrue(
            self._caught("The mean compliance cycle time is 150 days."),
            "a population count passed as a fabricated mean",
        )
        self.assertTrue(self._caught("Invoice Approved has a mean cycle time of 460 days."))

    def test_clock_times_are_not_quantitative_claims(self):
        """`47` was being read out of 16:47:00Z and halted as a fabrication."""
        self.assertFalse(
            self._caught("Case CASE-10001 was created on 2026-06-04T16:47:00Z."))
        # The digits of a timestamp must not be parsed as figures at all.
        from sentinel_stream import _NUM_RE
        self.assertEqual(
            [m.group() for m in _NUM_RE.finditer("occurred on 2026-06-04T16:47:00Z")], [])
