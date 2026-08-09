"""
Regression table from the adversarial grounding audit (2026-08-09).

46 defects, each found by one lens and then independently attacked by a
refuter told to default to "not a defect". These are what survived.

  38 false positives - a TRUE sentence marked `contradicted`. This is the
     expensive direction: it halts a correct answer in front of judges and
     makes the system indict itself.
   8 false negatives - a FALSE sentence not caught. Cheaper, since the
     entropy layer is a second chance, but still a defect.

Why this table exists rather than more asserts in demo():
TestTheSystemPassesItsOwnChecker pins the aggregate block and
ground_truth_answer, which are FIXED STRINGS that happen to avoid these
shapes. Every defect below is in model-generated prose - the recovery
answer - which no fixed-string test can reach.

Ground truth (read from cm.process_profile(), do not hardcode elsewhere):
  total 150 cases / 460 events; cycle time mean 8.5, median 9.0, max 23
  Compliance Review mean 10.43, max 16, count 122
  declared compliance 10.4; declared order-to-cash 12.0
  high-value orders count 122, mean cycle 9.11, threshold $100,000

Run: python -m pytest test_grounding_regression.py -v
"""

import unittest

from sentinel_stream import SentinelStream


# (sentence, must_contradict, lens, why)
CASES = [
    (
        "'Compliance Review' occurs 16 times in the log.",
        True ,  # alias-shadowing
        # 'occurs ... times' matches nothing in _STATISTIC_WORDS (count words are only
        # 'number of', 'how many', 'count of', 'total of'), so statistic=None and t…
    ),
    (
        'Bottleneck flagging affects 38% of orders.',
        True ,  # alias-shadowing
        # _ungrounded_number's percentage guard is written 'if "%" in span and metric is
        # None', so naming any metric disables it. A bare '%' matches no _STATIST…
    ),
    (
        'Compliance Review has 16 events.',
        True ,  # wrong-statistic
        # _STATISTIC_WORDS['count'] only matches nominal forms: 'number of', 'how many',
        # 'count of', 'total of'. Every natural predicate form of a count misses …
    ),
    (
        'Compliance Review takes at most 10.43 days.',
        True ,  # wrong-statistic
        # 'at most' appears in BOTH _STATISTIC_WORDS['max'] and _QUALIFIER_BEFORE. The
        # qualifier branch wins in audit_figures and forces metric=None, so the sta…
    ),
    (
        'Compliance Review took 16 days, on average.',
        True ,  # wrong-statistic
        # This is the exact claim the layer was built to reject, with one comma added.
        # _statistic_span clips the right edge at the first _CLAUSE_BREAK and ',' i…
    ),
    (
        'Compliance Review usually takes 16 days.',
        True ,  # wrong-statistic
        # _STATISTIC_WORDS['mean'] contains 'typically' but not 'usually', 'normally',
        # 'generally', or the adjective 'typical'. All four slip: 'Compliance Revie…
    ),
    (
        'The average Compliance Review duration across 122 cases was 16 days.',
        True ,  # wrong-statistic
        # _statistic_span left-bounds at the end of the PREVIOUS figure, so 16's span starts
        # after '122', and 'across' is additionally a clause break — 'average…
    ),
    (
        'The highest cycle time was 8.5 days.',
        True ,  # wrong-statistic
        # _STATISTIC_WORDS['max'] lists maximum/longest/slowest/worst/at most/peak/up to but
        # not 'highest', the most ordinary superlative for a duration. bound_…
    ),
    (
        'Compliance Review (122 events): the mean cycle time is 10.43 days.',
        False,  # alias-shadowing
        # 10.43 is by_activity['Compliance Review']['mean_cycle_days'] exactly. _metric_span
        # splits on _SENTENCE_BREAK, which includes '(' and ':', so the speci…
    ),
    (
        'Order-to-cash averages 12.0 days, and the median cycle time is 9.0 days.',
        False,  # alias-shadowing
        # Both figures are true (declared_avg_order_to_cash_days 12.0, median_cycle_days
        # 9.0). 'order-to-cash' (13) outranks 'cycle time' (10) across the comma,…
    ),
    (
        'Supply-chain bottlenecks were flagged on 38 of the 150 cases, adding a mean 12.47 days.',
        False,  # alias-shadowing
        # _QUALIFIER_BEFORE matches r'\d[\d,.]*\s+of\s+\$?$', so it exempts '38 of 150' but
        # not '38 of the 150' - one article defeats it. The denominator then b…
    ),
    (
        'The amount for the Purchase Order in CASE-10002 is $130,960.29.',
        False,  # alias-shadowing
        # $130,960.29 is the literal amount_usd on CASE-10002's Purchase Order Created
        # event. metric_statistics()['purchase order'] admits only {mean 2.4, max 4…
    ),
    (
        'The mean cycle time for Compliance Review is 10.43 days, well above the overall mean of 8.5 days.',
        False,  # alias-shadowing
        # A comma is not a _SENTENCE_BREAK, so _metric_span for the second figure is the
        # whole sentence and 'compliance review' - named in the first clause - go…
    ),
    (
        'Compliance Review averages 10.43 days, while the bottleneck step averages 12.47 days.',
        False,  # multi-clause
        # 12.47 IS the Supply Chain Bottleneck Flagged mean (by_activity['Supply Chain
        # Bottleneck Flagged']['mean_cycle_days']=12.47), and 10.43 is Compliance R…
    ),
    (
        'Invoice Approved averages 12.01 days, and the overall mean cycle time is 8.5 days.',
        False,  # multi-clause
        # 8.5 is process_profile()['mean_cycle_days'], the log's own overall mean, and the
        # clause explicitly says 'overall mean cycle time'. The correct alias '…
    ),
    (
        'Mean cycle time 8.5 days, declared compliance cycle time 10.4 days, median 9.0 days.',
        False,  # multi-clause
        # All three figures are the log's own values (8.5 mean, 10.4 declared compliance,
        # 9.0 median). The system's real recovery string orders them 'Mean cycle…
    ),
    (
        'Order-to-cash is declared at 12.0 days, and the observed mean cycle time is 8.5 days.',
        False,  # multi-clause
        # declared_avg_order_to_cash_days=12.0 and mean_cycle_days=8.5 are both true, and
        # the sentence explicitly contrasts declared vs observed. 'order-to-cash…
    ),
    (
        'The declared compliance cycle time is 10.4 days and the overall mean is 8.5 days.',
        False,  # multi-clause
        # declared_avg_compliance_cycle_time_days=10.4 and mean_cycle_days=8.5 are both true
        # and both are in the VERIFIED AGGREGATES block the model is told to …
    ),
    (
        'The mean compliance cycle time is 10.4 days, and the median across all events is 9.0 days.',
        False,  # multi-clause
        # median_cycle_days=9.0 is the log's overall median and the clause says 'across all
        # events'. The metric span is the whole sentence, so 'compliance cycle…
    ),
    (
        'The mean cycle time for high-value orders is 9.11 days, and the overall mean is 8.5 days.',
        False,  # multi-clause
        # high_value_mean_cycle_days=9.11 and mean_cycle_days=8.5 are both true. 'high-value
        # orders' (17 chars) beats 'cycle time' (10) on length and governs th…
    ),
    (
        'There are 150 cases in the order-to-cash process.',
        False,  # multi-clause
        # total_cases=150. The metric name appears AFTER the figure inside the same clause,
        # so _metric_span's forward reach pulls 'order-to-cash' in; but here '…
    ),
    (
        '38 of 150 cases, or 25.33%, were flagged as supply-chain bottlenecks.',
        False,  # number-forms
        # _metric_span's forward reach is `_CLAUSE_BREAK.split(tail)[0]`, and the appositive
        # comma immediately after '25.33%' makes that first clause empty, so …
    ),
    (
        'Bottleneck flags cover 25.33% of cases.',
        False,  # number-forms
        # _QUALIFIER_BEFORE's alternation `above|below|over|under|exceeding|...` has no word
        # boundaries, so any word ENDING in one of them and followed by white…
    ),
    (
        'Case CASE-10001 was created on 2026-06-04T16:47:00Z.',
        False,  # number-forms
        # _NUM_RE's lookbehind (?<![A-Za-z0-9_.\-]) blocks digits after '-' and after
        # letters, so 2026 / 06 / 04 / 16(after 'T') are correctly suppressed, but '…
    ),
    (
        'Compliance Review covers 122 of the 460 events.',
        False,  # number-forms
        # _QUALIFIER_BEFORE's denominator branch is `\d[\d,.]*\s+of\s+\$?$` - it requires
        # 'of' to sit immediately before the figure. 'of the 460' inserts an art…
    ),
    (
        'Compliance Review cycle times range from 4 to 16 days.',
        False,  # number-forms
        # metric_statistics() exposes only mean/median/max/count/rate/threshold - there is
        # no `min` key, so 'compliance review' admits exactly {10.43, 16.0, 122…
    ),
    (
        'For case CASE-10001 the cycle time so far is 2 days.',
        False,  # number-forms
        # scope is correctly 'all' (no aggregate word), but the phrase 'cycle time' binds
        # the metric, and metric binding overrides scope in is_grounded_number: …
    ),
    (
        'Total logged order value is $110.6M.',
        False,  # number-forms
        # Two defects in one. (a) Same scale-word blindness as the spelled-out form. (b) A
        # regex misparse: on '$110.6M' the trailing lookahead (?![A-Za-z0-9_\-]…
    ),
    (
        'Total order value across the log is approximately $110.6 million.',
        False,  # number-forms
        # _numbers_in strips '$' and ',' and parses the mantissa alone; the scale word
        # 'million' is prose the checker never reads, so 110.6 is compared against …
    ),
    (
        '38 out of 150 cases were flagged as supply-chain bottlenecks.',
        False,  # qualifier-slots
        # Both figures are true: by_activity['Supply Chain Bottleneck Flagged'].event_count
        # == 38 (and declared orders_flagged_for_bottleneck == 38) out of tota…
    ),
    (
        'Bottlenecks affect 25.33% of the 150 cases in the log.',
        False,  # qualifier-slots
        # 25.33 is exactly the rate metric_statistics stores for 'bottleneck' (38/150*100)
        # and 150 is total_cases, so the sentence is true end to end and the pe…
    ),
    (
        'Compliance Review accounts for 122 of the 460 events in the log.',
        False,  # qualifier-slots
        # Both figures are exact: Compliance Review event_count == 122, total_events == 460.
        # The guard requires digits immediately before 'of' plus optional '$'…
    ),
    (
        'Cycle times range between 1 and 23 days.',
        False,  # qualifier-slots
        # The event log's cycle_time_days run 1..23 (min(load_events cycle times) == 1,
        # max_cycle_days == 23), so the sentence is exactly true. 'between X and Y…
    ),
    (
        'The 122 high-value orders in excess of $100,000 have a mean cycle time of 9.11 days.',
        False,  # qualifier-slots
        # Every figure is the log's own: high_value_orders == 122,
        # high_value_mean_cycle_days == 9.11, HIGH_VALUE_USD == 100000.0, and
        # metric_statistics()['high…
    ),
    (
        'The 122 orders in excess of $100,000 have a mean cycle time of 9.11 days.',
        False,  # qualifier-slots
        # Worst observed case: a wholly true sentence in which every single figure is
        # contradicted. Without the literal token 'high-value'/'high value' in the s…
    ),
    (
        'The bottleneck rate is 38 per 150 cases.',
        False,  # qualifier-slots
        # 38 flagged of 150 cases is the log's own bottleneck rate (25.33% is stored in
        # metric_statistics as bottleneck/rate, derived as 38/150*100). 'per' is n…
    ),
    (
        'The mean cycle time across all 460 events is 8.5 days.',
        False,  # qualifier-slots
        # mean_cycle_days == 8.5 over total_events == 460; both figures are the log's own.
        # 460 sits in a population/scope slot with no qualifier the guard recog…
    ),
    (
        '38 of the 150 orders were flagged for a supply chain bottleneck.',
        False,  # true-statements
        # 'bottleneck' in the trailing clause binds 150, and metric_aliases['bottleneck'] =
        # {12.47, 38.0, 25.33}. sentinel_stream.py line 87 already carries a c…
    ),
    (
        '8.5 days is the mean cycle time recorded across all 460 events.',
        False,  # true-statements
        # 'cycle time' is named before the figure and _metric_span lets a metric govern the
        # rest of the sentence, so the EVENT COUNT 460 is checked against metr…
    ),
    (
        'Across the 150 cases the mean cycle time is 8.5 days.',
        False,  # true-statements
        # Bleed in the BACKWARD direction: _metric_span appends the forward clause, so
        # 'cycle time' sitting after 150 still binds to it. The figure is a case co…
    ),
    (
        'Compliance Review averages 10.43 days versus an overall mean of 8.5 days.',
        False,  # true-statements
        # Same cluster, confirming it is not an artefact of the parenthetical in the
        # previous case. 'versus' is not in _CLAUSE_BREAK, so the specific activity m…
    ),
    (
        'Compliance Review, at 10.43 days mean cycle time, is slower than the overall mean of 8.5 days.',
        False,  # true-statements
        # A metric named once governs the remainder of the sentence, so the comparison
        # BASELINE — which deliberately belongs to a broader metric — is checked ag…
    ),
    (
        'That is 25.33% of the 150 total orders flagged as bottlenecks.',
        False,  # true-statements
        # The percentage itself verifies, then the denominator that makes it true is halted,
        # because 'bottleneck' binds 150 to {12.47, 38.0, 25.33}. The system …
    ),
    (
        'The average cycle time is 8.5 days for 150 cases.',
        False,  # true-statements
        # Same forward metric bleed. 150 is the population denominator, not a cycle-time
        # value, but 'cycle time' binds it and metric_aliases['cycle time'] admit…
    ),
    (
        'Across 122 Compliance Review events the average was 10.43 days.',
        False,  # wrong-statistic
        # process_profile() gives Compliance Review event_count=122 and
        # mean_cycle_days=10.43, so every figure here is exactly right. But _statistic_span
        # for 12…
    ),
    (
        'The mean cycle time across 460 events is 8.5 days.',
        False,  # wrong-statistic
        # process_profile() total_events=460, mean_cycle_days=8.5. 460 is a
        # denominator/population size, not a cycle time, but _QUALIFIER_BEFORE only
        # neutralise…
    ),
]


class TestGroundingRegressionTable(unittest.TestCase):
    """
    Both surfaces must agree, because they have disagreed before: a defect
    visible only in audit_figures is a reporting bug, but one that also
    reaches _scan_new_figures physically halts the stream.
    """

    @classmethod
    def setUpClass(cls):
        cls.s = SentinelStream(runner=None)

    def test_audit_matches_expected_state(self):
        wrong = []
        for sentence, must_contradict, *_ in CASES:
            got = self.s.audit_figures(sentence)["contradicted"] > 0
            if got != must_contradict:
                kind = "FALSE POSITIVE" if got else "MISSED"
                wrong.append(f"  [{kind}] {sentence}")
        self.assertFalse(
            wrong,
            f"\n{len(wrong)}/{len(CASES)} cases wrong:\n" + "\n".join(wrong),
        )

    def test_halt_path_agrees_with_audit(self):
        """A true sentence must not halt; the audit state alone is not enough."""
        wrong = []
        for sentence, must_contradict, *_ in CASES:
            halted = self.s._scan_new_figures(sentence, 0, complete_only=False)[0] is not None
            if halted != must_contradict:
                kind = "HALTS A TRUE SENTENCE" if halted else "does not halt"
                wrong.append(f"  [{kind}] {sentence}")
        self.assertFalse(
            wrong,
            f"\n{len(wrong)}/{len(CASES)} cases wrong on the halt path:\n" + "\n".join(wrong),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
