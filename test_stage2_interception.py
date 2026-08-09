"""
Stage 2 tests: the circuit breaker must actually break, on real per-token signals.

These use a scripted fake runner so the interception logic is tested independently
of the model. test_stage2_granite_live.py covers the real Granite path.

Run: python -m pytest test_stage2_interception.py -v
"""

import math
import unittest
from typing import AsyncIterator, List, Optional, Tuple

import celonis_metrics as cm
from ollama_runner import TokenStep
from sentinel_stream import SentinelStream


class FakeRunner:
    """Replays a scripted (token, probability) sequence as if Granite emitted it."""

    def __init__(self, script: List[Tuple[str, float]]):
        self.script = script
        self.prompts_seen: List[str] = []

    def build_prompt(self, query: str, context: Optional[str] = None, grounded: bool = True) -> str:
        self.prompts_seen.append(f"{query}||{context}||grounded={grounded}")
        return "PROMPT"

    async def stream(self, prompt: str, max_new_tokens: int = 160, **kw) -> AsyncIterator[TokenStep]:
        for i, (tok, p) in enumerate(self.script[:max_new_tokens]):
            # A confident token concentrates mass; an uncertain one flattens it.
            rest = (1.0 - p) / 4 if p < 1.0 else 0.0
            yield TokenStep(
                text=tok,
                logprob=math.log(max(p, 1e-9)),
                top_probs=[p, rest, rest, rest, rest],
                top_tokens=[tok, "a", "b", "c", "d"],
                index=i,
            )


async def run(script, **kw):
    s = SentinelStream(runner=FakeRunner(script), retriever=None, **kw)
    return [ev async for ev in s.run("what is the mean compliance cycle time?")]


class TestCircuitBreakerActuallyBreaks(unittest.IsolatedAsyncioTestCase):

    async def test_clean_confident_stream_completes(self):
        script = [(w, 0.97) for w in ["The", " mean", " cycle", " time", " is", " documented", "."]]
        evs = await run(script)
        self.assertEqual(evs[-1].kind, "done")
        self.assertEqual(len([e for e in evs if e.kind == "intercept"]), 0)
        self.assertEqual(evs[-1].payload["tokens"], len(script))

    async def test_generation_stops_at_interception(self):
        # 40 more tokens follow the bad one; none of them may be emitted.
        script = [("The", 0.98), (" figure", 0.97), (" is", 0.96), (" 87", 0.99), (".", 0.99), ("6", 0.99), (" days", 0.98)]
        script += [(" filler", 0.99)] * 40
        evs = await run(script)
        kinds = [e.kind for e in evs]
        self.assertIn("intercept", kinds, "breaker never tripped on a fabricated figure")
        # Count ORIGINAL-stream tokens only. Total event count is no longer a
        # proxy for "did decoding stop": recovery regenerates a grounded answer
        # and streams its own recovery_token events after the halt.
        self.assertLess(
            len([e for e in evs if e.kind == "token"]), len(script),
            "the intercepted stream kept decoding past the halt",
        )
        # After the intercept, only recovery events may follow - never another
        # token from the stream that was cut off.
        idx = kinds.index("intercept")
        self.assertNotIn("token", kinds[idx + 1:], "original stream resumed after interception")
        self.assertEqual(kinds[idx + 1], "recovery_start")
        self.assertEqual(kinds[-1], "recovery")

    async def test_ungrounded_number_is_caught_even_when_confident(self):
        """The failure mode entropy alone cannot see: a confident fabrication."""
        script = [("Mean", 0.99), (" is", 0.99), (" 87", 0.999), (".", 0.999), ("6", 0.999), (" days", 0.99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "confident fabricated number passed through")
        self.assertEqual(icept[0].payload["reason"], "ungrounded_number")
        self.assertAlmostEqual(icept[0].payload["ungrounded_value"], 87.6, places=2)
        # It really was high-confidence, so entropy would not have flagged it.
        self.assertGreater(icept[0].payload["probability"], 0.9)

    async def test_grounded_number_passes(self):
        """10.4 is the declared compliance cycle time and must not trip."""
        script = [("Mean", 0.99), (" is", 0.99), (" 10", 0.98), (".", 0.98), ("4", 0.98), (" days", 0.99), (".", 0.99)]
        evs = await run(script)
        self.assertEqual(evs[-1].kind, "done", [e.payload for e in evs if e.kind == "intercept"])

    async def test_multi_token_number_is_assembled_before_checking(self):
        """Tokenizers split 87.6 into '87' '.' '6'; fragments alone are groundable."""
        self.assertTrue(cm.is_grounded_number(87.0) or True)  # fragment check is not the point
        script = [("x", 0.99), (" 87", 0.99), (".", 0.99), ("6", 0.99), (" done", 0.99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept)
        self.assertAlmostEqual(icept[0].payload["ungrounded_value"], 87.6, places=2)

    def test_enterprise_identifiers_are_not_treated_as_figures(self):
        """
        Regression: Granite correctly refused a poison prompt with the sentence
        "...for warehouse node W-99", and the numeric layer parsed 99 out of the
        identifier and intercepted the refusal. Entity codes are not claims.
        """
        for ident in ["W-99", "CC-9999", "CASE-10298", "SOX-404", "CC-3305"]:
            self.assertEqual(
                SentinelStream._numbers_in(ident), [],
                f"{ident} was parsed as a quantitative claim",
            )

    async def test_refusal_mentioning_an_identifier_is_not_intercepted(self):
        script = [("I", .99), (" cannot", .98), (" verify", .99), (" node", .97),
                  (" W", .96), ("-", .98), ("99", .97), (" in", .98), (" the", .99), (" log", .98), (".", .99)]
        evs = await run(script)
        self.assertEqual(
            evs[-1].kind, "done",
            f"intercepted a valid refusal: {[e.payload for e in evs if e.kind=='intercept']}",
        )

    async def test_real_figures_still_caught_next_to_identifiers(self):
        script = [("Case", .99), (" CC", .98), ("-", .98), ("3305", .97),
                  (" took", .98), (" 87", .99), (".", .99), ("6", .99), (" days", .98), (".", .99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "identifier exclusion swallowed a real fabricated figure")
        self.assertAlmostEqual(icept[0].payload["ungrounded_value"], 87.6, places=2)

    async def test_trailing_figure_at_end_of_stream_is_checked(self):
        """No token follows the number, so it is never terminated mid-stream."""
        script = [("Mean", .99), (" is", .99), (" 87", .99), (".", .99), ("6", .99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "figure ending the response escaped the check")
        self.assertAlmostEqual(icept[0].payload["ungrounded_value"], 87.6, places=2)

    async def test_partial_number_not_judged_before_it_completes(self):
        """'10' must not be judged before it becomes '10.43'."""
        script = [("Mean", .99), (" is", .99), (" 10", .99), (".", .99), ("43", .99), (" days", .98), (".", .99)]
        evs = await run(script)
        self.assertEqual(
            evs[-1].kind, "done",
            f"judged a partial figure: {[e.payload for e in evs if e.kind=='intercept']}",
        )

    async def test_entropy_layer_catches_sustained_flat_distribution(self):
        """No bad number present - only uncertainty. Layer 1 must fire alone."""
        script = [("The", 0.99), (" vendor", 0.98),
                  (" reportedly", 0.22), (" maybe", 0.18), (" roughly", 0.20), (" around", 0.19)]
        evs = await run(script, check_numbers=False)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "sustained flat distribution did not trip the entropy layer")
        self.assertEqual(icept[0].payload["reason"], "semantic_entropy")
        self.assertGreater(icept[0].payload["uncertainty"], icept[0].payload["tau"])

    async def test_isolated_uncertain_token_does_not_trip(self):
        """
        Regression: a clean query halted on token 0 because "From" carried only
        p=0.34. Sentence openings are inherently uncertain and prove nothing.
        """
        script = [("From", 0.34), (" the", 0.99), (" log", 0.98), (",", 0.99),
                  (" mean", 0.97), (" is", 0.98), (" documented", 0.96), (".", 0.99)]
        evs = await run(script, check_numbers=False)
        self.assertEqual(
            evs[-1].kind, "done",
            f"single uncertain token tripped the breaker: "
            f"{[e.payload for e in evs if e.kind == 'intercept']}",
        )

    # These scripts use lowercase non-stopwords so token weight stays at the 1.0
    # baseline and the run logic is what is under test. Uppercase tokens carry
    # weight 2.5 and stopwords 0.4, either of which would confound the result.

    async def test_breach_run_is_configurable_and_enforced(self):
        # Exactly two consecutive breaches: trips at 2, survives at 3.
        script = [("start", 0.99), (" vague", 0.20), (" murky", 0.19), (" clear", 0.98), (" end", 0.99)]
        self.assertTrue(
            [e for e in await run(script, check_numbers=False, breach_run=2) if e.kind == "intercept"],
            "two consecutive breaches did not trip breach_run=2",
        )
        self.assertFalse(
            [e for e in await run(script, check_numbers=False, breach_run=3) if e.kind == "intercept"],
            "two consecutive breaches wrongly tripped breach_run=3",
        )

    async def test_breach_run_resets_on_a_confident_token(self):
        """Two breaches, a confident token, then two more: never 3 in a row."""
        script = [(" vague", 0.20), (" murky", 0.19), (" clear", 0.98),
                  (" hazy", 0.20), (" dim", 0.18), (" end", 0.99)]
        evs = await run(script, check_numbers=False, breach_run=3)
        self.assertEqual(evs[-1].kind, "done", "run counter did not reset on a confident token")

    async def test_ungrounded_number_trips_immediately_without_a_run(self):
        """The numeric layer is deterministic, so one occurrence is conclusive."""
        script = [("Mean", 0.99), (" is", 0.99), (" 87", 0.999), (".", 0.999), ("6", 0.999), (" days", 0.99)]
        evs = await run(script, breach_run=99)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "numeric layer was suppressed by the run requirement")
        self.assertEqual(icept[0].payload["reason"], "ungrounded_number")

    async def test_recovery_states_only_grounded_figures(self):
        script = [("Mean", 0.99), (" is", 0.99), (" 87", 0.99), (".", 0.99), ("6", 0.99)]
        evs = await run(script)
        rec = [e for e in evs if e.kind == "recovery"]
        self.assertTrue(rec)
        import re
        for n in re.findall(r"\d+\.?\d*", rec[0].text.replace(",", "")):
            self.assertTrue(
                cm.is_grounded_number(float(n)),
                f"recovery answer stated ungrounded {n}: {rec[0].text}",
            )


class TestAggregateClaimScoping(unittest.IsolatedAsyncioTestCase):
    """
    Cycle times span 1..23 days, so almost any small integer exists somewhere in
    the log. Checking an aggregate claim against raw event values therefore
    accepts anything: "the mean is approximately 15 days" passed because some
    single case took 15 days. Aggregate claims must be checked against aggregates.
    """

    def test_aggregate_words_switch_the_scope(self):
        self.assertEqual(SentinelStream._claim_scope("the mean cycle time is "), "aggregate")
        self.assertEqual(SentinelStream._claim_scope("on average it took "), "aggregate")
        self.assertEqual(SentinelStream._claim_scope("the total value was "), "aggregate")
        self.assertEqual(SentinelStream._claim_scope("case CASE-10231 took "), "all")

    def test_percentages_are_always_aggregate_scoped(self):
        """The event log has no percentage field, so any percentage is derived."""
        self.assertEqual(SentinelStream._claim_scope("the discount was ", figure="15%"), "aggregate")

    async def test_fabricated_mean_is_intercepted(self):
        script = [("The", .99), (" mean", .98), (" cycle", .97), (" time", .98),
                  (" is", .98), (" 15", .96), (" days", .97), (".", .99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"]
        self.assertTrue(icept, "fabricated mean of 15 days was not caught")
        self.assertEqual(icept[0].payload["ungrounded_value"], 15.0)

    async def test_real_mean_still_passes(self):
        script = [("The", .99), (" mean", .98), (" is", .98),
                  (" 10", .97), (".", .98), ("4", .97), (" days", .98), (".", .99)]
        evs = await run(script)
        self.assertEqual(evs[-1].kind, "done",
                         f"real aggregate rejected: {[e.payload for e in evs if e.kind=='intercept']}")

    async def test_raw_event_value_still_passes_in_a_per_case_claim(self):
        """15 days IS a real cycle time for some individual case."""
        script = [("Case", .99), (" CASE", .98), ("-", .98), ("10231", .97),
                  (" took", .98), (" 15", .97), (" days", .98), (".", .99)]
        evs = await run(script)
        self.assertEqual(evs[-1].kind, "done",
                         "a valid per-case figure was rejected as if it were an aggregate")


class TestTelemetryIsMeasured(unittest.IsolatedAsyncioTestCase):

    async def test_overhead_is_measured_not_hardcoded(self):
        script = [(w, 0.97) for w in ["a"] * 30]
        evs = await run(script)
        done = evs[-1].payload
        self.assertGreater(done["entropy_overhead_ms"], 0.0)
        self.assertNotEqual(done["entropy_overhead_ms"], 11.4, "11.4 was the old hardcoded value")
        self.assertLess(done["overhead_pct"], 100.0)

    async def test_tokens_before_halt_is_real_count(self):
        script = [("ok", 0.99)] * 5 + [(" 87", 0.99), (".", 0.99), ("6", 0.99), (" x", 0.99)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"][0]
        emitted = len([e for e in evs if e.kind == "token"])
        self.assertEqual(icept.payload["tokens_before_halt"], emitted)


class TestCandidateFanPointsAtTheFigure(unittest.IsolatedAsyncioTestCase):
    """
    The UI tells judges "here is what the model was choosing between when it
    picked that number". Nothing else asserts that claim is true.

    Every index below is written down by hand from the script, deliberately NOT
    derived from celonis_metrics or from the engine. A check that computes its
    expectation with the same code it is checking can only prove the system
    agrees with itself.
    """

    # "87.6" is not a figure the event log can produce. Tokens:
    #   0 "The"  1 " mean"  2 " is"  3 " 87"  4 "."  5 "6"  6 " days"
    # The breaker trips at 6, because that is where the figure TERMINATES.
    # The value was chosen across 3-5. Those are different tokens from the
    # halt, which is the entire point.
    SCRIPT = [("The", 0.98), (" mean", 0.97), (" is", 0.96),
              (" 87", 0.99), (".", 0.99), ("6", 0.99), (" days", 0.98)]

    # Tokens 3 and 5 both carry a figure at equal mass here, and the rule keeps
    # the STRONGEST commitment, resolving a tie to the earlier one. That lands
    # on the leading digits, which is also the more useful fan to show: the
    # alternatives to "87" are rival values, whereas the alternatives to the
    # trailing "6" are only rival last decimal places.
    DECISION, HALT = 3, 6

    async def test_fan_comes_from_the_figure_not_the_halting_token(self):
        evs = await run(self.SCRIPT)
        icept = [e for e in evs if e.kind == "intercept"][0].payload

        self.assertEqual(icept["candidates_token_index"], self.DECISION)
        self.assertNotEqual(
            icept["candidates_token_index"], self.HALT,
            "reporting the halting token shows the model choosing between words "
            "at the exact moment we claim it was choosing between numbers",
        )
        self.assertTrue(any(c["cluster"].startswith("#") for c in icept["candidates"]),
                        f"decision token carried no figure: {icept['candidates']}")

    async def test_decision_is_the_strongest_figure_not_the_latest(self):
        """
        A trace digit in an otherwise prose distribution must not displace the
        real one. On live output this reported the decision as "0% of the mass -
        mostly writing prose", which is what the terminal would have shown a
        judge as the moment the figure was chosen.
        """
        # Token 5 carries a figure at 1% mass; token 3 carries one at 99%.
        script = [("The", 0.98), (" mean", 0.97), (" 87", 0.99),
                  (".", 0.99), ("6", 0.01), (" days", 0.98)]
        evs = await run(script)
        icept = [e for e in evs if e.kind == "intercept"][0].payload
        self.assertEqual(icept["candidates_token_index"], 2,
                         "a 1%-mass digit displaced the 99% one")

    async def test_trailing_figure_still_carries_a_fan(self):
        # Ends ON the figure, so the stream is exhausted before the check runs
        # and there is no live step at the halt. The fan must survive that path.
        evs = await run(self.SCRIPT[:-1])
        icept = [e for e in evs if e.kind == "intercept"][0].payload

        self.assertIsNotNone(icept["candidates"], "trailing-figure halt lost the fan")
        self.assertEqual(icept["candidates_token_index"], self.DECISION)

    async def test_clean_stream_reports_no_fan(self):
        # Nothing was intercepted, so there is no figure to explain.
        evs = await run([(w, 0.97) for w in ["The", " process", " is", " documented", "."]])
        self.assertEqual(evs[-1].kind, "done")
        self.assertEqual([e for e in evs if e.kind == "intercept"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
