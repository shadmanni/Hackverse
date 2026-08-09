"""
Stage 2 live tests: the log-probabilities must actually come from Granite.

Skipped automatically if the model weights are not present locally, so CI and
teammates without the download still get a green suite.

Run: python -m pytest test_stage2_granite_live.py -v -s
"""

import os
import unittest

MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-3.3-2b-instruct")


def _weights_present() -> bool:
    """
    The weights are fetched with allow_patterns, so the cached snapshot is
    deliberately partial and snapshot_download(local_files_only=True) rejects it.
    Check for the files we actually need instead.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg = try_to_load_from_cache(MODEL_ID, "config.json")
        if not isinstance(cfg, str):
            return False
        from pathlib import Path
        return any(Path(cfg).parent.glob("*.safetensors"))
    except Exception:
        return False


@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not downloaded yet")
class TestGraniteLogprobsAreReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from granite_runner import GraniteRunner
        cls.runner = GraniteRunner.get()

    def test_logprobs_are_valid_and_varying(self):
        steps = list(self.runner.stream(
            self.runner.build_prompt("List three colours."), max_new_tokens=32))
        self.assertGreater(len(steps), 3, "model emitted almost nothing")
        lps = [s.logprob for s in steps]
        for lp in lps:
            self.assertLessEqual(lp, 0.0, "a log-probability cannot be positive")
        self.assertGreater(
            len(set(round(lp, 4) for lp in lps)), 1,
            "log-probabilities are constant - these are not real model outputs",
        )
        # The old code pinned every token at exactly -0.1 / -0.05.
        self.assertNotEqual(set(round(lp, 2) for lp in lps), {-0.10})
        self.assertNotEqual(set(round(lp, 2) for lp in lps), {-0.05})

    def test_top_probs_form_a_distribution(self):
        steps = list(self.runner.stream(
            self.runner.build_prompt("Say hello."), max_new_tokens=12))
        for s in steps:
            self.assertAlmostEqual(sum(s.top_probs), 1.0, places=4)
            self.assertEqual(s.top_probs, sorted(s.top_probs, reverse=True))
            self.assertTrue(all(0.0 <= p <= 1.0 for p in s.top_probs))

    def test_chosen_token_is_the_argmax_under_greedy(self):
        """
        `prob` is the true probability from the full 49k-vocab softmax, while
        `top_probs` is the top-5 head RENORMALISED to sum to 1. So the chosen
        token's renormalised weight is always >= its true probability, by exactly
        the mass the truncation discarded. Asserting equality is wrong.

        Consequence for calibration: Shannon entropy over the renormalised head
        UNDERSTATES true next-token entropy. tau must be calibrated against these
        same top-5 values, not against full-vocabulary entropy.
        """
        steps = list(self.runner.stream(
            self.runner.build_prompt("Say hello."), max_new_tokens=8, temperature=0.0))
        for s in steps:
            self.assertEqual(max(s.top_probs), s.top_probs[0], "top_probs not sorted")
            self.assertGreaterEqual(
                max(s.top_probs) + 1e-6, s.prob,
                "renormalised head cannot be below the true probability",
            )
            # Truncation discards little mass when the model is confident.
            self.assertLess(max(s.top_probs) - s.prob, 0.2,
                            "top-5 head is missing too much probability mass")

    def test_uncertainty_is_higher_on_unanswerable_questions(self):
        """The core premise: the model's own confidence separates known from invented."""
        import statistics
        answerable = list(self.runner.stream(self.runner.build_prompt(
            "What is 2 + 2? Answer with the number only."), max_new_tokens=10))
        invented = list(self.runner.stream(self.runner.build_prompt(
            "State the exact internal ledger balance of case CC-99812 in the "
            "Zorblax division to the cent."), max_new_tokens=30))
        mean_answerable = statistics.mean(s.logprob for s in answerable)
        mean_invented = statistics.mean(s.logprob for s in invented)
        print(f"\n  mean logprob answerable={mean_answerable:.4f} invented={mean_invented:.4f}")
        self.assertLess(mean_invented, mean_answerable,
                        "model was not less confident on the fabricated question")


@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not downloaded yet")
class TestEndToEndInterception(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from granite_runner import GraniteRunner
        from sentinel_stream import SentinelStream
        cls.stream = SentinelStream(runner=GraniteRunner.get(), retriever=None, max_new_tokens=80)

    def test_real_stream_produces_measured_telemetry(self):
        evs = list(self.stream.run("What is the mean compliance cycle time?"))
        self.assertTrue(evs)
        final = evs[-1]
        self.assertIn(final.kind, {"done", "recovery"})
        toks = [e for e in evs if e.kind == "token"]
        if toks:
            lps = [e.payload["logprob"] for e in toks]
            self.assertGreater(len(set(lps)), 1, "logprobs constant across a real stream")
        for e in evs:
            if e.kind == "done":
                self.assertGreater(e.payload["entropy_overhead_ms"], 0)
                self.assertLess(e.payload["overhead_pct"], 5.0,
                                "entropy check should be a rounding error on decode time")
                print(f"\n  overhead: {e.payload['entropy_overhead_ms']:.3f} ms "
                      f"({e.payload['overhead_pct']:.4f}% of {e.payload['elapsed_ms']:.0f} ms)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
