"""
Live tests: the log-probabilities must actually come from Granite via Ollama.

Skipped automatically when the Ollama daemon is down or the model is not pulled,
so the suite stays green for teammates who have not pulled it yet.

Run: python -m pytest test_stage2_granite_live.py -v -s
"""

import os
import unittest

MODEL = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _model_available() -> bool:
    try:
        import requests
        tags = requests.get(f"{HOST}/api/tags", timeout=5).json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        return MODEL in names or f"{MODEL}:latest" in names
    except Exception:
        return False


@unittest.skipUnless(_model_available(), f"{MODEL} not available via Ollama")
class TestGraniteLogprobsAreReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ollama_runner import OllamaRunner
        cls.runner = OllamaRunner()

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
        # The original implementation pinned every token at -0.1 / -0.05.
        self.assertNotEqual(set(round(lp, 2) for lp in lps), {-0.10})
        self.assertNotEqual(set(round(lp, 2) for lp in lps), {-0.05})

    def test_top_probs_form_a_distribution(self):
        steps = list(self.runner.stream(
            self.runner.build_prompt("Say hello."), max_new_tokens=12))
        self.assertTrue(steps)
        for s in steps:
            self.assertAlmostEqual(sum(s.top_probs), 1.0, places=4)
            self.assertEqual(s.top_probs, sorted(s.top_probs, reverse=True))
            self.assertTrue(all(0.0 <= p <= 1.0 for p in s.top_probs))

    def test_logprobs_arrive_during_streaming_not_after(self):
        """
        The product intercepts mid-generation. If logprobs only landed with the
        final response the audit would be post-hoc, which is exactly the design
        this project replaces. Assert the first token carries its distribution.
        """
        gen = self.runner.stream(self.runner.build_prompt("Count to five."), max_new_tokens=20)
        first = next(gen)
        self.assertLessEqual(first.logprob, 0.0)
        self.assertGreaterEqual(len(first.top_probs), 2, "no top-k on the first streamed token")
        gen.close()

    def test_uncertainty_is_higher_on_unanswerable_questions(self):
        """The core premise: the model's own confidence separates known from invented."""
        import statistics
        answerable = list(self.runner.stream(self.runner.build_prompt(
            "What is 2 + 2? Answer with the number only."), max_new_tokens=10))
        invented = list(self.runner.stream(self.runner.build_prompt(
            "State the exact internal ledger balance of case CC-99812 in the "
            "Zorblax division to the cent."), max_new_tokens=30))
        self.assertTrue(answerable and invented)
        mean_answerable = statistics.mean(s.logprob for s in answerable)
        mean_invented = statistics.mean(s.logprob for s in invented)
        print(f"\n  mean logprob answerable={mean_answerable:.4f} invented={mean_invented:.4f}")
        self.assertLess(mean_invented, mean_answerable,
                        "model was not less confident on the fabricated question")


@unittest.skipUnless(_model_available(), f"{MODEL} not available via Ollama")
class TestEndToEndInterception(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ollama_runner import OllamaRunner
        from sentinel_stream import SentinelStream
        cls.stream = SentinelStream(runner=OllamaRunner(), retriever=None, max_new_tokens=80)

    def test_real_stream_produces_measured_telemetry(self):
        evs = list(self.stream.run("What is the mean compliance cycle time?"))
        self.assertTrue(evs)
        self.assertIn(evs[-1].kind, {"done", "recovery"})
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
