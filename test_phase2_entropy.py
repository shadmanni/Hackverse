import unittest
from draft_logprob_engine import DraftLogprobExtractor
from entropy_engine import EntropyEngine

class TestPhase2Entropy(unittest.TestCase):

    def setUp(self):
        self.extractor = DraftLogprobExtractor()
        self.engine = EntropyEngine(threshold_tau=0.65, window_size=5)

    def test_granite_logprob_parsing(self):
        sample_payload = {
            "token": "Celonis",
            "logprob": -0.05,
            "top_logprobs": {
                "Celonis": -0.05,
                "SAP": -3.2,
                "Oracle": -4.5
            }
        }
        token, logprob, probs = self.extractor.parse_granite_token_logprobs(sample_payload)
        self.assertEqual(token, "Celonis")
        self.assertAlmostEqual(logprob, -0.05)
        self.assertTrue(len(probs) == 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_draft_estimation_fallback(self):
        context = ["According", "to", "the", "Celonis", "logs"]
        numeric_token = "42.8_days"
        logprob = self.extractor.estimate_draft_logprob(context, numeric_token)
        self.assertLess(logprob, -1.5)

    def test_evaluate_granite_payload_factual_vs_hallucination(self):
        self.engine.reset()
        
        # Factual payload
        factual_payload = {
            "token": "process",
            "logprob": -0.02,
            "top_logprobs": {"process": -0.02, "workflow": -3.5}
        }
        is_hallucinating, uncertainty, var = self.engine.evaluate_granite_payload(factual_payload)
        self.assertFalse(is_hallucinating)

        # Hallucination payload with entropy/logprob drop
        hallucinated_payload = {
            "token": "fabricated_number_$10M",
            "logprob": -3.10,
            "top_logprobs": {"fabricated_number_$10M": -3.10, "valid": -3.20, "other": -3.30}
        }
        is_hallucinating, uncertainty, var = self.engine.evaluate_granite_payload(hallucinated_payload)
        self.assertTrue(is_hallucinating)

    def test_reset_clears_ema_state(self):
        self.engine.evaluate_token("prior", logprob=-4.0)
        self.engine.reset()

        fresh_engine = EntropyEngine(threshold_tau=0.65, window_size=5)
        expected = fresh_engine.evaluate_token("$100", logprob=-0.1)
        actual = self.engine.evaluate_token("$100", logprob=-0.1)

        self.assertEqual(actual, expected)

    def test_contrastive_pmi_requires_no_context_logprob(self):
        self.assertEqual(self.engine.compute_contrastive_pmi(-2.0, None), 0.0)
        self.assertAlmostEqual(self.engine.compute_contrastive_pmi(-2.0, -0.2), 1.8)

if __name__ == "__main__":
    unittest.main()
