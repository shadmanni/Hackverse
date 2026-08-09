"""
PII masking tests.

The UI claims "Zero-Knowledge PII" and the deck claims Data Prep Kit redaction.
Before this suite existed, every actor's real name went into the vector store in
clear, and the pipeline's own integrity check could not detect it: the check
scanned for emails and phone numbers, neither of which chunk_event ever wrote
into the chunk text, so it passed vacuously on every run.

Run: python -m pytest test_pii_masking.py -v
"""

import json
import re
import unittest
from pathlib import Path

from phase2_ingestion_pipeline import (
    EMAIL_RE,
    PHONE_RE,
    actor_names,
    chunk_event,
    chunk_summary,
    pseudonymise,
    redact_pii,
)

BASE = Path(__file__).parent.resolve()
DATASETS = ["data/mock_celonis_data.json", "data/mock_celonis_data_large.json"]


def load(name):
    return json.loads((BASE / name).read_text())


class TestNoDirectIdentifiersReachTheVectorStore(unittest.TestCase):

    def test_no_actor_name_survives_in_any_chunk(self):
        for ds in DATASETS:
            payload = load(ds)
            names = actor_names(payload)
            self.assertTrue(names, f"{ds}: no actors found - fixture wrong?")
            for ev in payload["events"]:
                text = chunk_event(ev, names)["text"]
                for n in names:
                    self.assertIsNone(
                        re.search(rf"\b{re.escape(n)}\b", text),
                        f"{ds}: actor name {n!r} leaked into a chunk:\n{text}",
                    )

    def test_no_email_or_phone_survives(self):
        for ds in DATASETS:
            payload = load(ds)
            names = actor_names(payload)
            for ev in payload["events"]:
                text = chunk_event(ev, names)["text"]
                self.assertIsNone(EMAIL_RE.search(text), f"{ds}: email leaked:\n{text}")
                self.assertIsNone(PHONE_RE.search(text), f"{ds}: phone leaked:\n{text}")

    def test_free_text_identifiers_are_caught_by_the_backstop(self):
        names = {"Anita Rao"}
        out = redact_pii("Ping anita.rao@acmecorp.com or +91-98765-43210 about Anita Rao", names)
        self.assertNotIn("anita.rao@acmecorp.com", out)
        self.assertNotIn("+91-98765-43210", out)
        self.assertNotIn("Anita Rao", out)


class TestPseudonymisationPreservesAnalysis(unittest.TestCase):
    """Blanket redaction would collapse every actor into one token and make
    handoffs, rework loops and segregation-of-duty checks uncomputable."""

    def test_alias_is_stable_for_the_same_actor(self):
        self.assertEqual(pseudonymise("Anita Rao"), pseudonymise("Anita Rao"))
        self.assertEqual(pseudonymise("anita rao"), pseudonymise("  Anita Rao "))

    def test_distinct_actors_get_distinct_aliases(self):
        payload = load("data/mock_celonis_data_large.json")
        names = actor_names(payload)
        aliases = {pseudonymise(n) for n in names}
        self.assertEqual(len(aliases), len(names), "alias collision destroys handoff analysis")

    def test_alias_does_not_contain_the_original_name(self):
        for n in ("Anita Rao", "Deepak Menon"):
            alias = pseudonymise(n)
            self.assertNotIn(n.lower().replace(" ", ""), alias.lower())
            self.assertTrue(alias.startswith("ACTOR_"))

    def test_actor_identity_is_still_recoverable_within_the_log(self):
        """Same person across two events must map to the same alias."""
        payload = load("data/mock_celonis_data_large.json")
        names = actor_names(payload)
        by_actor = {}
        for ev in payload["events"]:
            if not ev.get("resource"):
                continue
            alias = re.search(r"ACTOR_[0-9A-F]{8}", chunk_event(ev, names)["text"])
            self.assertIsNotNone(alias, "no alias written into chunk")
            by_actor.setdefault(ev["resource"], set()).add(alias.group())
        for actor, aliases in by_actor.items():
            self.assertEqual(len(aliases), 1, f"{actor} got inconsistent aliases {aliases}")


class TestIntegrityCheckCanActuallyFail(unittest.TestCase):
    """A check that cannot fail is not a check."""

    def test_unredacted_name_is_detected(self):
        names = {"Deepak Menon"}
        leaked = "Case CASE-1: handled by Deepak Menon (Finance)."
        hit = any(re.search(rf"\b{re.escape(n)}\b", leaked) for n in names)
        self.assertTrue(hit, "integrity predicate fails to spot a name in clear")

    def test_summary_chunk_carries_no_identifiers(self):
        payload = load("data/mock_celonis_data_large.json")
        text = chunk_summary(payload["summary_metrics"])["text"]
        for n in actor_names(payload):
            self.assertNotIn(n, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
