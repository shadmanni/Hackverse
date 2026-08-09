"""
PII masking tests — DPK/Presidio integration.

The pipeline now uses IBM Data Prep Kit's Presidio stack
(presidio_analyzer + presidio_anonymizer) via dpk_pii_engine.py, which
mirrors the internals of dpk_pii_redactor's PIIRedactorTransform.

Test coverage:
  1. No direct actor name reaches the vector store.
  2. No email or phone survives in any chunk.
  3. Free-text fields (note) are caught by the DPK/Presidio backstop.
  4. Pseudonymisation is stable (same actor → same alias across events).
  5. Distinct actors get distinct aliases (no collision → handoff analysis intact).
  6. DPK/Presidio entity detection works for each supported entity type.
  7. The integrity gate (has_raw_pii) correctly fires on unredacted text.

Run: python -m pytest test_pii_masking.py -v
"""

import json
import re
import unittest
from pathlib import Path

# Imported from the pipeline, not from dpk_pii_engine directly: the pipeline
# selects the DPK/Presidio engine when it imports and the regex path otherwise,
# so these suites assert the guarantee the vector store actually got, under
# whichever engine ran. Suite 4 tests Presidio itself and skips without it.
from phase2_ingestion_pipeline import (
    actor_names,
    chunk_event,
    chunk_summary,
    has_raw_pii,
    pseudonymise,
    redact_pii,
)

try:
    from dpk_pii_engine import DPK_ENTITIES, analyze_entities
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False

BASE = Path(__file__).parent.resolve()
DATASETS = ["data/mock_celonis_data.json", "data/mock_celonis_data_large.json"]


def load(name):
    return json.loads((BASE / name).read_text())


# ---------------------------------------------------------------------------
# Backstop regex patterns — kept here for the integrity assertions only;
# the pipeline itself uses dpk_pii_engine.has_raw_pii().
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
_PHONE_RE = re.compile(r"\+\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5}")


# ===========================================================================
# Suite 1 — No direct identifiers reach the vector store
# ===========================================================================

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
                self.assertIsNone(_EMAIL_RE.search(text), f"{ds}: email leaked:\n{text}")
                self.assertIsNone(_PHONE_RE.search(text), f"{ds}: phone leaked:\n{text}")

    def test_free_text_identifiers_are_redacted(self):
        """Identifiers in the free-text `note` field must not survive."""
        names = {"Anita Rao"}
        raw = "Ping anita.rao@acmecorp.com or +91-98765-43210 about Anita Rao"
        out, _entities = redact_pii(raw, known_actors=names)
        self.assertNotIn("anita.rao@acmecorp.com", out)
        self.assertNotIn("+91-98765-43210", out)
        self.assertNotIn("Anita Rao", out)


# ===========================================================================
# Suite 2 — Pseudonymisation preserves process-mining structure
# ===========================================================================

class TestPseudonymisationPreservesAnalysis(unittest.TestCase):
    """Blanket redaction collapses every actor into one token and makes
    handoffs, rework loops and SoD checks uncomputable."""

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


# ===========================================================================
# Suite 3 — Integrity gate can actually fire
# ===========================================================================

class TestIntegrityCheckCanActuallyFail(unittest.TestCase):
    """A check that cannot fail is not a check."""

    def test_unredacted_name_is_detected_by_has_raw_pii(self):
        names = {"Deepak Menon"}
        leaked = "Case CASE-1: handled by Deepak Menon (Finance)."
        self.assertTrue(has_raw_pii(leaked, actor_names=names))

    def test_clean_text_passes_integrity_gate(self):
        alias = pseudonymise("Deepak Menon")
        clean = f"Case CASE-1: handled by {alias} (Finance)."
        self.assertFalse(has_raw_pii(clean, actor_names={"Deepak Menon"}))

    def test_summary_chunk_carries_no_identifiers(self):
        payload = load("data/mock_celonis_data_large.json")
        text = chunk_summary(payload["summary_metrics"])["text"]
        for n in actor_names(payload):
            self.assertNotIn(n, text)


# ===========================================================================
# Suite 4 — DPK/Presidio entity detection
# ===========================================================================

@unittest.skipUnless(HAS_PRESIDIO, "presidio not installed in this venv (see .venv-dpk)")
class TestDPKPresidioEntityDetection(unittest.TestCase):
    """
    Validate that the DPK/Presidio engine (presidio_analyzer via dpk_pii_engine)
    detects the entity types DPK advertises as supported.
    """

    def test_email_entity_detected(self):
        entities = analyze_entities("Contact user@example.com for details.")
        self.assertIn("EMAIL_ADDRESS", entities)

    def test_person_entity_detected(self):
        entities = analyze_entities("This was approved by Anita Rao from procurement.")
        self.assertIn("PERSON", entities)

    def test_phone_entity_detected(self):
        entities = analyze_entities("Call +91-98765-43210 to confirm.")
        self.assertIn("PHONE_NUMBER", entities)

    def test_redact_pii_returns_entity_list(self):
        """redact_pii() must return (text, entity_list) not just text."""
        result = redact_pii("Email user@corp.com", known_actors=())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        _, entities = result
        self.assertIsInstance(entities, list)

    def test_redacted_text_has_no_email(self):
        text, _ = redact_pii("Contact user@corp.com immediately.", known_actors=())
        self.assertIsNone(_EMAIL_RE.search(text))

    def test_alias_matches_the_regex_fallback_exactly(self):
        """
        Both engines must derive the same alias from the same name. They are
        selected by which venv is running, and the 461 chunks already in Milvus
        were written by one of them - a different formula would orphan every
        actor reference in the store while every test above still passed.
        """
        import hashlib
        import os

        from dpk_pii_engine import pseudonymise as dpk_alias
        salt = os.getenv("SENTINEL_PII_SALT", "sentinel-demo-salt")
        for name in ("Anita Rao", "Deepak Menon"):
            expect = "ACTOR_" + hashlib.sha256(
                f"{salt}:{name.lower()}".encode()
            ).hexdigest()[:8].upper()
            self.assertEqual(dpk_alias(name), expect)

    def test_all_dpk_entities_in_supported_list(self):
        """The direct-identifier types the pipeline is configured for."""
        for entity in ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"):
            self.assertIn(entity, DPK_ENTITIES)

    def test_organization_is_not_redacted(self):
        """
        ORGANIZATION is off by design. spaCy tags process vocabulary as
        organizations, so enabling it rewrites activity and department names out
        of the chunk - see the DPK_ENTITIES comment for the measurement.
        """
        self.assertNotIn("ORGANIZATION", DPK_ENTITIES)

    def test_activity_names_survive_redaction(self):
        """
        Retrieval matches on the activity name. If redaction removes it the
        chunk is still PII-clean and completely unretrievable, which is the
        failure mode that passes every other test in this file.
        """
        payload = load("data/mock_celonis_data_large.json")
        names = actor_names(payload)
        for ev in payload["events"][:60]:
            text = chunk_event(ev, names)["text"]
            self.assertIn(ev["activity"], text,
                          f"activity {ev['activity']!r} was redacted out of its own chunk")
            self.assertIn(ev["department"], text,
                          f"department {ev['department']!r} was redacted out of its chunk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
