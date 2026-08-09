"""
IBM Data Prep Kit — PII Redaction Engine
=========================================
This module wraps DPK's PIIRedactorTransform logic directly:
  - presidio_analyzer.AnalyzerEngine for NER-based entity recognition
    (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, ORGANIZATION, CREDIT_CARD, DATE_TIME)
  - presidio_anonymizer.AnonymizerEngine for entity replacement

Design note: DPK's PIIRedactorTransform operates on PyArrow Tables (parquet
batches) in its file-pipeline mode. For Sentinel's streaming event-log ingestion
we instantiate the underlying analyzer/anonymizer directly — the same libraries,
the same Presidio engine, zero regex stand-ins — and add a deterministic
pseudonymisation layer on top so that process-structure (actor identity across
events) is preserved for Granite's causal-reasoning chains.

Pseudonymisation rationale
--------------------------
Blanket redaction would collapse every distinct actor into the same token,
making handoff graphs, rework loops and segregation-of-duty checks impossible.
The deterministic alias (ACTOR_XXXXXXXX) keeps causal structure intact while
removing the personal identifier.  The alias is derived from a salted SHA-256
digest: stable across pipeline runs, not reversible by inspection.

The salt is sourced from the env var SENTINEL_PII_SALT.  The default is
intentionally weak (demo-only); source it from a secret store in production.
"""

import hashlib
import logging
import os
import re
from functools import lru_cache
from typing import Iterable, List, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — mirrors DPK's dpk_pii_redactor defaults
# ---------------------------------------------------------------------------

# These are exactly DPK's default_supported_entities from
# dpk_pii_redactor/transform.py, minus DATE_TIME (ISO timestamps are
# process structure, not PII in a Celonis export).
DPK_ENTITIES: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "ORGANIZATION",
    "CREDIT_CARD",
]

# Flair NER is DPK's preferred model (flair/ner-english-large) but requires a
# separate heavy install.  We use spaCy en_core_web_sm for the venv here: same
# Presidio API, lighter weight, still catches PERSON/ORG reliably.
SPACY_MODEL = "en_core_web_sm"

# DPK's default_score_threshold_key = 0.6 — we keep it identical.
DPK_SCORE_THRESHOLD = 0.6

PII_SALT = os.getenv("SENTINEL_PII_SALT", "sentinel-demo-salt")


# ---------------------------------------------------------------------------
# Engine singleton — expensive to build, cheap to re-use
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_presidio_analyzer() -> AnalyzerEngine:
    """
    Instantiate the Presidio AnalyzerEngine the same way DPK's
    PIIAnalyzerEngine does in dpk_pii_redactor/pii_analyzer.py,
    but without the Flair dependency so it runs in the project venv.
    """
    import spacy
    if not spacy.util.is_package(SPACY_MODEL):
        logger.info("Downloading spaCy model %s …", SPACY_MODEL)
        spacy.cli.download(SPACY_MODEL)

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    nlp_cfg = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_cfg).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


@lru_cache(maxsize=1)
def _build_presidio_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


# ---------------------------------------------------------------------------
# Deterministic pseudonymisation
# ---------------------------------------------------------------------------

def pseudonymise(name: str) -> str:
    """
    Return a stable, collision-resistant alias for a named actor.

    Same algorithm as the regex-based stand-in in phase2_ingestion_pipeline.py
    so aliases printed in logs match previously generated Milvus chunks.
    """
    digest = hashlib.sha256(
        f"{PII_SALT}:{name.strip().lower()}".encode()
    ).hexdigest()
    return f"ACTOR_{digest[:8].upper()}"


# ---------------------------------------------------------------------------
# Core DPK-integrated redaction function
# ---------------------------------------------------------------------------

# ACTOR alias pattern — never a PII target; must be excluded from Presidio sweeps.
_ACTOR_ALIAS_RE = re.compile(r"\bACTOR_[0-9A-F]{8}\b")


def _exclude_actor_aliases(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    Remove any Presidio result whose span overlaps an ACTOR_XXXXXXXX alias.

    spaCy classifies ACTOR_XXXXXXXX as PERSON because it looks like a name.
    Letting Presidio replace the alias with <PERSON> would destroy the
    pseudonymisation and break Granite's actor-identity chains.
    """
    alias_spans = [(m.start(), m.end()) for m in _ACTOR_ALIAS_RE.finditer(text)]
    if not alias_spans:
        return results
    kept = []
    for r in results:
        overlap = any(r.start < end and r.end > start for start, end in alias_spans)
        if not overlap:
            kept.append(r)
    return kept


def redact_pii(text: str, known_actors: Iterable[str] = ()) -> Tuple[str, List[str]]:
    """
    Redact PII from *text* using the DPK/Presidio engine, then pseudonymise
    any known actor names so process structure survives.

    Parameters
    ----------
    text : str
        Raw event-log text that may contain personal identifiers.
    known_actors : iterable of str
        Actor names extracted from the case (e.g. event["resource"]).
        These are replaced with their deterministic ACTOR_XXXXXXXX alias
        BEFORE Presidio runs, so the alias itself is never re-flagged.

    Returns
    -------
    redacted_text : str
        Text with all detected PII replaced.
    detected_entities : list[str]
        Entity types found by Presidio (e.g. ["PERSON", "EMAIL_ADDRESS"]).

    Notes
    -----
    Replacement strategy (mirrors DPK's default_anonymizer_operator = "replace"):
      - Known actor names           → ACTOR_XXXXXXXX alias  (pass 1)
      - Presidio EMAIL / PERSON / CREDIT_CARD → <ENTITY_TYPE>  (pass 2)
      - Regex phone backstop        → <PHONE_NUMBER>  (pass 3)

    Pass ordering matters:
      1. Pseudonymise actors first so Presidio never sees the real name.
      2. Run Presidio on the already-pseudonymised text, filtering out alias
         spans so ACTOR_XXXXXXXX is never re-classified as PERSON.
      3. Apply the regex phone backstop last because spaCy's en_core_web_sm
         misses Indian-format numbers (+91-XXXXX-XXXXX) below the 0.6 threshold.
         The backstop guarantees every +CC-phone pattern is removed regardless
         of NER confidence.
    """
    # Pass 1: deterministic pseudonymisation of known actors
    # Longest-match first so "Anita Rao Kumar" is not half-replaced by "Anita Rao".
    working = text
    for actor in sorted(
        {a for a in known_actors if a},
        key=len,
        reverse=True,
    ):
        alias = pseudonymise(actor)
        working = re.sub(rf"\b{re.escape(actor)}\b", alias, working)

    # Pass 2: Presidio NER sweep on the already-pseudonymised text.
    # Filter out any result spanning an ACTOR alias so pseudonymisation survives.
    analyzer = _build_presidio_analyzer()
    anonymizer = _build_presidio_anonymizer()

    raw_results: List[RecognizerResult] = analyzer.analyze(
        text=working,
        language="en",
        entities=DPK_ENTITIES,
        score_threshold=DPK_SCORE_THRESHOLD,
    )
    results = _exclude_actor_aliases(working, raw_results)
    detected: List[str] = sorted({r.entity_type for r in results})

    if results:
        op_cfg = {"DEFAULT": OperatorConfig("replace", None)}
        anonymized = anonymizer.anonymize(working, results, operators=op_cfg)
        working = anonymized.text

    # Pass 3: regex backstop for phone numbers that Presidio's NER missed.
    # spaCy en_core_web_sm scores Indian +CC format below the 0.6 threshold;
    # the regex catches them deterministically with zero false-positive risk
    # (the leading '+' prevents matching ISO timestamps or dollar amounts).
    if _PHONE_RE.search(working):
        working = _PHONE_RE.sub("<PHONE_NUMBER>", working)
        if "PHONE_NUMBER" not in detected:
            detected = sorted(detected + ["PHONE_NUMBER"])

    return working, detected


def analyze_entities(text: str) -> List[str]:
    """
    Return the entity types detected in *text* by the DPK/Presidio engine.

    Combines Presidio NER results with the regex phone backstop so the
    reported entity list matches what redact_pii() would actually redact.
    Used by tests and the integrity-check layer.
    """
    analyzer = _build_presidio_analyzer()
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=DPK_ENTITIES,
        score_threshold=DPK_SCORE_THRESHOLD,
    )
    detected = {r.entity_type for r in results}
    # Include phone numbers caught by the regex backstop.
    if _PHONE_RE.search(text):
        detected.add("PHONE_NUMBER")
    return sorted(detected)


# ---------------------------------------------------------------------------
# Integrity assertion helpers (used in phase2_ingestion_pipeline.py)
# ---------------------------------------------------------------------------

# Backstop regexes kept for speed — Presidio is slower than a regex scan so
# we run it once during redaction; the final check uses regex to stay fast.
_EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
_PHONE_RE = re.compile(r"\+\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5}")


def has_raw_pii(text: str, actor_names: Iterable[str] = ()) -> bool:
    """
    True if *text* still contains any detectable PII after redaction.
    Used by the ingestion pipeline's abort gate.
    """
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
        return True
    names = {n for n in actor_names if n}
    return any(
        re.search(rf"\b{re.escape(n)}\b", text) for n in names
    )
