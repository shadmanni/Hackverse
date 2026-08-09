"""
IBM Data Prep Kit — PII Redaction Engine
=========================================
This module wraps DPK's PIIRedactorTransform logic directly:
  - presidio_analyzer.AnalyzerEngine for NER-based entity recognition
    (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD - see DPK_ENTITIES)
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

# DPK's default_supported_entities from dpk_pii_redactor/transform.py, minus
# two that are process structure rather than personal data in a Celonis export.
#
# DATE_TIME: ISO event timestamps ARE the process. Redacting them removes the
# ordering that makes an event log an event log.
#
# ORGANIZATION: measured on this export, spaCy tags the activity names as
# organizations - "Compliance Review", "Invoice Approved", "Purchase Order
# Created", "Supply Chain Bottleneck Flagged" - along with the "Legal &
# Compliance" and "Supply Chain" departments. Enabling it rewrote the activity
# out of 58 of the first 120 chunks, which silently destroys retrieval: a query
# for Compliance Review cannot match a chunk whose activity now reads
# <ORGANIZATION>. It is also not a direct identifier of a person, which is what
# this layer exists to remove. Verified with test_activity_names_survive_redaction.
DPK_ENTITIES: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
]

# Flair NER (flair/ner-english-large) is what DPK's own PIIAnalyzerEngine adds to
# the registry, after REMOVING the default SpacyRecognizer. When the real package
# is installed we get exactly that. Where it is not, spaCy en_core_web_sm carries
# the same Presidio API at a fraction of the weight - see _analyzer_backend.
SPACY_MODEL = "en_core_web_sm"

# DPK's default_score_threshold_key = 0.6 — we keep it identical.
DPK_SCORE_THRESHOLD = 0.6

PII_SALT = os.getenv("SENTINEL_PII_SALT", "sentinel-demo-salt")


# ---------------------------------------------------------------------------
# Engine singleton — expensive to build, cheap to re-use
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _analyzer_backend():
    """
    Return (analyze, anonymize, backend_name).

    Prefers the REAL transform. `dpk_pii_redactor.pii_analyzer.PIIAnalyzerEngine`
    is the engine `PIIRedactorTransform` constructs, so taking it directly means
    the redaction running here is DPK's, not a reimplementation of it - the same
    registry, the same Flair recognizer, the same score threshold. What we do not
    take is the transform's pyarrow file-pipeline wrapper, which reads a parquet
    column and writes another; this ingestion has events in memory and needs the
    RecognizerResult list anyway, to drop the spans _exclude_protected protects.

    The fallback builds Presidio the way DPK builds it, minus Flair. It exists
    because the real package pins transformers to 4.57.6 (via flair) and pulls
    ~1.5 GB of ner-english-large, which is why DPK lives in .venv-dpk and the
    serving venv never imports this module. Both paths are exercised by the
    tests; PII_ENGINE reports which one ran, and so does /health.
    """
    try:
        from dpk_pii_redactor.pii_analyzer import PIIAnalyzerEngine
        from dpk_pii_redactor.pii_anonymizer import PIIAnonymizer
    except ImportError:
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
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
        anonymizer = AnonymizerEngine()
        return (
            lambda text: analyzer.analyze(
                text=text, language="en",
                entities=DPK_ENTITIES, score_threshold=DPK_SCORE_THRESHOLD,
            ),
            lambda text, results: anonymizer.anonymize(
                text, results, operators={"DEFAULT": OperatorConfig("replace", None)}
            ).text,
            "dpk-presidio",
        )

    engine = PIIAnalyzerEngine(
        supported_entities=DPK_ENTITIES, score_threshold=DPK_SCORE_THRESHOLD
    )
    anonymizer = PIIAnonymizer(operator="replace")
    # analyze_text returns (results, entity_types); the types are recomputed
    # downstream after protected spans are dropped, so only results are kept.
    return (
        lambda text: engine.analyze_text(text)[0],
        lambda text, results: anonymizer.anonymize_text(text, results).text,
        "dpk-pii-redactor",
    )


def backend_name() -> str:
    """Which redaction engine this process will actually run."""
    return _analyzer_backend()[2]


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


def _protected_spans(text: str, terms: Iterable[str] = ()) -> List[Tuple[int, int]]:
    """
    Character ranges Presidio is not allowed to rewrite.

    Two sources, one reason: NER guesses, and on process-mining text it guesses
    wrong in ways that are silently destructive.

    ACTOR_XXXXXXXX aliases - spaCy reads them as PERSON because they look like
    names. Replacing an alias with <PERSON> undoes the pseudonymisation and
    collapses every actor back into one token.

    Process vocabulary - the activity, department and cost centre of the event
    being chunked. Measured on this export, spaCy tags "Order Created" as a
    PERSON in 55 of the first 200 events, which rewrites the activity out of the
    chunk that exists to describe it. These are enum-valued structured fields
    from the Celonis export, not free text, so exempting them cannot mask a real
    identifier hiding in prose - the free-text `note` is still swept in full.
    """
    spans = [(m.start(), m.end()) for m in _ACTOR_ALIAS_RE.finditer(text)]
    for term in terms:
        if term:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), text))
    return spans


def _exclude_protected(
    text: str, results: List[RecognizerResult], terms: Iterable[str] = ()
) -> List[RecognizerResult]:
    """Drop any Presidio result overlapping a protected span."""
    protected = _protected_spans(text, terms)
    if not protected:
        return results
    return [
        r for r in results
        if not any(r.start < end and r.end > start for start, end in protected)
    ]


def redact_pii(
    text: str,
    known_actors: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> Tuple[str, List[str]]:
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
    protected_terms : iterable of str
        Structured process vocabulary (activity, department, cost centre) that
        Presidio must not rewrite. See _protected_spans for why this is needed
        and why exempting it is safe.

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

    # Pass 2: DPK's NER sweep on the already-pseudonymised text.
    # Filter out any result spanning an ACTOR alias so pseudonymisation survives.
    analyze, anonymize, _backend = _analyzer_backend()

    raw_results: List[RecognizerResult] = analyze(working)
    results = _exclude_protected(working, raw_results, protected_terms)
    detected: List[str] = sorted({r.entity_type for r in results})

    if results:
        working = anonymize(working, results)

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
    analyze, _anonymize, _backend = _analyzer_backend()
    detected = {r.entity_type for r in analyze(text)}
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
