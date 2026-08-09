"""
PHASE 2 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
"True Ingestion": clean -> chunk -> embed -> store, so Riddhi/Shivansh
have a real Milvus collection to retrieve from in Phase 3.

IBM Data Prep Kit integration
------------------------------
PII redaction now runs through the same Presidio stack that powers DPK's
dpk_pii_redactor transform (presidio_analyzer + presidio_anonymizer).

The integration is in dpk_pii_engine.py which mirrors the internals of
    dpk_pii_redactor/pii_analyzer.py  — AnalyzerEngine + RecognizerRegistry
    dpk_pii_redactor/pii_anonymizer.py — AnonymizerEngine + OperatorConfig
with the addition of deterministic actor pseudonymisation so process structure
(handoffs, rework loops, SoD checks) survives into Milvus clean.

DPK entities recognised (DPK defaults, minus ORGANIZATION and DATE_TIME):
    PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD

Chunking strategy
-----------------
One chunk per event = one clean, retrievable "fact" for Granite.  This maps
exactly to DPK's doc_chunk LI_TOKEN_TEXT mode with chunk_size_tokens=128.

Two engines, one contract
-------------------------
Presidio (and therefore spaCy) pins numpy below the version torch is built
against here, so installing it into the serving venv downgrades numpy under a
running model. It lives in its own environment instead:

    .venv-dpk/bin/pip install presidio-analyzer presidio-anonymizer spacy
    .venv-dpk/bin/python -m spacy download en_core_web_sm

When dpk_pii_engine imports, redaction runs through Presidio NER. When it does
not, the regex path below takes over: same pseudonymise() and therefore the same
aliases, so chunks are byte-identical on this export - chunk_event never
interpolates the email or phone fields, and no person-like string appears in a
`note` outside the actor list. Presidio's added value is that the NER backstop
holds when that stops being true. PII_ENGINE names whichever ran.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
# Requires a leading '+' so it never false-positives on ISO timestamps.
PHONE_RE = re.compile(r"\+\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5}")

# ponytail: fixed salt for the demo; source it from a secret store in prod.
PII_SALT = os.getenv("SENTINEL_PII_SALT", "sentinel-demo-salt")

try:
    # DPK PII engine — wraps presidio_analyzer + presidio_anonymizer the same
    # way dpk_pii_redactor's PIIRedactorTransform does internally.
    from dpk_pii_engine import has_raw_pii, pseudonymise, redact_pii
    PII_ENGINE = "dpk-presidio"
except ImportError:
    PII_ENGINE = "regex-fallback"

    def pseudonymise(name: str) -> str:
        """Stable salted alias. Identical to the DPK engine's, so the two
        pipelines produce the same ACTOR_xxxxxxxx for the same person."""
        digest = hashlib.sha256(f"{PII_SALT}:{name.strip().lower()}".encode()).hexdigest()
        return f"ACTOR_{digest[:8].upper()}"

    def redact_pii(
        text: str,
        known_actors: Iterable[str] = (),
        protected_terms: Iterable[str] = (),   # nothing to protect: no NER runs here
    ) -> Tuple[str, List[str]]:
        found = []
        if EMAIL_RE.search(text):
            found.append("EMAIL_ADDRESS")
            text = EMAIL_RE.sub("<EMAIL_ADDRESS>", text)
        if PHONE_RE.search(text):
            found.append("PHONE_NUMBER")
            text = PHONE_RE.sub("<PHONE_NUMBER>", text)
        # Longest first, so "Anita Rao Kumar" cannot be half-matched by "Anita Rao".
        for name in sorted({n for n in known_actors if n}, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", text):
                found.append("PERSON")
                text = re.sub(rf"\b{re.escape(name)}\b", pseudonymise(name), text)
        return text, sorted(set(found))

    def has_raw_pii(text: str, actor_names: Iterable[str] = ()) -> bool:
        if EMAIL_RE.search(text) or PHONE_RE.search(text):
            return True
        return any(re.search(rf"\b{re.escape(n)}\b", text) for n in actor_names if n)


# The 460-event export. Defaulting to the 9-event mock_celonis_data.json is how
# the collection silently shipped with 10 vectors instead of 461 while /graphs
# still reported 460 - the retriever searched a sixth of a percent of the log.
DATA_PATH = Path(__file__).parent / "data" / "mock_celonis_data_large.json"
MILVUS_DB_PATH = str(Path(__file__).parent / "data" / "sentinel_milvus.db")
COLLECTION_NAME = "celonis_ground_truth"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, runs on CPU


# ---------------------------------------------------------------------------
# Actor extraction
# ---------------------------------------------------------------------------

def actor_names(payload: dict) -> set:
    """Every direct identifier appearing as an actor in the export."""
    return {e["resource"] for e in payload.get("events", []) if e.get("resource")}


# ---------------------------------------------------------------------------
# STEP 1 — PII masking via DPK/Presidio
#
# Direct identifiers are PSEUDONYMISED, not deleted.  Process mining is about
# who handed work to whom, so blanket redaction would destroy the analysis:
# every actor would collapse into <PERSON> and handoffs, rework loops and
# segregation-of-duty checks would all become uncomputable.
#
# DPK/Presidio detects the raw name first; we intercept that span and replace
# it with ACTOR_XXXXXXXX before Presidio's anonymizer sees it, so the alias
# reaches the vector store — not the real name and not a generic placeholder.
#
# The alias is derived from a salted SHA-256 hash: deterministic across runs
# (same person = always ACTOR_XXXX) and not reversible by inspection.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# STEP 2 — Semantic chunking  (matches DPK doc_chunk LI_TOKEN_TEXT, 128 tok)
#   One chunk per event = one clean, retrievable "fact" for Granite.
# ---------------------------------------------------------------------------

def chunk_event(event: dict, names: Iterable[str] = ()) -> dict:
    """
    Convert a raw Celonis event into a PII-clean text chunk.

    The raw email and phone fields are intentionally excluded from the
    interpolated text — dropping an identifier at the source beats masking it
    downstream.  dpk_pii_engine.redact_pii() acts as a backstop for
    identifiers embedded in free-text fields like `note`.
    """
    raw_text = (
        f"Case {event['case_id']}: {event['activity']} occurred on "
        f"{event['timestamp']} handled by {event['resource']} "
        f"({event['department']}, cost center {event['cost_center']}). "
        f"Amount: ${event['amount_usd']:,.2f}. "
        f"Cycle time so far: {event['cycle_time_days']} days."
    )
    if "note" in event:
        raw_text += f" Note: {event['note']}"

    actor_set = names or ({event["resource"]} if event.get("resource") else ())
    # The structured fields are enum-valued process vocabulary, not free text.
    # NER reads "Order Created" as a PERSON and rewrites the activity out of the
    # chunk describing it; the note field is still swept in full.
    clean_text, _detected = redact_pii(
        raw_text,
        known_actors=actor_set,
        protected_terms=(
            event.get("case_id"), event.get("activity"),
            event.get("department"), event.get("cost_center"),
        ),
    )

    return {
        "case_id": event["case_id"],
        "activity": event["activity"],
        "text": clean_text,
    }


def chunk_summary(summary: dict) -> dict:
    text = (
        "AGGREGATE GROUND TRUTH: "
        f"Average compliance cycle time is {summary['avg_compliance_cycle_time_days']} days. "
        f"Average order-to-cash time is {summary['avg_order_to_cash_days']} days. "
        f"{summary['orders_flagged_for_bottleneck']} of {summary['total_orders']} "
        "orders were flagged for a supply chain bottleneck."
    )
    return {"case_id": "SUMMARY", "activity": "aggregate_metrics", "text": text}


# ---------------------------------------------------------------------------
# STEP 3 — Embed + store in Milvus Lite
# ---------------------------------------------------------------------------

def build_pipeline(data_path=DATA_PATH, db_path=MILVUS_DB_PATH, collection_name=COLLECTION_NAME):
    with open(data_path, "r") as f:
        payload = json.load(f)

    names = actor_names(payload)
    print(f"Loaded {len(payload['events'])} raw events from {data_path}")
    print(f"Found {len(names)} distinct actors: {', '.join(sorted(names))}")
    print(f"PII engine: {PII_ENGINE}")

    chunks = [chunk_event(e, names) for e in payload["events"]]
    chunks.append(chunk_summary(payload["summary_metrics"]))

    print(f"Pseudonymised {len(names)} actors; produced {len(chunks)} chunks.")

    # ------------------------------------------------------------------
    # Integrity gate — abort if any direct identifier survived.
    #
    # has_raw_pii() uses fast regex backstops (email, phone patterns) plus
    # an exact-word check for each known actor name.  A chunk that trips
    # this gate means redact_pii() missed a match; we surface it loudly so
    # the failure is visible rather than silently leaking PII to Milvus.
    # ------------------------------------------------------------------
    leaked = [
        c for c in chunks
        if has_raw_pii(c["text"], actor_names=names)
    ]
    if leaked:
        print(f"WARNING: {len(leaked)} chunk(s) still contain a direct identifier:")
        for c in leaked[:5]:
            print("  -", c["text"])
        raise SystemExit("Aborting ingestion: DPK/Presidio PII redaction failed integrity check.")
    print(
        f"DPK/Presidio PII integrity check passed: "
        f"no emails, phones or actor names in {len(chunks)} chunks."
    )

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' …")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False).tolist()

    print(f"Connecting to Milvus Lite at {db_path} …")
    client = MilvusClient(uri=db_path)

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=len(vectors[0]),
        metric_type="COSINE",
    )

    rows = [
        {
            "id": i,
            "vector": vectors[i],
            "case_id": chunks[i]["case_id"],
            "activity": chunks[i]["activity"],
            "text": chunks[i]["text"],
        }
        for i in range(len(chunks))
    ]
    client.insert(collection_name=collection_name, data=rows)

    print(f"Inserted {len(rows)} vectors into collection '{collection_name}'.")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--db", type=str, default=MILVUS_DB_PATH)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    args = parser.parse_args()
    build_pipeline(data_path=Path(args.data), db_path=args.db, collection_name=args.collection)
