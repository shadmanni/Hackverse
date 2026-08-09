"""
PHASE 2 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
"True Ingestion": clean -> chunk -> embed -> store, so Riddhi/Shivansh
have a real Milvus collection to retrieve from in Phase 3.

Install (run once):
    pip install pymilvus sentence-transformers

We use MILVUS LITE - an embedded, file-based Milvus that needs no
Docker/server. This is the officially supported "quick start" mode of
pymilvus and is the right choice for a 24-hour hackathon.

If you get real DPK (data-prep-toolkit-transforms) installed later,
swap `redact_pii()` and `chunk_event()` below for:
    from dpk_pii_redactor.transform_python import PIIRedactorTransform
    from dpk_doc_chunk.transform_python import DocChunkTransform
The rest of this file (embedding + Milvus) stays identical - that's
the point of keeping ingestion, chunking, and storage as separate
functions.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent / "data" / "mock_celonis_data.json"
MILVUS_DB_PATH = str(Path(__file__).parent / "data" / "sentinel_milvus.db")
COLLECTION_NAME = "celonis_ground_truth"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, runs on CPU

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
# Requires a leading '+' so it never false-positives on ISO timestamps.
PHONE_RE = re.compile(r"\+\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5}")


# ---------------------------------------------------------------------
# STEP 1: PII masking  (stand-in for DPK's pii_redactor transform)
#
# Direct identifiers are PSEUDONYMISED, not deleted. Process mining is about
# who handed work to whom, so blanket redaction would destroy the analysis it
# exists to support: every actor would collapse into [REDACTED] and handoffs,
# rework loops and segregation-of-duty checks would all become uncomputable.
# A stable per-actor alias keeps those relationships intact while removing the
# identity, which is what enterprise de-identification actually calls for.
#
# The alias is derived from a salted hash so it is deterministic across runs
# (the same person is always ACTOR_xxxx) but not reversible by inspection. The
# salt is per-deployment; leaving it at the default means aliases are stable
# but guessable by anyone with the name list, which is fine for synthetic demo
# data and NOT sufficient for a real export.
# ponytail: fixed salt for the demo; source it from a secret store in prod.
PII_SALT = os.getenv("SENTINEL_PII_SALT", "sentinel-demo-salt")


def pseudonymise(name: str) -> str:
    digest = hashlib.sha256(f"{PII_SALT}:{name.strip().lower()}".encode()).hexdigest()
    return f"ACTOR_{digest[:8].upper()}"


def redact_pii(text: str, names: Iterable[str] = ()) -> str:
    """Mask emails and phone numbers, and pseudonymise any known actor name."""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    # Longest first, so "Anita Rao Kumar" cannot be half-matched by "Anita Rao".
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", pseudonymise(name), text)
    return text


def actor_names(payload: dict) -> set:
    """Every direct identifier appearing as an actor in the export."""
    return {e["resource"] for e in payload.get("events", []) if e.get("resource")}


# ---------------------------------------------------------------------
# STEP 2: Semantic chunking  (stand-in for DPK's doc_chunk transform)
#   One chunk per event = one clean, retrievable "fact" for Granite.
# ---------------------------------------------------------------------
def chunk_event(event: dict, names: Iterable[str] = ()) -> dict:
    # The raw email and phone fields are never interpolated here: dropping an
    # identifier at the source beats masking it downstream. The regexes remain
    # as a backstop for identifiers embedded in free-text fields like `note`.
    text = (
        f"Case {event['case_id']}: {event['activity']} occurred on "
        f"{event['timestamp']} handled by {event['resource']} "
        f"({event['department']}, cost center {event['cost_center']}). "
        f"Amount: ${event['amount_usd']:,.2f}. "
        f"Cycle time so far: {event['cycle_time_days']} days."
    )
    if "note" in event:
        text += f" Note: {event['note']}"
    return {
        "case_id": event["case_id"],
        "activity": event["activity"],
        "text": redact_pii(text, names or ({event["resource"]} if event.get("resource") else ())),
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


# ---------------------------------------------------------------------
# STEP 3: Embed + store in Milvus Lite
# ---------------------------------------------------------------------
def build_pipeline(data_path=DATA_PATH, db_path=MILVUS_DB_PATH, collection_name=COLLECTION_NAME):
    with open(data_path, "r") as f:
        payload = json.load(f)

    names = actor_names(payload)
    chunks = [chunk_event(e, names) for e in payload["events"]]
    chunks.append(chunk_summary(payload["summary_metrics"]))

    print(f"Loaded {len(payload['events'])} raw events from {data_path}")
    print(f"Pseudonymised {len(names)} distinct actors; prepared {len(chunks)} chunks.")

    # Integrity check: abort if any direct identifier survived into a chunk.
    #
    # This previously scanned only for emails and phone numbers, neither of
    # which chunk_event ever wrote into the text - so the check could not fail
    # and "integrity passed" meant nothing, while every actor's real name went
    # into the vector store in clear. Names are now the primary assertion.
    leaked = [
        c for c in chunks
        if EMAIL_RE.search(c["text"])
        or PHONE_RE.search(c["text"])
        or any(re.search(rf"\b{re.escape(n)}\b", c["text"]) for n in names)
    ]
    if leaked:
        print(f"WARNING: {len(leaked)} chunk(s) still contain a direct identifier:")
        for c in leaked[:5]:
            print("  -", c["text"])
        raise SystemExit("Aborting ingestion: PII redaction failed integrity check.")
    print(f"PII integrity check passed: no emails, phones or actor names in {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False).tolist()

    print(f"Connecting to Milvus Lite at {db_path} ...")
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