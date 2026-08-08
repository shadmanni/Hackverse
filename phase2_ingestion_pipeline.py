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
import json
import re
from pathlib import Path

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
# ---------------------------------------------------------------------
def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


# ---------------------------------------------------------------------
# STEP 2: Semantic chunking  (stand-in for DPK's doc_chunk transform)
#   One chunk per event = one clean, retrievable "fact" for Granite.
# ---------------------------------------------------------------------
def chunk_event(event: dict) -> dict:
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
        "text": redact_pii(text),
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

    chunks = [chunk_event(e) for e in payload["events"]]
    chunks.append(chunk_summary(payload["summary_metrics"]))

    print(f"Loaded {len(payload['events'])} raw events from {data_path}")
    print(f"Prepared {len(chunks)} clean, PII-redacted chunks.")

    # Integrity check: abort if any raw PII survived into a chunk.
    leaked = [c for c in chunks if EMAIL_RE.search(c["text"]) or PHONE_RE.search(c["text"])]
    if leaked:
        print(f"WARNING: {len(leaked)} chunk(s) still contain raw PII after redaction:")
        for c in leaked[:5]:
            print("  -", c["text"])
        raise SystemExit("Aborting ingestion: PII redaction failed integrity check.")
    print("PII integrity check passed: no raw emails/phones in any chunk.")

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