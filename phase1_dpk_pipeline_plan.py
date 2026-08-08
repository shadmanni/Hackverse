"""
PHASE 1 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Goal for tonight's 8:30 PM eval: prove the ETL methodology is fully
planned and ready to execute, WITHOUT depending on a full IBM Data Prep
Kit (DPK) install (that install is heavy - conda, Python 3.11, gcc for
fasttext - too risky to do live during judging).

This script:
  1. Loads the mock Celonis export.
  2. Shows the exact DPK transforms we will wire in during Phase 2
     (pii_redactor, doc_chunk) with their real import paths.
  3. Runs OUR lightweight stand-in versions of those two transforms
     right now, so you can demo real before/after output tonight.

Real DPK reference (verified from IBM's own docs, for your slide):
  pip install data-prep-toolkit
  pip install 'data-prep-toolkit-transforms[pii_redactor,doc_chunk]'
  Repo: https://github.com/IBM/data-prep-kit
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "mock_celonis_data.json"

# ---------------------------------------------------------------------
# 1. THE REAL DPK IMPORTS WE WILL USE IN PHASE 2 (shown to judges as
#    proof of planning; commented out so this file runs with zero
#    installs tonight).
# ---------------------------------------------------------------------
#
# from dpk_pii_redactor.transform_python import PIIRedactorTransform
# from dpk_pii_redactor.transform import PIIRedactorTransformConfiguration
# from dpk_doc_chunk.transform_python import DocChunkTransform
# from dpk_doc_chunk.transform import DocChunkTransformConfiguration
#
# These ship inside the `data-prep-toolkit-transforms` package. In
# Phase 2 we point PIIRedactorTransformConfiguration at our event log
# fields (resource_email, resource_phone) and DocChunkTransformConfiguration
# at the free-text `note` fields so Granite only ever retrieves clean,
# right-sized chunks.


def load_mock_events():
    with open(DATA_PATH, "r") as f:
        payload = json.load(f)
    return payload["events"], payload["summary_metrics"]


# ---------------------------------------------------------------------
# 2. STAND-IN "pii_redactor" - same job the DPK transform does
#    (mask emails/phones), simple regex version for tonight's demo.
# ---------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
# Requires a leading '+' (our data uses +91-XXXXX-XXXXX). This is
# intentionally narrow so it never matches ISO timestamps like
# "2026-07-01T09:15:00Z" - broaden it only if you add phone formats
# without a country-code prefix.
PHONE_RE = re.compile(r"\+\d{1,3}[-\s]?\d{4,5}[-\s]?\d{4,5}")


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


# ---------------------------------------------------------------------
# 3. STAND-IN "doc_chunk" - turn each event into one semantic chunk
#    of ground-truth text, ready for embedding in Phase 2.
# ---------------------------------------------------------------------
def event_to_chunk(event: dict) -> str:
    chunk = (
        f"Case {event['case_id']}: {event['activity']} occurred on "
        f"{event['timestamp']} handled by {event['resource']} "
        f"({event['department']}, cost center {event['cost_center']}). "
        f"Amount: ${event['amount_usd']:,.2f}. "
        f"Cycle time so far: {event['cycle_time_days']} days."
    )
    if "note" in event:
        chunk += f" Note: {event['note']}"
    return redact_pii(chunk)


def main():
    events, summary = load_mock_events()
    print("=" * 70)
    print("PHASE 1 - ETL METHODOLOGY DEMO (mock data, no external installs)")
    print("=" * 70)

    for e in events:
        chunk = event_to_chunk(e)
        print(f"\n[RAW ]  resource_email={e['resource_email']}  phone={e['resource_phone']}")
        print(f"[CLEAN] {chunk}")

    print("\n" + "-" * 70)
    print("Ground-truth summary metrics (the ONLY numbers Granite may")
    print("state as aggregates - anything else is a hallucination):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()