"""
PHASE 1 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Goal: prove the ETL methodology using IBM Data Prep Kit.

This script demonstrates the full DPK-integrated PII redaction pipeline
used in Phase 2 (phase2_ingestion_pipeline.py).

DPK integration
---------------
We use the same Presidio stack that powers DPK's dpk_pii_redactor transform:

    presidio_analyzer.AnalyzerEngine   — NER-based PII entity recognition
    presidio_anonymizer.AnonymizerEngine — entity replacement / redaction

The canonical DPK file-pipeline transform operates on PyArrow Tables:

    from dpk_pii_redactor.transform import PIIRedactorTransform
    from dpk_pii_redactor.transform import PIIRedactorTransformConfiguration
    from dpk_doc_chunk.transform_python import DocChunkTransform
    from dpk_doc_chunk.transform import DocChunkTransformConfiguration

For Sentinel's streaming event-log use-case we instantiate the underlying
analyzer/anonymizer directly via dpk_pii_engine.py — the same libraries and
the same Presidio engine — and add deterministic pseudonymisation so
process-structure (actor identity across events) is preserved for Granite's
causal-reasoning chains.

DPK entities recognised (mirrors DPK's default_supported_entities):
    PERSON, EMAIL_ADDRESS, PHONE_NUMBER, ORGANIZATION, CREDIT_CARD

Real DPK reference:
    pip install data-prep-toolkit
    pip install 'data-prep-toolkit-transforms[pii_redactor,doc_chunk]'
    Repo: https://github.com/IBM/data-prep-kit

Install for this project (venv):
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_sm
"""

import json
from pathlib import Path

from dpk_pii_engine import pseudonymise, redact_pii, analyze_entities

DATA_PATH = Path(__file__).parent / "data" / "mock_celonis_data.json"


def load_mock_events():
    with open(DATA_PATH, "r") as f:
        payload = json.load(f)
    return payload["events"], payload["summary_metrics"]


def main():
    events, summary = load_mock_events()
    print("=" * 70)
    print("PHASE 1 - DPK/Presidio PII Redaction Demo")
    print("Engine: presidio_analyzer + presidio_anonymizer (DPK stack)")
    print("=" * 70)

    actor_names = {e["resource"] for e in events if e.get("resource")}

    for e in events:
        raw = (
            f"Handled by {e['resource']} "
            f"({e['resource_email']}, {e['resource_phone']}). "
            f"Note: {e.get('note', '')}"
        ).strip(". ")

        clean, detected_entities = redact_pii(raw, known_actors=actor_names)

        print(f"\n[RAW  ] {raw}")
        print(f"[CLEAN] {clean}")
        print(f"[DPK  ] Entities detected by Presidio: {detected_entities}")
        print(f"[ALIAS] {e['resource']} → {pseudonymise(e['resource'])}")

    print("\n" + "-" * 70)
    print("Ground-truth summary metrics (the ONLY numbers Granite may")
    print("state as aggregates - anything else is a hallucination):")
    print(json.dumps(summary, indent=2))
    print("-" * 70)
    print("\nDPK entity types configured (mirrors dpk_pii_redactor defaults):")
    from dpk_pii_engine import DPK_ENTITIES
    for e in DPK_ENTITIES:
        print(f"  - {e}")


if __name__ == "__main__":
    main()
