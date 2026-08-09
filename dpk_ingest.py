"""
IBM Data Prep Kit PII redaction, as an OFFLINE pre-ingestion step.

Not yet wired in — this is the staged path for when DPK gets installed.

WHY A SEPARATE VENV
    data-prep-toolkit-transforms[pii_redactor] resolves transformers to 4.57.6
    and moves numpy. granite_runner.py runs on transformers 5.x, so installing
    DPK into .venv would break the model the whole product depends on
    (verified with `pip install --dry-run`). Keeping DPK in its own venv and
    running it offline means the runtime never imports it: this script reads the
    raw export and writes a redacted one, and the pipeline just reads that file.

SETUP (one time, ~5-10 min of downloads)
    python3 -m venv .venv-dpk
    .venv-dpk/bin/pip install "data-prep-toolkit-transforms[pii_redactor]"

RUN
    .venv-dpk/bin/python dpk_ingest.py \
        --in data/mock_celonis_data_large.json \
        --out data/celonis_dpk_redacted.json

    Then point phase2_ingestion_pipeline.py at the output:
    python phase2_ingestion_pipeline.py --data data/celonis_dpk_redacted.json

WHAT THIS ADDS OVER phase2_ingestion_pipeline.py
    Ours masks the known `resource` field plus email/phone regexes. That covers
    structured fields but cannot see identifiers sitting in free text. DPK's
    redactor is Presidio-backed NER, so it catches names in `note` fields that
    field-level masking structurally misses. The two are complementary: keep
    the field-level pseudonymisation (it is deterministic and preserves actor
    identity for handoff analysis) and use DPK to sweep the free text.
"""

import argparse
import json
import sys
from pathlib import Path

# Free-text fields worth sweeping. Structured identifiers are already handled
# deterministically upstream; NER on those would only add nondeterminism.
FREE_TEXT_FIELDS = ("note", "comment", "description")

SETUP_HINT = """
IBM Data Prep Kit is not importable in this interpreter.

Run this script with the DPK venv, not the runtime venv:

    python3 -m venv .venv-dpk
    .venv-dpk/bin/pip install "data-prep-toolkit-transforms[pii_redactor]"
    .venv-dpk/bin/python dpk_ingest.py --in <raw.json> --out <redacted.json>

Do NOT install DPK into .venv: it downgrades transformers to 4.57.6 and would
break granite_runner.py.
""".strip()


def load_redactor():
    """Import DPK lazily so the module stays importable without it installed."""
    try:
        from dpk_pii_redactor.transform_python import PIIRedactorTransform
    except ImportError:
        print(SETUP_HINT, file=sys.stderr)
        raise SystemExit(2)
    return PIIRedactorTransform(
        {"doc_transformer_params": {"supported_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]}}
    )


def redact_events(events, redactor):
    changed = 0
    for ev in events:
        for field in FREE_TEXT_FIELDS:
            original = ev.get(field)
            if not original:
                continue
            out, _ = redactor.transform_text(original)
            if out != original:
                ev[field] = out
                changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.src).read_text())
    events = payload.get("events", [])
    redactor = load_redactor()

    changed = redact_events(events, redactor)
    Path(args.dst).write_text(json.dumps(payload, indent=2))

    print(f"DPK pii_redactor: swept {len(events)} events, rewrote {changed} free-text field(s)")
    print(f"Wrote {args.dst}")
    print("Next: python phase2_ingestion_pipeline.py --data " + args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
