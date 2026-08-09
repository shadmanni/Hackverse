"""
patching_engine.py — Sentinel-RAG Self-Healing Knowledge Patching Engine

Turns Sentinel-RAG from a PASSIVE guardrail into an ACTIVE system administrator.

When the circuit breaker trips, this module:
  1. Diagnoses whether the root cause is:
       - MISSING_KNOWLEDGE: retrieval similarity too low → entity not in vector DB
       - CONFABULATION_REASONING: model ignores good context → reasoning/parametric error
  2. Generates a structured JSON ticket for missing-knowledge events, including
     a suggested remediation action and an admin-approval hook for Milvus re-ingestion.
  3. Persists the ticket queue to disk (data/patch_tickets.json) so the
     Streamlit Admin Panel can display and act on it.
"""

from __future__ import annotations

import json
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Persistence path (same data/ directory as every other artefact)
# ---------------------------------------------------------------------------
TICKETS_PATH = Path(__file__).parent / "data" / "patch_tickets.json"
TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
KNOWLEDGE_GAP_THRESHOLD = 0.55   # top-chunk similarity below this → MISSING_KNOWLEDGE
CONFABULATION_THRESHOLD  = 0.75   # above this but intercepted → CONFABULATION_REASONING

# ---------------------------------------------------------------------------
# Document-type heuristics for remediation suggestions
# ---------------------------------------------------------------------------
_DOC_TYPE_HINTS: dict[str, str] = {
    "q4":           "Q4 financial forecast report",
    "forecast":     "forward-projection financial document",
    "override":     "vendor contract override authorisation log",
    "w-99":         "warehouse node W-99 operations record",
    "cc-9999":      "cost-centre CC-9999 expense log",
    "unapproved":   "executive authorisation audit trail",
    "vendor":       "vendor master data extract",
    "budget":       "budget approval documentation",
    "discount":     "discount authorisation sign-off record",
    "margin":       "margin reconciliation report",
    "price":        "price list or contract amendment",
}

# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------
_ENTITY_RE = re.compile(
    r"\b([A-Z]{2,}[-\w]*"          # all-caps codes: CASE-10231, SOX, Q4
    r"|[A-Z][a-z]+ [A-Z][a-z]+"    # proper names: Anita Rao
    r"|W-\d+|CC-\d+|PO-\d+"        # known code patterns
    r"|\$[\d,.M]+|\d+[\d.,]*%"     # monetary / percentage
    r"|\b[Qq][1-4]\b)\b"           # Q1-Q4
)

def _extract_entities(text: str) -> list[str]:
    """Pull candidate entity strings from a query for ticket labelling."""
    return list(dict.fromkeys(m.group() for m in _ENTITY_RE.finditer(text)))


def _suggest_doc_type(query: str, entities: list[str]) -> str:
    """Heuristically derive the most likely missing document type."""
    q_lower = query.lower()
    for kw, doc in _DOC_TYPE_HINTS.items():
        if kw in q_lower:
            return doc
    if entities:
        return f"{entities[0]}-related process record"
    return "Celonis process log or supporting document"


# ===========================================================================
# Gap Classifier
# ===========================================================================

def classify_interception(
    query: str,
    top_similarity: float,
    retrieved_chunk_count: int,
    entropy_score: float,
) -> str:
    """
    Returns:
      "MISSING_KNOWLEDGE"       – entity absent from the vector DB
      "CONFABULATION_REASONING" – model ignored available context
      "POISON_INJECTION"        – explicit out-of-scope keyword attack
    """
    if top_similarity < KNOWLEDGE_GAP_THRESHOLD or retrieved_chunk_count == 0:
        return "MISSING_KNOWLEDGE"

    if entropy_score > 2.5:
        return "POISON_INJECTION"

    return "CONFABULATION_REASONING"


# ===========================================================================
# Ticket Store (thin JSON file persistence)
# ===========================================================================

def _load_tickets() -> list[dict]:
    if TICKETS_PATH.exists():
        try:
            return json.loads(TICKETS_PATH.read_text())
        except Exception:
            pass
    return []


def _save_tickets(tickets: list[dict]) -> None:
    TICKETS_PATH.write_text(json.dumps(tickets, indent=2))


def get_all_tickets() -> list[dict]:
    """Return all persisted tickets (newest first)."""
    return list(reversed(_load_tickets()))


def get_pending_tickets() -> list[dict]:
    """Return only open / unapproved tickets."""
    return [t for t in get_all_tickets() if t.get("status") == "PENDING_APPROVAL"]


# ===========================================================================
# Ticket Generator
# ===========================================================================

def create_patch_ticket(
    query: str,
    gap_type: str,
    top_similarity: float,
    entropy_score: float,
    graph_key: str = "p2p",
    triggered_by_keyword: Optional[str] = None,
) -> Optional[dict]:
    """
    Creates a structured patch ticket for MISSING_KNOWLEDGE events.
    Returns None for CONFABULATION_REASONING / POISON_INJECTION (no ticket needed).
    Increments frequency counter if an identical gap has been seen before.
    """
    if gap_type != "MISSING_KNOWLEDGE":
        return None

    entities = _extract_entities(query)
    missing_entity = entities[0] if entities else (triggered_by_keyword or "unknown_entity")
    doc_type = _suggest_doc_type(query, entities)

    human_title = f"Missing {doc_type} covering '{missing_entity}'"
    remediation = (
        f"Ingest {doc_type} covering {missing_entity} into the "
        f"'{graph_key}' Milvus collection. Suggested source: Celonis EMS export "
        f"or SharePoint document library. Re-run embedding pipeline after upload."
    )

    tickets = _load_tickets()

    # Frequency deduplication: match on normalised query prefix (first 80 chars)
    query_key = query.strip().lower()[:80]
    existing = next((t for t in tickets if t.get("query_key") == query_key), None)

    if existing:
        existing["frequency"] = existing.get("frequency", 1) + 1
        existing["last_seen"] = _now()
        existing["status"] = "PENDING_APPROVAL"   # re-open if it was dismissed
        _save_tickets(tickets)
        return existing

    ticket: dict = {
        "ticket_id":      f"KP-{uuid.uuid4().hex[:8].upper()}",
        "status":         "PENDING_APPROVAL",
        "gap_type":       gap_type,
        "graph":          graph_key,
        "query":          query,
        "query_key":      query_key,
        "missing_entity": missing_entity,
        "title":          human_title,
        "doc_type":       doc_type,
        "top_similarity": round(top_similarity, 4),
        "entropy_score":  round(entropy_score, 4),
        "remediation":    remediation,
        "frequency":      1,
        "created_at":     _now(),
        "last_seen":      _now(),
        "approved_at":    None,
        "approved_by":    None,
    }

    tickets.append(ticket)
    _save_tickets(tickets)
    return ticket


# ===========================================================================
# Admin Approval (stub — triggers re-ingestion placeholder)
# ===========================================================================

def approve_ticket(ticket_id: str, approved_by: str = "admin") -> dict:
    """
    Marks a ticket as APPROVED.
    In a production deployment this would trigger a Milvus re-ingestion job.
    Returns the updated ticket dict, or raises KeyError if not found.
    """
    tickets = _load_tickets()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            t["status"] = "APPROVED"
            t["approved_at"] = _now()
            t["approved_by"] = approved_by
            _save_tickets(tickets)
            # ── stub: in production, fire off async Milvus re-ingestion here ──
            print(f"[PatchingEngine] Ticket {ticket_id} approved. "
                  f"Stub: would ingest '{t['doc_type']}' into Milvus collection '{t['graph']}'.")
            return t
    raise KeyError(f"Ticket {ticket_id} not found.")


def dismiss_ticket(ticket_id: str) -> dict:
    """Marks a ticket as DISMISSED (false-positive gap)."""
    tickets = _load_tickets()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            t["status"] = "DISMISSED"
            _save_tickets(tickets)
            return t
    raise KeyError(f"Ticket {ticket_id} not found.")


# ===========================================================================
# Helpers
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
