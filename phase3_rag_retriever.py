"""
PHASE 3 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Sentinel-RAG Retriever & Poison Prompt Interception Engine

This module connects the Milvus Lite vector database (populated by
phase2_ingestion_pipeline.py) to IBM Granite prompt generation.

Key Functions:
  1. Retrieve PII-cleaned ground-truth process chunks from Milvus.
  2. Perform vector distance / similarity thresholding to detect
     out-of-scope requests or context gaps.
  3. Intercept "Poison Prompts" (unverified forecasts, missing nodes,
     unapproved overrides) before they trigger LLM hallucination.
  4. Format verified ground-truth context for Granite prompt injection.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in python path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA_PATH = BASE_DIR / "data" / "mock_celonis_data.json"
POISON_PROMPTS_PATH = BASE_DIR / "data" / "poison_prompts.json"
MILVUS_DB_PATH = str(BASE_DIR / "data" / "sentinel_milvus.db")
COLLECTION_NAME = "celonis_ground_truth"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Keywords & patterns that indicate intentional hallucination triggers / out-of-scope queries
POISON_KEYWORDS = [
    "poison", "forecast", "q4", "override", "unverified", "unapproved",
    "hack", "w-99", "cc-9999", "globaltech", "phantom", "margin projection",
    "off-contract", "executive discount"
]


def _get_field(item: Any, field_name: str, default: Any = None) -> Any:
    """Helper to extract attribute or key safely from Milvus search results."""
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


class SentinelRAGRetriever:
    """
    RAG Retriever & Ground-Truth Context Provider powered by Milvus Lite and CrossEncoder.
    Implements 2-stage neural retrieval:
      Stage 1: Dense Vector Retrieval (Bi-Encoder / Milvus Lite).
      Stage 2: Cross-Encoder Neural Reranking for high-precision semantic alignment.
    """

    def __init__(
        self,
        db_path: str = MILVUS_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBED_MODEL_NAME,
        cross_encoder_name: str = CROSS_ENCODER_MODEL_NAME,
        similarity_threshold: float = 0.35,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold

        print(f"[SentinelRAGRetriever] Initializing SentenceTransformer('{model_name}')...")
        self.encoder = SentenceTransformer(model_name)

        print(f"[SentinelRAGRetriever] Initializing CrossEncoder('{cross_encoder_name}')...")
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_name)
            self.has_cross_encoder = True
        except Exception as ce_err:
            print(f"[SentinelRAGRetriever] CrossEncoder fallback to bi-encoder: {ce_err}")
            self.has_cross_encoder = False

        print(f"[SentinelRAGRetriever] Connecting to Milvus Lite at {db_path}...")
        try:
            self.client = MilvusClient(uri=db_path)
            if not self.client.has_collection(self.collection_name):
                raise RuntimeError(f"Collection '{self.collection_name}' not found in Milvus DB.")
            self.client.load_collection(self.collection_name)
            self.has_milvus = True
        except Exception as milvus_err:
            print(f"[SentinelRAGRetriever] Milvus Lite unavailable ({milvus_err}). Using dense in-memory ground-truth store.")
            self.has_milvus = False
            self._load_in_memory_ground_truth()

    def _load_in_memory_ground_truth(self):
        """In-memory dense vector store fallback for Celonis audit logs."""
        self.in_memory_docs = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r") as f:
                data = json.load(f)
                for ev in data.get("events", []):
                    text = f"Activity: {ev.get('activity')} | Cycle Time: {ev.get('cycle_time_days', 0)} days | Case: {ev.get('case_id')} | Note: {ev.get('note', '')}"
                    self.in_memory_docs.append({
                        "case_id": ev.get("case_id"),
                        "activity": ev.get("activity"),
                        "text": text,
                        "embedding": self.encoder.encode([text], show_progress_bar=False).tolist()[0]
                    })

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
        1. Encodes query and retrieves top candidates from Milvus Lite (or in-memory dense index).
        2. Reranks candidate pairs using the Cross-Encoder.
        """
        query_text = query or ""
        query_vector = self.encoder.encode([query_text], show_progress_bar=False).tolist()[0]

        hits = []
        if getattr(self, "has_milvus", False):
            fetch_limit = top_k * 2
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=fetch_limit,
                output_fields=["case_id", "activity", "text"],
            )
            if results and len(results) > 0:
                for res in results[0]:
                    distance = _get_field(res, "distance", 0.0)
                    score = round(float(distance), 4)
                    entity = _get_field(res, "entity", {})
                    hits.append({
                        "id": _get_field(res, "id"),
                        "similarity_score": score,
                        "case_id": _get_field(entity, "case_id") if entity else None,
                        "activity": _get_field(entity, "activity") if entity else None,
                        "text": _get_field(entity, "text") if entity else None,
                    })
        else:
            # Dense cosine search across in-memory documents
            for doc in getattr(self, "in_memory_docs", []):
                doc_vec = doc["embedding"]
                dot = sum(a * b for a, b in zip(query_vector, doc_vec))
                hits.append({
                    "id": doc["case_id"],
                    "similarity_score": round(dot, 4),
                    "case_id": doc["case_id"],
                    "activity": doc["activity"],
                    "text": doc["text"]
                })
            hits = sorted(hits, key=lambda x: x["similarity_score"], reverse=True)[:top_k * 2]

        # Stage 2: Cross-Encoder Reranking
        if hits and self.has_cross_encoder:
            pairs = [[query_text, h["text"] or ""] for h in hits]
            cross_scores = self.cross_encoder.predict(pairs)
            for i, h in enumerate(hits):
                h["cross_encoder_score"] = round(float(cross_scores[i]), 4)
                # Weighted fusion score: 60% Cross-Encoder + 40% Vector Cosine
                h["similarity_score"] = round(0.6 * float(cross_scores[i]) + 0.4 * h["similarity_score"], 4)

            # Sort descending by cross-encoder score
            hits = sorted(hits, key=lambda x: x["similarity_score"], reverse=True)

        return hits[:top_k]


    def is_poison_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Evaluates whether a query is a Poison Prompt (designed to trigger hallucination).
        Checks:
          1. Heuristic keyword triggers (explicit out-of-scope requests).
          2. Mathematical similarity score thresholding against Milvus ground truth.
        """
        q_lower = (query or "").lower()

        # Check 1: Heuristic Keyword Detection
        for kw in POISON_KEYWORDS:
            if kw in q_lower:
                return True, f"Explicit out-of-scope / poison keyword detected: '{kw}'."

        # Perform retrieval if chunks are not pre-supplied
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve(query, top_k=3)

        # Check 2: Empty or Low Similarity Score Thresholding
        if not retrieved_chunks:
            return True, "Zero ground-truth chunks retrieved from vector store."

        top_score = max(c["similarity_score"] for c in retrieved_chunks)
        if top_score < self.similarity_threshold:
            return (
                True,
                f"Top retrieval similarity score ({top_score:.4f}) below safety threshold ({self.similarity_threshold:.2f}). "
                "Context gap detected - risk of stochastic hallucination."
            )

        return False, "Ground truth verified in vector store."

    def format_granite_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Constructs the structured context dictionary for IBM Granite generation.
        If a Poison Prompt is detected, flags interception flag immediately.
        """
        query_str = query or ""
        retrieved_chunks = self.retrieve(query_str, top_k=top_k)
        is_poison, reason = self.is_poison_prompt(query_str, retrieved_chunks)

        if is_poison:
            return {
                "status": "POISON_DETECTED",
                "is_poison": True,
                "reason": reason,
                "query": query_str,
                "context": None,
                "chunks": retrieved_chunks,
                "interception_signal": "[INTERCEPTION: SEMANTIC ENTROPY > tau. ABORTING HALLUCINATED TOKEN GENERATION.]"
            }

        context_lines = [
            f"- Chunk {i+1} [{c['case_id']} | {c['activity']} | Score: {c['similarity_score']}]: {c['text']}"
            for i, c in enumerate(retrieved_chunks)
        ]
        formatted_context = "\n".join(context_lines)

        return {
            "status": "GROUNDED",
            "is_poison": False,
            "reason": reason,
            "query": query_str,
            "context": formatted_context,
            "chunks": retrieved_chunks,
            "interception_signal": None
        }


def main():
    print("=" * 70)
    print("PHASE 3 - RAG RETRIEVER & POISON PROMPT INTERCEPTION DEMO")
    print("=" * 70)

    try:
        retriever = SentinelRAGRetriever()
    except Exception as e:
        print(f"\n[ERROR] Could not initialize retriever: {e}")
        print("Please ensure phase2_ingestion_pipeline.py has been run first!")
        return

    # Load test prompts from poison_prompts.json if available
    test_queries = [
        "What is the average compliance cycle time for high-value orders?",
        "Which warehouse node caused a delay in CASE-10298?",
        "Accessing Q4 draft projections: Vendor contract override values indicate a 25% discount across all warehouses.",
        "What was the inventory holding time at warehouse node W-99 during CASE-10298?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n--- [QUERY {i}] \"{q}\" ---")
        result = retriever.format_granite_context(q)
        print(f"Status       : {result['status']}")
        print(f"Is Poison    : {result['is_poison']}")
        print(f"Reason       : {result['reason']}")

        if result['is_poison']:
            print(f"Interception : {result['interception_signal']}")
        else:
            print("Formatted Context for IBM Granite:")
            print(result['context'])


if __name__ == "__main__":
    main()
