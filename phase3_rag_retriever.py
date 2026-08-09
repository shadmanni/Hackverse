"""
PHASE 3 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Sentinel-RAG Retriever & Poison Prompt Interception Engine

This module connects the Milvus Lite vector database (populated by
phase2_ingestion_pipeline.py) to IBM Granite prompt generation.

What this layer does:
  1. Retrieve PII-cleaned ground-truth process chunks from Milvus Lite.
  2. Rerank them with a CrossEncoder and report how well-supported retrieval was.
  3. Format supported chunks as Granite context; withhold them when they are noise.
  4. Export the measurement showing that retrieval score cannot decide
     answerability, which is why detection happens during generation.

What it does NOT do: decide whether a query will produce a hallucination. It
used to claim that, via a keyword blocklist. See the note above DATA_PATH.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in python path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# The 460-event export, matching what is embedded in Milvus and what
# celonis_metrics derives its aggregates from. Defaulting to the 9-event
# mock_celonis_data.json silently shrinks retrieval to a sixth of a percent of
# the log while /graphs still reports 460.
DATA_PATH = BASE_DIR / "data" / "mock_celonis_data_large.json"
POISON_PROMPTS_PATH = BASE_DIR / "data" / "poison_prompts.json"
MILVUS_DB_PATH = str(BASE_DIR / "data" / "sentinel_milvus.db")
EVAL_REPORT_PATH = BASE_DIR / "data" / "phase3_evaluation_report.json"
COLLECTION_NAME = "celonis_ground_truth"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# There is deliberately no keyword list here any more.
#
# What used to sit at this line was POISON_KEYWORDS - "q4", "w-99", "globaltech",
# "off-contract" and ten others - substring-matched against the query. It made
# the headline detection number a blocklist scored against a fixture written to
# contain those exact strings, and it would not have transferred one row to a
# different customer's event log.
#
# Removing it raised the obvious question: can the retrieval score alone decide
# whether the store can answer a query? Measured over both classes, no. The two
# distributions overlap almost completely:
#
#   "throughput delay and inventory holding at warehouse node X"   0.8645  (unanswerable)
#   "How many cases are in the event log?"                         0.1384  (answerable)
#
# Embedding similarity measures TOPICAL RELATEDNESS, not answerability. A
# fabricated warehouse question retrieves real warehouse events and scores high;
# a legitimate corpus-level count resembles no individual event chunk and scores
# low. Best case at any threshold was 2 of 9 real questions falsely flagged.
#
# So this layer no longer returns a hallucination verdict at all. It reports how
# well retrieval is supported, which is a real and useful signal for deciding
# whether to put the retrieved chunks in front of the model - and nothing more.
# The detection that has to hold is in SentinelStream, during generation, where
# it works on the model's own distribution and on the numbers it actually emits.
# Both of those are derived from the data and transfer to any event log.


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
        """
        In-memory dense store, used only when Milvus fails to open.

        Chunks are built by the ingestion pipeline's own chunk_event, not by a
        second formatter here. The previous version wrote its own sentence
        ("Activity: X | Cycle Time: N days | ..."), which broke the fallback two
        ways: the wording differed from what was embedded into Milvus, so
        similarity_threshold was comparing against a distribution nobody
        calibrated, and it interpolated event["resource"] raw - putting real
        actor names into the retrieval context on exactly the path that runs
        when the vector store is unavailable. chunk_event applies the PII
        redaction, so the fallback cannot drift from the real store or leak.
        """
        from phase2_ingestion_pipeline import actor_names, chunk_event

        self.in_memory_docs = []
        if not DATA_PATH.exists():
            return
        payload = json.loads(DATA_PATH.read_text())
        names = actor_names(payload)
        for ev in payload.get("events", []):
            doc = chunk_event(ev, names)
            doc["embedding"] = self.encoder.encode(
                [doc["text"]], show_progress_bar=False
            ).tolist()[0]
            self.in_memory_docs.append(doc)

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
                raw_logit = float(cross_scores[i])
                # ms-marco CrossEncoder emits an unbounded relevance LOGIT (~ -11..+8),
                # not a similarity. Fusing it raw with a 0..1 cosine made the fused
                # score unbounded, so similarity_threshold compared against nothing
                # meaningful and grounded queries were intercepted as poison.
                # Squash to 0..1 first so both terms share a scale.
                cross_prob = 1.0 / (1.0 + math.exp(-raw_logit))
                h["cross_encoder_logit"] = round(raw_logit, 4)
                h["cross_encoder_score"] = round(cross_prob, 4)
                # Weighted fusion score: 60% Cross-Encoder + 40% Vector Cosine
                h["similarity_score"] = round(0.6 * cross_prob + 0.4 * h["similarity_score"], 4)

            # Sort descending by fused score
            hits = sorted(hits, key=lambda x: x["similarity_score"], reverse=True)

        return hits[:top_k]


    def retrieval_support(self, query: str, retrieved_chunks: List[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Is retrieval well-supported enough that the chunks are worth showing the model?

        Returns (weakly_supported, reason). This is a CONTEXT-QUALITY decision,
        not a hallucination verdict, and the difference matters: a weakly
        supported query is not necessarily unanswerable, and a strongly
        supported one is not necessarily answerable.

        Both directions were measured on this log and both occur. "How many
        cases are in the event log?" scores 0.1384 - answerable from the
        aggregate block, but no individual event chunk resembles a corpus-level
        count. "What was the throughput delay and inventory holding time at
        warehouse node X" scores 0.8645 - it pulls real warehouse events, and
        the log holds neither figure. So a low score means "these chunks add
        noise, send the aggregates alone", and nothing more may be concluded
        from it. Deciding whether the ANSWER is grounded is SentinelStream's
        job, made per token against the model's own output.
        """
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve(query, top_k=3)

        if not retrieved_chunks:
            return True, "No chunks retrieved; answering from verified aggregates only."

        top_score = max(c["similarity_score"] for c in retrieved_chunks)
        if top_score < self.similarity_threshold:
            return (
                True,
                f"Top retrieval score {top_score:.4f} below {self.similarity_threshold:.2f}; "
                "chunks withheld as noise, answering from verified aggregates only."
            )
        return False, f"Retrieval supported (top score {top_score:.4f})."

    def format_granite_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Constructs the structured context dictionary for IBM Granite generation.
        If a Poison Prompt is detected, flags interception flag immediately.
        """
        query_str = query or ""
        retrieved_chunks = self.retrieve(query_str, top_k=top_k)
        weak, reason = self.retrieval_support(query_str, retrieved_chunks)
        top_score = max((c["similarity_score"] for c in retrieved_chunks), default=0.0)

        if weak:
            # Chunks withheld, NOT an interception. The caller falls back to the
            # verified aggregate block, which answers corpus-level questions that
            # no single event chunk resembles. This used to return
            # status=POISON_DETECTED with a pre-written interception banner, so a
            # low similarity score was reported to the UI as a hallucination
            # caught before a single token existed.
            return {
                "status": "AGGREGATES_ONLY",
                "retrieval_supported": False,
                "top_score": top_score,
                "reason": reason,
                "query": query_str,
                "context": None,
                "chunks": retrieved_chunks,
            }

        # The similarity score is deliberately NOT interpolated into the context.
        # It is a number that is not in the event log, sitting in text the model
        # is told to quote figures from; the grounding layer would then have to
        # reject the model for repeating something this file put in front of it.
        context_lines = [
            f"- Chunk {i+1} [{c['case_id']} | {c['activity']}]: {c['text']}"
            for i, c in enumerate(retrieved_chunks)
        ]

        return {
            "status": "RETRIEVAL_SUPPORTED",
            "retrieval_supported": True,
            "top_score": top_score,
            "reason": reason,
            "query": query_str,
            "context": "\n".join(context_lines),
            "chunks": retrieved_chunks,
        }

    def evaluate_test_suite(self, suite_file: Path = POISON_PROMPTS_PATH) -> Dict[str, Any]:
        """
        Measures whether retrieval score can decide answerability. It cannot.

        This suite used to report an interception count. It no longer reports
        one, because the layer it exercises no longer makes that claim: every
        flag it used to count came from a hardcoded keyword list matched against
        a fixture authored to contain those keywords.

        What it measures now is the thing that decided the architecture - the
        score distributions of answerable and unanswerable prompts, and how far
        they overlap. The overlap is the evidence for doing detection during
        generation rather than before it, so it is worth exporting rather than
        asserting.
        """
        if not suite_file.exists():
            print(f"[SentinelRAGRetriever] Warning: Suite file {suite_file} not found.")
            return {}

        suite_data = json.loads(suite_file.read_text())

        def score(prompt: str) -> float:
            hits = self.retrieve(prompt, top_k=3)
            return max((h["similarity_score"] for h in hits), default=0.0)

        unanswerable = [
            {"id": i.get("id"), "prompt": i["prompt"], "top_score": round(score(i["prompt"]), 4)}
            for i in suite_data.get("poison_prompts", [])
        ]
        answerable = [
            {"id": i.get("id"), "prompt": i["prompt"], "top_score": round(score(i["prompt"]), 4)}
            for i in suite_data.get("grounded_prompts", [])
        ]

        un = [r["top_score"] for r in unanswerable]
        an = [r["top_score"] for r in answerable]
        # Positive means a threshold exists that separates the classes.
        gap = (min(an) - max(un)) if (an and un) else 0.0

        sweep = []
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            sweep.append({
                "threshold": t,
                "answerable_wrongly_flagged": sum(1 for v in an if v < t),
                "unanswerable_flagged": sum(1 for v in un if v < t),
            })

        report = {
            "measures": "whether retrieval similarity can decide answerability, "
                        "before any token is generated",
            # Milvus Lite is single-process: running this while the backend holds
            # the lock silently falls back to the in-memory index, and the scores
            # below are then not the ones the product serves.
            "vector_store": "milvus_lite" if getattr(self, "has_milvus", False) else "in_memory_fallback",
            "does_not_measure": [
                "semantic entropy interception",
                "numeric grounding",
                "autonomous recovery",
            ],
            "finding": (
                "Embedding similarity measures topical relatedness, not "
                "answerability. Unanswerable prompts about real entities retrieve "
                "real chunks and score high; answerable corpus-level questions "
                "resemble no single event chunk and score low. No threshold "
                "separates the two classes, so this layer only decides whether "
                "retrieved chunks are worth showing the model."
                if gap <= 0 else
                "The classes separate on this fixture. Small n - do not "
                "generalise from it to a deployed threshold."
            ),
            "summary": {
                "unanswerable_tested": len(un),
                "answerable_tested": len(an),
                "unanswerable_score_range": [min(un), max(un)] if un else None,
                "answerable_score_range": [min(an), max(an)] if an else None,
                "separation_gap": round(gap, 4),
                "separable_by_any_threshold": gap > 0,
                "active_threshold": self.similarity_threshold,
                "threshold_sweep": sweep,
            },
            "unanswerable_details": unanswerable,
            "answerable_details": answerable,
        }

        EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(f"[SentinelRAGRetriever] Retrieval-separation report saved to {EVAL_REPORT_PATH}")
        return report


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

    # Run full Phase 3 evaluation benchmark suite
    print("\nExecuting Phase 3 Poison Prompt & Grounded Retrieval Evaluation Suite...")
    report = retriever.evaluate_test_suite()

    summary = report.get("summary", {})
    print("\n" + "=" * 70)
    print("PHASE 3 BENCHMARK REPORT SUMMARY")
    print("=" * 70)
    print(f"Measures                  : {report.get('measures')}")
    print(f"Poison prompts tested     : {summary.get('poison_prompts_tested')}")
    print(f"Poison prompts flagged    : {summary.get('poison_prompts_flagged')}")
    print(f"  by mechanism            : {summary.get('poison_flags_by_mechanism')}")
    print(f"Grounded prompts tested   : {summary.get('grounded_prompts_tested')}")
    print(f"Grounded prompts passed   : {summary.get('grounded_prompts_passed')}")
    print("-" * 70)
    print(summary.get("caveat", ""))
    print("=" * 70)


if __name__ == "__main__":
    main()
