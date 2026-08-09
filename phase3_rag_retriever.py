"""
PHASE 3 DELIVERABLE - Shadman (Data Pipeline & Ingestion Lead)
================================================================
Sentinel-RAG Retriever & Poison Prompt Interception Engine

This module connects the Milvus Lite vector database (populated by
phase2_ingestion_pipeline.py) to IBM Granite prompt generation.

What this layer does:
  1. Retrieve PII-cleaned ground-truth process chunks from Milvus Lite (dense)
     and from an in-process BM25 index over the same chunks (sparse), and fuse
     the two candidate lists by Reciprocal Rank Fusion.
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
import re
from collections import Counter, defaultdict
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


# Hyphens and dots are kept INSIDE a token, so "CASE-10101", "CC-3305" and
# "W-99" survive whole. Splitting on them would reduce an id query to the term
# "case", which all 461 chunks contain and which therefore carries no idf - and
# exact identifier matching is the only reason the sparse arm exists.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _BM25:
    """
    Okapi BM25 over the same chunks that are in Milvus.

    Written out instead of importing rank_bm25: it is twenty lines, and a
    dependency that has to pip-install successfully is a demo that can fail on a
    conference wifi. The corpus is 461 short chunks, so the postings list is
    free to hold in memory and there is nothing to persist.
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [_tokenize(t) for t in corpus]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 1.0
        self.postings = defaultdict(list)
        for i, counts in enumerate(Counter(d) for d in docs):
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self.postings.items()
        }

    def top(self, query: str, limit: int) -> List[int]:
        """Corpus indices of the best-scoring documents, best first."""
        scores = defaultdict(float)
        for term in set(_tokenize(query)):
            for i, freq in self.postings.get(term, ()):
                norm = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += self.idf[term] * freq * (self.k1 + 1) / norm
        return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:limit]]


def ground_truth_chunks() -> List[Dict[str, Any]]:
    """
    The chunk list ingestion put in Milvus, rebuilt in Milvus id order.

    Chunks come from the ingestion pipeline's own chunk_event/chunk_summary, not
    from a second formatter here. A local formatter existed once and broke two
    ways: its wording differed from what was embedded, so similarity_threshold
    compared against a distribution nobody calibrated, and it interpolated
    event["resource"] raw - putting real actor names into retrieval context.

    Position in this list IS the Milvus primary key (build_pipeline inserts
    id=i over the same file in the same order), which is what lets the sparse
    arm and the dense arm be deduplicated by id rather than by text.
    """
    from phase2_ingestion_pipeline import actor_names, chunk_event, chunk_summary

    if not DATA_PATH.exists():
        return []
    payload = json.loads(DATA_PATH.read_text())
    names = actor_names(payload)
    chunks = [chunk_event(ev, names) for ev in payload.get("events", [])]
    if "summary_metrics" in payload:
        chunks.append(chunk_summary(payload["summary_metrics"]))
    return chunks


# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# and is what every implementation uses; it damps the head of each list so a
# single arm's top hit cannot dominate a document both arms agree on.
RRF_K = 60


class SentinelRAGRetriever:
    """
    RAG Retriever & Ground-Truth Context Provider powered by Milvus Lite and CrossEncoder.
    Implements 3-stage hybrid retrieval:
      Stage 1: Dense Vector Retrieval (Bi-Encoder / Milvus Lite) + sparse BM25.
      Stage 2: Reciprocal Rank Fusion of the two candidate lists.
      Stage 3: Cross-Encoder Neural Reranking for high-precision semantic alignment.
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
        self._corpus = None   # shared by the sparse arm and the in-memory fallback
        self._bm25 = None

        print(f"[SentinelRAGRetriever] Initializing SentenceTransformer('{model_name}')...")
        self.encoder = SentenceTransformer(model_name)

        print(f"[SentinelRAGRetriever] Initializing CrossEncoder('{cross_encoder_name}')...")
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_name)
            self.has_cross_encoder = True
        except Exception as ce_err:
            print(f"[SentinelRAGRetriever] CrossEncoder fallback to bi-encoder: {ce_err}")
            self.has_cross_encoder = False

        # Reported to /health and the UI. Derived from the flag rather than
        # hardcoded, so a failed CrossEncoder load cannot make /health claim a
        # rerank stage that did not run.
        self.retrieval_mode = "hybrid_bm25_dense" + ("_ce" if self.has_cross_encoder else "")

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

    def _corpus_chunks(self) -> List[Dict[str, Any]]:
        """The ingested chunks, loaded once and shared by both retrieval arms."""
        if self._corpus is None:
            self._corpus = ground_truth_chunks()
        return self._corpus

    def _sparse_index(self) -> "_BM25":
        """
        BM25 over the ingested chunks, built on first query (~30ms for 461).

        Deferred rather than built in __init__ because constructing a retriever
        is on the API server's startup path, and re-chunking the export there
        would add to a warm-up that already costs ~35s.
        """
        if self._bm25 is None:
            self._bm25 = _BM25([c["text"] for c in self._corpus_chunks()])
        return self._bm25

    def _load_in_memory_ground_truth(self):
        """
        In-memory dense store, used only when Milvus fails to open.

        Chunks come from ground_truth_chunks() - the ingestion pipeline's own
        formatter - so the fallback cannot drift from the real store or leak an
        actor name. See that function for what a second formatter cost.
        """
        self.in_memory_docs = self._corpus_chunks()
        if not self.in_memory_docs:
            return
        vectors = self.encoder.encode(
            [d["text"] for d in self.in_memory_docs], show_progress_bar=False
        ).tolist()
        for doc, vec in zip(self.in_memory_docs, vectors):
            doc["embedding"] = vec

    def _sparse_hits(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """
        BM25 candidates, in the same dict shape the dense arm produces.

        similarity_score is left None: these documents have no cosine yet
        because the dense arm did not return them, and _fuse fills it in.
        """
        corpus = self._corpus_chunks()
        return [
            {
                "id": i,   # corpus position == Milvus primary key, see ground_truth_chunks
                "similarity_score": None,
                "case_id": corpus[i]["case_id"],
                "activity": corpus[i]["activity"],
                "text": corpus[i]["text"],
            }
            for i in self._sparse_index().top(query_text, limit)
        ]

    def _fuse(self, arms: List[List[Dict[str, Any]]], query_vector: List[float]) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion of the dense and sparse candidate lists.

        The two arms score on incomparable scales - cosine is 0..1, BM25 is
        unbounded and corpus-dependent - and this file has already eaten one bug
        from fusing two such scales directly (see cross_prob below). RRF reads
        only the rank, so there is nothing to calibrate and nothing to re-tune
        when the corpus changes.
        """
        fused: Dict[Any, Dict[str, Any]] = {}
        for arm in arms:
            for rank, hit in enumerate(arm, start=1):
                entry = fused.setdefault(hit["id"], hit)
                entry["rrf_score"] = round(entry.get("rrf_score", 0.0) + 1.0 / (RRF_K + rank), 6)

        sparse_only = [h for h in fused.values() if h["similarity_score"] is None]
        if sparse_only:
            # all-MiniLM's sentence-transformers config ends in a Normalize
            # module, so a plain dot product IS the cosine Milvus reports for
            # the dense arm. Both terms of the final fusion stay on one scale,
            # which is what similarity_threshold=0.35 was calibrated against.
            # Floored at 0: the dense arm only ever returned the top few cosines,
            # which are positive, but BM25 hands over any document sharing a
            # term, and a document that shares "47" with "recipe for lasagna 47"
            # cosines NEGATIVE. That drove the fused score below zero on 29 of
            # 900 adversarial queries, breaking the 0..1 scale the reported
            # similarity and similarity_threshold both assume. Below orthogonal
            # is not more informative than orthogonal, so clamping loses nothing.
            vectors = self.encoder.encode([h["text"] for h in sparse_only], show_progress_bar=False)
            for hit, vec in zip(sparse_only, vectors):
                dot = sum(a * b for a, b in zip(query_vector, vec))
                hit["similarity_score"] = round(max(0.0, float(dot)), 4)

        return sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Three-stage hybrid retrieval:
        1. Dense candidates from Milvus Lite (or the in-memory dense index) and
           sparse BM25 candidates over the same chunks.
        2. Reciprocal Rank Fusion of the two lists.
        3. Cross-Encoder rerank of the fused pool.

        The sparse arm is what makes an identifier query work. Dense-only,
        "CASE-10101" returned CASE-10119/10121/10150 and never the case asked
        for: an embedding has no reason to separate one case id from another,
        and a wrong-case chunk in the context is exactly what the model
        fabricates from.
        """
        query_text = query or ""
        query_vector = self.encoder.encode([query_text], show_progress_bar=False).tolist()[0]
        # Deeper than the old top_k*2: RRF only helps if each arm reaches far
        # enough down to find the documents the other arm ranked highly.
        fetch_limit = top_k * 4
        hits = []
        if getattr(self, "has_milvus", False):
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
            # Dense cosine search across in-memory documents. The id is the
            # corpus position, not the case_id: several chunks share a case_id,
            # and the fusion below deduplicates the two arms by id.
            for i, doc in enumerate(getattr(self, "in_memory_docs", [])):
                dot = sum(a * b for a, b in zip(query_vector, doc["embedding"]))
                hits.append({
                    "id": i,
                    "similarity_score": round(dot, 4),
                    "case_id": doc["case_id"],
                    "activity": doc["activity"],
                    "text": doc["text"]
                })
            hits = sorted(hits, key=lambda x: x["similarity_score"], reverse=True)[:fetch_limit]

        # Stage 2: fuse the dense list with the sparse one, then cap the pool the
        # CrossEncoder has to score - it is the expensive stage, and RRF has
        # already put the documents both arms liked at the top.
        hits = self._fuse([hits, self._sparse_hits(query_text, fetch_limit)], query_vector)[:fetch_limit]

        # Stage 3: Cross-Encoder Reranking
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

            # Order by the LOGIT, not by the fused score, and let the cosine
            # break ties. The sigmoid saturates above ~7: for the query
            # "CASE-10101" the exactly-matching chunk scored logit 8.59 against
            # 7.55 for an unrelated case, a decisive gap worth 0.0003 once
            # squashed - while the 0.036 cosine gap between them was worth
            # 0.014. The bi-encoder therefore overruled the reranker and the
            # exact identifier match, retrieved by BM25, was dropped from the
            # top 3. The fused score is still what gets REPORTED, so
            # similarity_threshold keeps the scale it was calibrated on.
            # ponytail: the CrossEncoder still gets the last word, so a query
            # mixing an identifier with a topic ("Compliance Review in
            # CASE-10077") can see BM25's exact-id chunks demoted below generic
            # Compliance Review chunks it likes better. Fix by reserving a slot
            # in the returned set for the sparse arm's top hit when the query
            # contains a term whose idf says it is an identifier.
            hits = sorted(
                hits,
                key=lambda x: (x["cross_encoder_logit"], x["similarity_score"]),
                reverse=True,
            )

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
                "retrieval_mode": self.retrieval_mode,
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
            "retrieval_mode": self.retrieval_mode,
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


def demo():
    """
    Self-check for the sparse arm. Needs no Milvus and no network.

    The assertion is the reason the arm was added: dense-only, the query
    "CASE-10101" ranked CASE-10119, CASE-10121 and CASE-10150 above every chunk
    of the case actually named.
    """
    assert _tokenize("Cost center CC-3305 in CASE-10101") == [
        "cost", "center", "cc-3305", "in", "case-10101"
    ], "tokeniser split an identifier; the sparse arm has nothing left to match on"

    chunks = ground_truth_chunks()
    assert chunks, f"no chunks built from {DATA_PATH}"
    bm25 = _BM25([c["text"] for c in chunks])

    ranked = bm25.top("CASE-10101", limit=10)
    assert ranked, "BM25 ranked nothing for a case id that is in the corpus"
    hit_cases = [chunks[i]["case_id"] for i in ranked]
    assert set(hit_cases) == {"CASE-10101"}, (
        f"exact case-id query pulled unrelated chunks: {sorted(set(hit_cases))}"
    )

    # An unrelated chunk must score zero, not merely less - it shares no term.
    unrelated = next(i for i, c in enumerate(chunks) if c["case_id"] == "CASE-10121")
    assert unrelated not in ranked, "an unrelated case outranked the case asked for"
    print(f"demo OK: BM25 over {len(chunks)} chunks returned only "
          f"{len(ranked)} CASE-10101 chunks for the id query")


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
    demo()
    main()
