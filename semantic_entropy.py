"""
Semantic entropy the way the paper defines it: K sampled continuations, clustered
by MEANING, entropy over the clusters.

Why this exists alongside semantic_divergence.py
------------------------------------------------
`semantic_divergence` clusters the top-k candidates AT ONE DECODE STEP by numeric
value. It is free - the distribution already exists - and it is genuinely useful,
but its own docstring is honest about the ceiling: all non-figures collapse into
one cluster, so a fabricated NAME ("Munich" vs "Berlin") reads as H = 0 and is
invisible to it. It also cannot see a claim that only becomes false several
tokens later, because it never looks past one step.

Farquhar et al. (doi:10.1038/s41586-024-07421-0) measure something different and
strictly stronger: sample N whole continuations, cluster them by bidirectional
entailment, and take the entropy over the clusters. Where token entropy asks "is
the model unsure of the next WORD", this asks "does the model tell a different
STORY each time it is run" - and a different story each time is the actual
signature of fabrication. A model quoting a figure out of its context returns the
same claim on every sample (H = 0) however uncertain the individual word choices
are; a model filling a gap from parametric memory returns a different claim each
time.

This module implements that, adapted for MID-STREAM use:

  * the "question" is the conversation so far PLUS the text already generated, fed
    back as an assistant prefill, so the samples continue the actual sentence
    rather than restarting the answer. Verified against the local Ollama build:
    a messages list ending in an assistant turn is continued, not answered.
  * continuations are a LOOKAHEAD WINDOW (default 10 tokens), not full answers.
    See `lookahead` for why that is enough and what it costs.
  * clustering is embedding-based by default, with a numeric override that no
    embedding model can be talked out of. See `_cluster`.

What it costs, stated up front
------------------------------
K draft generations per checkpoint. On this machine that is hundreds of
milliseconds - three orders of magnitude more than the free per-token layers. It
therefore CANNOT run per token, and the cascade in `should_checkpoint` is not an
optimisation, it is the only thing that makes the method usable at all. The
amortised per-token cost is (checkpoint cost x checkpoint rate), and
`benchmark_semantic_entropy.py` measures both halves rather than asserting them.
"""

import asyncio
import hashlib
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# A SEPARATE, SMALLER model, for two independent reasons.
#
# 1. Ollama on this machine runs with `-np 1`: one slot per loaded model. Drafting
#    on granite itself would queue behind - and therefore stall - the very stream
#    the user is watching. A different model gets its own llama-server process and
#    its own slot, so the checkpoint runs beside the main decode instead of in
#    front of it.
# 2. K samples of a 0.5B model cost a fraction of K samples of an 8B one, and the
#    checkpoint budget is the binding constraint.
#
# The methodological cost is real and must not be glossed: this measures the
# DRAFT model's semantic uncertainty, which is a proxy for granite's, not granite's
# own. The paper samples from the model under test. Set SENTINEL_DRAFT_MODEL to
# the serving model to run the faithful version - benchmark_semantic_entropy.py
# reports both, and the agreement between them is the evidence for the proxy.
DRAFT_MODEL = os.getenv("SENTINEL_DRAFT_MODEL", "qwen2.5:0.5b")

# Only figures matter for the numeric override below - reuse the same notion of
# "is this token a figure" the rest of the system already uses rather than
# inventing a second one that can drift from it.
_FIGURE_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

# Share of value-stating paths the leading value may hold before the split stops
# counting as a dice roll. Deliberately the same 0.75 as
# semantic_divergence.analyse's max_top_share, which makes the same judgement
# over one decode step's candidates rather than over K sampled paths.
MAX_TOP_SHARE = 0.75


def _figures(text: str) -> frozenset:
    """
    The distinct numeric values a continuation actually asserts.

    A figure touching the end of the window is DISCARDED, because early-exit
    sampling cuts continuations mid-number and a half-emitted figure is a
    different value from the one being written. This is not hypothetical - it was
    measured, and it inverted the verdict on a correct answer:

        "...is **8.5 days** across all 4|"      <- "460 events", truncated to 4
        "...is 8.5 days for 150|"               <- "150 cases", truncated at the edge
        "...is estimated at 8.5 days based on the"

    All three say 8.5. The phantom 4 and 150 made rule 1 in `_cluster` split them
    into three mutually-exclusive claims, so a grounded answer scored 0.83 of
    maximum entropy and would have been blocked - the exact false positive that
    costs enterprise trust fastest.

    SentinelStream._scan_new_figures already declines to judge a figure that has
    not been terminated by a following character, for the same reason. Same rule,
    applied at the other end of the pipeline.
    """
    out = set()
    for m in _FIGURE_RE.finditer(text):
        # Trailing separators do not terminate a number: "87." may still become
        # "87.6" had the window been one token longer.
        tail = text[m.end():]
        if not tail or set(tail) <= {".", ",", " "}:
            continue
        cleaned = m.group().replace("$", "").replace(",", "").replace("%", "").rstrip(".")
        try:
            out.add(round(float(cleaned), 2))
        except ValueError:
            pass
    return frozenset(out)


@dataclass
class SemanticEntropyResult:
    """One K-path measurement."""
    entropy: float                         # bits over semantic clusters
    normalised: float                      # entropy / log2(K); comparable across K
    clusters: List[List[str]]              # continuations grouped by meaning
    paths: List[str]                       # the raw samples, for the audit trail
    k: int
    sim_tau: float
    clusterer: str                         # embedding | lexical | numeric-only
    elapsed_ms: float
    cached: bool = False
    figures: List[float] = field(default_factory=list)   # distinct values claimed
    figure_share: float = 1.0     # share of value-stating paths on the modal value

    @property
    def divergent_figures(self) -> bool:
        """
        The model rolled dice on a NUMBER - the enterprise case.

        Requires that the leading value not DOMINATE, not merely that two values
        appeared. Measured false positive on a correct answer: asked how many
        cases are in the log, four of five paths said "150 cases" and one said
        "460 cases", and blocking on the bare presence of two values halted an
        answer the model had right four times out of five.

        One dissenting sample is not evidence, for the same reason
        SentinelStream.breach_run exists: a single outlier is what sampling at
        temperature 1.0 does. MAX_TOP_SHARE is the same 0.75 that
        semantic_divergence.analyse uses for the same judgement one layer down -
        one concept, one number.
        """
        return len(self.figures) > 1 and self.figure_share <= MAX_TOP_SHARE

    def explain(self) -> str:
        """One sentence a non-specialist can read off the screen."""
        n = len(self.clusters)
        if self.divergent_figures:
            vals = ", ".join(str(f) for f in self.figures[:4])
            return (
                f"{self.k} independent continuations of this exact sentence produced "
                f"{len(self.figures)} different figures - {vals} - so the model does not "
                f"know the value; it settles on one only because a single sample has to "
                f"pick something"
            )
        if n == 1:
            return (
                f"all {self.k} continuations said the same thing, so the model is not "
                f"guessing here - it is reproducing something it was given"
            )
        if self.figures and self.figure_share > MAX_TOP_SHARE:
            return (
                f"{self.figure_share:.0%} of the continuations that stated a value agreed on "
                f"{self.figures[0] if len(self.figures) == 1 else 'one figure'} - the "
                f"{n - 1} dissenting path{'' if n == 2 else 's'} did not shift it, so this "
                f"reads as one claim, not a guess"
            )
        if self.normalised < 0.5:
            return (
                f"{self.k} continuations fell into {n} meanings ({self.entropy:.2f} bits, "
                f"{self.normalised:.0%} of maximum) - mostly agreement, with some variation "
                f"in how it is worded"
            )
        return (
            f"{self.k} continuations fell into {n} distinct meanings "
            f"({self.entropy:.2f} bits, {self.normalised:.0%} of maximum) - the model "
            f"tells a different story each run, which is what fabrication looks like"
        )


class SemanticEntropy:
    """K-path speculative semantic entropy over a draft model."""

    def __init__(
        self,
        draft_model: str = DRAFT_MODEL,
        host: str = OLLAMA_HOST,
        k: int = 5,
        lookahead: int = 10,
        sim_tau: float = 0.60,
        temperature: float = 1.0,
        embedder=None,
        cache_size: int = 128,
        timeout: int = 60,
    ):
        """
        :param k: number of speculative paths. The entropy estimate is over K
            samples, so its ceiling is log2(K) - K=5 gives 2.32 bits, which is
            plenty of resolution for a block/hedge/pass decision and is where the
            latency curve is still flat. K is the dominant cost term; see the
            benchmark before raising it.

        :param lookahead: tokens per path. EARLY-EXIT SAMPLING - the paper
            generates full answers, which is right for offline evaluation and
            wrong here. Mid-stream we are asking one question - "are these
            continuations the same claim or different ones" - and that is settled
            within a few tokens of the decision point, because the figure or the
            entity lands almost immediately after the prefix that set it up.
            Generating 60 tokens to read the first 10 spends 6x the compute on
            text that only adds noise to the clustering. Measured: the full-length
            variant is available for calibration via `lookahead=None`, and
            benchmark_semantic_entropy.py reports what the truncation costs in
            agreement.

        :param sim_tau: cosine above which two continuations are the same meaning.

            Swept, not guessed. `calibrate_sentinel.py` clusters the K paths from
            100 labelled queries at each threshold and reports mean normalised
            entropy per class; the column that decides it is the SEPARATION
            between them, because a threshold that raises both equally has bought
            nothing. Measured, numeric override ON:

                sim_tau   H(answerable)  H(unanswerable)  separation
                  0.50        0.507           0.670          0.163
                  0.60        0.550           0.734          0.184   <- this one
                  0.65        0.592           0.761          0.169
                  0.72        0.681           0.805          0.125
                  0.80        0.738           0.866          0.129
                  0.90        0.834           0.898          0.064

            Both failure directions are visible in that table. Toward 0.90 every
            paraphrase splits, answerable entropy climbs to 0.834, and the
            classes converge - false alarms. Toward 0.50 everything merges and
            unanswerable entropy falls to 0.670 - false confidence. 0.60 is the
            maximum, and the default was 0.72 on nothing but a guess.

            The same table with the numeric override DISABLED is what justifies
            that rule: at 0.50 separation collapses from 0.163 to 0.057, because
            without it two continuations that differ only in the figure they
            state embed at ~0.98 and merge into one cluster.
        """
        self.draft_model = draft_model
        self.host = host.rstrip("/")
        self.k = k
        self.lookahead = lookahead
        self.sim_tau = sim_tau
        self.temperature = temperature
        self.timeout = timeout
        self._embedder = embedder
        self._embedder_tried = embedder is not None
        # CACHED CLUSTER REUSE. An OrderedDict used as an LRU rather than
        # functools.lru_cache, because the key has to be built from a normalised
        # prefix rather than from the raw arguments: the same claim reached by a
        # slightly different route must hit the same entry. Poison prompts get
        # asked more than once in a demo, and common questions repeat in
        # production - and a hit costs 0 draft generations instead of K.
        self._cache: "OrderedDict[str, SemanticEntropyResult]" = OrderedDict()
        self._cache_size = cache_size
        self.hits = 0
        self.misses = 0

    # ---------- embedding ----------

    def _embed(self, texts: Sequence[str]):
        """
        Sentence embeddings, or None when the model is unavailable.

        Loaded lazily and never fatally: the offline demo must survive a missing
        or moved model file, so a failure here degrades the clusterer to lexical
        overlap rather than taking the stream down. all-MiniLM-L6-v2 is already a
        dependency of the retriever and already cached, so in the normal case
        this costs no download and shares the process's existing weights.
        """
        if self._embedder is None and not self._embedder_tried:
            self._embedder_tried = True
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as err:
                print(f"[SemanticEntropy] embedder unavailable ({err}); using lexical clustering")
                self._embedder = None
        if self._embedder is None:
            return None
        return self._embedder.encode(list(texts), show_progress_bar=False,
                                     normalize_embeddings=True)

    # ---------- clustering ----------

    def _cluster(self, paths: List[str]) -> Tuple[List[List[int]], str]:
        """
        Group continuations by meaning. Returns (clusters as index lists, method).

        Two rules, and the ORDER matters.

        1. A DIFFERENT FIGURE IS A DIFFERENT MEANING, whatever the embedding says.
           This is the single most important line in the module. "is 10.4 days"
           and "is 12.5 days" embed at cosine ~0.98 - they are the same sentence
           but for two characters - so a pure embedding clusterer merges them,
           reports one cluster, H = 0, and concludes the model is confident at the
           exact moment it is rolling dice on a number. That is not a tuning
           problem that a lower sim_tau fixes: lowering it far enough to split
           those also splits every honest paraphrase. The two claims are
           mutually exclusive assertions about the world and no similarity
           threshold expresses that, so it is asserted directly.

           This is the same equivalence relation semantic_divergence.cluster_key
           applies to single tokens, lifted to whole continuations.

        2. Otherwise, greedy single-link agglomeration on cosine >= sim_tau.
           Greedy and single-link because K is 5: the O(K^2) comparison is 10 dot
           products on 384-dim unit vectors, and any cleverer algorithm would cost
           more to justify than it saves.

           ponytail: greedy single-link is order-dependent at the margin - a
           different sample order can merge a borderline pair differently. At K=5
           the effect is below the resolution of the block/hedge decision.
           Upgrade path is proper agglomerative clustering if K ever exceeds ~20.
        """
        n = len(paths)
        if n <= 1:
            return [[0]] if n else [], "trivial"

        figs = [_figures(p) for p in paths]
        vecs = self._embed(paths)
        method = "embedding" if vecs is not None else "lexical"

        def same(i: int, j: int) -> bool:
            # Rules 1 and 2, both only when BOTH continuations commit to a value:
            # a path that has not reached its figure yet must not be forced
            # either way against one that has, or an early-exit window would move
            # the entropy purely by being short.
            if figs[i] and figs[j]:
                # Rule 1: different figures are different claims.
                # Rule 2: the SAME figure is the SAME claim, and the prose around
                # it is wording. This is the converse and it is equally
                # load-bearing. Measured on five real qwen2.5 continuations of a
                # correct answer, all four that stated a value said 8.5 - and
                # cosine put "estimated at 8.5 days based on the" and "8.5 days
                # for 150" below sim_tau against the others, so the embedding
                # alone reported four clusters on one claim. In a process-mining
                # answer the claim IS the figure; everything else is phrasing,
                # which is the exact distinction semantic_divergence is built on.
                #
                # ponytail: this cannot separate "8.5 days for Compliance Review"
                # from "8.5 days for order-to-cash" - same value, different
                # metric. That is a scope error, and it is the numeric-grounding
                # layer's job (cm.bound_metrics), not this one's: for entropy
                # purposes the model committed to one value either way. Upgrade
                # path is to key the figure by its bound metric, which needs the
                # full sentence the lookahead window deliberately does not have.
                return figs[i] == figs[j]
            if vecs is not None:
                return float(vecs[i] @ vecs[j]) >= self.sim_tau
            # Lexical fallback: Jaccard over content words. Coarse, and only ever
            # reached when the embedding model is missing, which is a degraded
            # demo rather than a normal run.
            a = set(re.findall(r"[a-z0-9.]+", paths[i].lower()))
            b = set(re.findall(r"[a-z0-9.]+", paths[j].lower()))
            return bool(a & b) and len(a & b) / len(a | b) >= 0.5

        clusters: List[List[int]] = []
        for i in range(n):
            for c in clusters:
                if any(same(i, j) for j in c):        # single-link
                    c.append(i)
                    break
            else:
                clusters.append([i])
        return clusters, method

    # ---------- measurement ----------

    def _key(self, messages: List[Dict[str, str]], prefix: str) -> str:
        """
        Cache key: the question plus the tail of what has been generated.

        The TAIL, not the whole prefix, because the continuation only depends on
        recent context in any way this measurement can detect, and keying on the
        full prefix would miss every repeat of the same sentence reached after a
        different opening - which is most of them.
        """
        q = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        blob = f"{self.draft_model}|{self.k}|{self.lookahead}|{self.sim_tau}|{q}|{prefix[-120:]}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    async def _sample(self, client: httpx.AsyncClient, messages, prefix, seed: int) -> str:
        """One speculative path, continuing `prefix` rather than restarting it."""
        body = {
            "model": self.draft_model,
            # The assistant turn is a PREFILL: Ollama continues it instead of
            # answering afresh, which is what makes this a measurement of the
            # sentence in flight rather than of the question in general.
            "messages": list(messages) + ([{"role": "assistant", "content": prefix}] if prefix else []),
            "stream": False,
            "options": {
                "num_predict": self.lookahead,
                # Temperature is load-bearing and must NOT be 0. The serving path
                # decodes greedily, and greedy sampling returns the identical
                # continuation K times - entropy 0 for every input, including
                # outright fabrications. Semantic entropy is a measurement of the
                # DISTRIBUTION, so the samples have to be drawn from it.
                "temperature": self.temperature,
                "top_p": 0.95,
                "seed": seed,
            },
            "keep_alive": -1,
        }
        r = await client.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    async def measure(
        self, messages: List[Dict[str, str]], prefix: str = ""
    ) -> Optional[SemanticEntropyResult]:
        """
        Sample K continuations of `prefix` and return the entropy over their
        meanings. None when the draft model is unreachable - a checkpoint that
        cannot run must leave the other layers in charge, not halt the stream.
        """
        key = self._key(messages, prefix)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            hit = self._cache[key]
            # Copied with cached=True so a caller can tell a reused verdict from a
            # fresh one; the UI says so, because "we did not recompute this" is a
            # claim about the evidence and hiding it would overstate the check.
            return SemanticEntropyResult(**{**hit.__dict__, "cached": True})

        self.misses += 1
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                paths = await asyncio.gather(
                    *[self._sample(client, messages, prefix, seed) for seed in range(self.k)]
                )
        except Exception as err:
            print(f"[SemanticEntropy] draft sampling failed ({err}); checkpoint skipped")
            return None

        paths = [p for p in paths if p and p.strip()]
        if not paths:
            return None

        clusters, method = self._cluster(paths)
        total = len(paths)
        h = -sum((len(c) / total) * math.log2(len(c) / total) for c in clusters)
        # abs(): a single cluster gives -(1.0 * log2 1.0) == -0.0, which renders as
        # "-0.00". Entropy is never negative; the sign is a float artefact.
        h = abs(h)
        ceiling = math.log2(total) if total > 1 else 1.0

        per_path = [_figures(p) for p in paths]
        figures = sorted({f for s in per_path for f in s})
        # How concentrated the value claims are. Counted over paths that actually
        # STATED a value, not over K: a path that hedged or ran out of window has
        # no opinion, and counting it as dissent would make a short lookahead
        # look like disagreement.
        stating = [s for s in per_path if s]
        share = 1.0
        if stating:
            tally: Dict[float, int] = {}
            for s in stating:
                for f in s:
                    tally[f] = tally.get(f, 0) + 1
            share = max(tally.values()) / len(stating)

        result = SemanticEntropyResult(
            entropy=round(h, 4),
            normalised=round(h / ceiling, 4) if ceiling else 0.0,
            clusters=[[paths[i] for i in c] for c in clusters],
            paths=paths,
            k=total,
            sim_tau=self.sim_tau,
            clusterer=method,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
            figures=figures,
            figure_share=round(share, 4),
        )
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result


# --------------------------------------------------------------------------- #
# Adaptive checkpointing - what makes the method affordable
# --------------------------------------------------------------------------- #

# Token-type risk is already classified once, by EntropyEngine.get_token_type_weight
# (3.5 for figures and currency, 3.0 for identifiers, 2.5 for regulatory
# acronyms, 2.2 for titles and jurisdictions, 0.4 for grammatical glue). Reuse it
# rather than writing a second taxonomy that can drift out of step with the one
# the scalar layer weights by.
HIGH_STAKES_WEIGHT = 2.2


def should_checkpoint(
    divergence,
    tokens_since_last: int,
    token_weight: float,
    min_gap: int = 6,
    numeric_mass_floor: float = 0.25,
) -> Optional[str]:
    """
    Should a K-path check run at this token? Returns the reason, or None.

    ADAPTIVE CHECKPOINT FREQUENCY. Entropy at every token is unaffordable here -
    a checkpoint costs K draft generations, hundreds of milliseconds - and
    entropy only at the end is post-hoc, which is the thing this project exists
    to replace. So the expensive check is GATED by the two signals that are
    already free:

      * `divergence.numeric_mass` - how much of granite's own next-token
        distribution is sitting on digits. This is the strongest trigger and the
        earliest: it fires BEFORE the figure is emitted, at the moment the model
        is deciding whether to state a number at all. Nothing else in the system
        can see that far ahead.
      * the token just emitted being high-stakes by weight - a figure, a currency
        amount, an identifier, a date, a proper noun. Risk of fabrication is
        concentrated on exactly these, and connective prose carries almost none.

    Filler text ("the", "of", "is", punctuation) never triggers a checkpoint, so
    compute lands where the hallucination risk actually is. `min_gap` stops a run
    of digits inside one figure from firing K checkpoints for one claim.

    Returns a reason string rather than a bool so the audit trail can say WHY the
    expensive check ran - "a figure was 40% of the distribution" is evidence a
    judge can check, and a bare True is not.

    MEASURED, AND IT DOES NOT MEET THE 15 ms/TOKEN BUDGET. Say so plainly.
    Over 101 labelled queries (calibrate_sentinel.py):

        one checkpoint          943 ms median (p90 1591)
        checkpoint rate         7.5%  - 1 per 13.3 tokens
        amortised               70.9 ms/token          <- 4.7x over budget
        always-on layers         0.18 ms/token         <- 80x under budget

    The gate is doing its job - it divides the naive per-token cost by 13 - and
    the layer is still over. The knob table in the report prices both levers
    against the same cached generations, and the only combination that clears
    15 ms is min_gap=40 AND raising OLLAMA_NUM_PARALLEL so K=5 costs what K=1
    does today (14.7 ms/token, at the edge). On this hardware, in the always-on
    streaming path, this layer is a production blocker rather than a shipped
    feature.

    What it IS affordable for, and what it is used for here: evidence at
    interception time and escalation of a hedge to a block, where one checkpoint
    per intercepted response costs ~1 s against a repair pass that already costs
    more than that. The cheap layers stay in the hot path; this one earns its
    cost only where they have already flagged something. Raise `min_gap` toward
    40 to move it in that direction without changing any other behaviour.
    """
    if tokens_since_last < min_gap:
        return None
    # Ahead of the figure: the model is about to commit to a value.
    if divergence is not None and divergence.numeric_mass >= numeric_mass_floor:
        return f"figure_imminent(numeric_mass={divergence.numeric_mass:.2f})"
    # On the figure: it has landed and the claim can now be sampled around.
    if token_weight >= HIGH_STAKES_WEIGHT:
        return f"high_stakes_token(weight={token_weight:.1f})"
    return None


# --------------------------------------------------------------------------- #
# Confidence-aware fallback tiering
# --------------------------------------------------------------------------- #

# Tier boundaries, as MULTIPLES of tau rather than absolute scores, so the tiers
# move with the calibrated threshold instead of having to be recalibrated
# alongside it.
HEDGE_FACTOR = 1.0        # at or above tau
BLOCK_FACTOR = 1.6        # far above tau
# Normalised semantic entropy (0..1 of log2 K) at which the K-path evidence is
# conclusive on its own, whatever the scalar score says.
SEMANTIC_BLOCK = 0.75


def fallback_tier(
    uncertainty: float,
    tau: float,
    semantic: Optional[SemanticEntropyResult] = None,
) -> Tuple[str, str]:
    """
    Choose what to DO about an uncertain claim. Returns (tier, why).

    CONFIDENCE-AWARE FALLBACK TIERING. A binary halt throws away the magnitude of
    the very quantity this system spends all its compute measuring: a score one
    percent over tau and a score three times over it produce the identical
    response, which is what makes a threshold feel arbitrary to the person it
    just blocked. Three tiers:

      pass  - below tau. Stream it.
      hedge - above tau but not decisively. The answer is delivered WITH an
              explicit uncertainty marker rather than withheld. This is the tier
              that protects against the failure mode that actually destroys
              enterprise trust: blocking a correct answer. A hedge on a true
              statement costs a caveat; a block on a true statement costs the
              user's belief that the system works.
      block - decisively over, or the K-path samples disagree with each other
              about a figure. Halt, log the gap, run the repair.

    The semantic-entropy escalation is deliberately one-directional: K paths
    disagreeing can PROMOTE a hedge to a block, but K paths agreeing never
    demotes a block, because agreement is not grounding - the model can be
    consistently wrong, and that is precisely the case the deterministic numeric
    layer exists for.
    """
    if semantic is not None:
        if semantic.divergent_figures:
            return "block", (
                f"{semantic.k} speculative continuations produced "
                f"{len(semantic.figures)} different figures {semantic.figures}"
            )
        if semantic.normalised >= SEMANTIC_BLOCK:
            return "block", (
                f"semantic entropy {semantic.entropy:.2f} bits over "
                f"{len(semantic.clusters)} distinct meanings "
                f"({semantic.normalised:.0%} of maximum)"
            )
    ratio = uncertainty / tau if tau else 0.0
    if ratio >= BLOCK_FACTOR:
        return "block", f"uncertainty {uncertainty:.3f} is {ratio:.1f}x tau ({tau})"
    if ratio >= HEDGE_FACTOR:
        return "hedge", f"uncertainty {uncertainty:.3f} is {ratio:.2f}x tau ({tau})"
    return "pass", f"uncertainty {uncertainty:.3f} below tau ({tau})"


# --------------------------------------------------------------------------- #

def demo() -> None:
    """
    Self-check. Offline: exercises clustering, tiering and the checkpoint gate
    without touching Ollama, so it runs in the test suite. The live sampling path
    is measured by benchmark_semantic_entropy.py.
    """
    se = SemanticEntropy(k=5)

    # THE case the numeric override exists for. These five differ by two
    # characters and embed at cosine ~0.98, so a pure embedding clusterer merges
    # them all and reports H = 0 on a model that is plainly guessing.
    paths = [" is 10.4 days", " is 12.5 days", " is 7 days", " is 10.4 days", " is 15 days"]
    clusters, method = se._cluster(paths)
    assert len(clusters) == 4, (clusters, method)      # 10.4 twice, then 12.5, 7, 15

    # Paraphrases of ONE claim must not split, or every honest answer looks like
    # a fabrication.
    same = [
        " is approximately 10.4 days",
        " is about 10.4 days",
        " is 10.4 days on average",
        " averages 10.4 days",
        " is 10.4 days",
    ]
    cl2, _ = se._cluster(same)
    assert len(cl2) == 1, cl2

    # A path that has not reached its figure yet must not be forced apart from
    # one that has - otherwise the lookahead window inflates entropy by itself.
    partial = [" is 10.4 days", " is 10.4 days", " is roughly"]
    assert len(se._cluster(partial)[0]) <= 2, se._cluster(partial)

    # A figure the window cut in half is not a claim. Measured regression: these
    # are five real qwen2.5 continuations of a CORRECT grounded answer, where
    # "across all 4" is a truncated 460 and "for 150" a truncated case count.
    # Counting them scored a true statement at 0.83 of maximum entropy - a block
    # on a correct answer, the most expensive error this system can make.
    assert _figures(" is 8.5 days across all 4") == frozenset({8.5})
    assert _figures(" is 8.5 days for 150") == frozenset({8.5})
    assert _figures(" is 8.5 days across all 460 events.") == frozenset({8.5, 460.0})
    grounded = [
        " is **8.5 days** across all 4",
        " is estimated at 8.5 days based on the",
        " is 8.5 days across all events.",
        " is 8.5 days for 150",
        " is calculated as the average of all the computed",
    ]
    gc, _ = se._cluster(grounded)
    assert len(gc) <= 2, gc     # four ways of saying 8.5, plus one hedge

    # Entropy arithmetic: 5 paths in 5 clusters is the ceiling, log2(5).
    assert abs(math.log2(5) - 2.3219) < 1e-3

    # Checkpoint gate: filler never fires, an imminent figure does, and the rate
    # limit holds.
    class D:
        def __init__(self, m): self.numeric_mass = m
    assert should_checkpoint(D(0.0), 99, 0.4) is None
    assert should_checkpoint(D(0.0), 99, 3.5).startswith("high_stakes")
    assert should_checkpoint(D(0.61), 99, 1.0).startswith("figure_imminent")
    assert should_checkpoint(D(0.9), 2, 3.5) is None               # rate-limited

    # Tiering: a hedge exists between pass and block, and K-path figure
    # disagreement escalates to block regardless of the scalar score.
    assert fallback_tier(0.30, 0.65)[0] == "pass"
    assert fallback_tier(0.70, 0.65)[0] == "hedge"
    assert fallback_tier(1.40, 0.65)[0] == "block"
    split = SemanticEntropyResult(entropy=1.92, normalised=0.83, clusters=[[], [], []],
                                  paths=[], k=5, sim_tau=0.72, clusterer="embedding",
                                  elapsed_ms=0.0, figures=[7.0, 10.4, 12.5],
                                  figure_share=0.4)
    tier, why = fallback_tier(0.10, 0.65, split)
    assert tier == "block", (tier, why)
    assert "different figures" in why

    # ...but ONE dissenting sample is not a dice roll. Measured false positive:
    # four of five paths answered "150 cases" correctly and one said "460", and
    # blocking on the mere presence of two values halted a right answer.
    outlier = SemanticEntropyResult(entropy=0.72, normalised=0.31, clusters=[[], []],
                                    paths=[], k=5, sim_tau=0.72, clusterer="embedding",
                                    elapsed_ms=0.0, figures=[150.0, 460.0],
                                    figure_share=0.8)
    assert not outlier.divergent_figures, outlier.figure_share
    assert fallback_tier(0.30, 0.65, outlier)[0] == "pass"

    # Agreement must NOT demote: the model can be consistently wrong.
    agreed = SemanticEntropyResult(entropy=0.0, normalised=0.0, clusters=[[]], paths=[],
                                   k=5, sim_tau=0.72, clusterer="embedding",
                                   elapsed_ms=0.0, figures=[4.2])
    assert fallback_tier(1.40, 0.65, agreed)[0] == "block"

    print("semantic_entropy self-check OK")
    print("  clusterer:", method)
    print("  ", split.explain())
    print("  ", outlier.explain())


if __name__ == "__main__":
    demo()
