"""
The actual interception loop: real Granite decoding, real entropy, real halt.

What changed from the previous implementation
---------------------------------------------
Before, `main.py` decided `is_poison` by keyword-matching the query BEFORE
generation started, streamed a hardcoded f-string with `logprob=-0.1` pinned on
every token, and threw away the `is_hallucinating` flag the engine returned. The
breaker never broke a circuit; it played a scripted animation.

Here the decision is made per token, mid-decode, from the model's own output
distribution, and tripping it actually terminates the generator.

Two detection layers, because they fail differently:

  Layer 1 - Semantic entropy (model-internal). Catches the model *guessing*:
      flat next-token distribution, high log-prob variance. Detects uncertainty
      even when the sentence is fluent. Blind to confident falsehood.

  Layer 2 - Numeric grounding (data-deterministic). Catches the model being
      *confidently wrong*: a figure stated at p=0.99 that the event log cannot
      produce. Blind to hedging prose.

An enterprise hallucination is usually a confident fabricated number, which is
exactly the case Layer 1 alone misses. That is why both exist.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import celonis_metrics as cm
import gap_ledger
import semantic_divergence as sd
import semantic_entropy as se
from entropy_engine import EntropyEngine

AUDIT_LOG_PATH = Path(__file__).parent / "data" / "audit_log.jsonl"


def log_compliance_breach(query: str, reason: str, score: float, value: Optional[float] = None) -> None:
    """
    Append-only record of every interception, for the compliance trail.

    The query is SHA-256'd, not stored: an audit log that reproduces the
    questions people asked is itself a disclosure risk, and the hash is enough to
    prove the same prompt recurred.

    JSONL rather than a JSON array, because the array form has to read the whole
    file, append and rewrite it on every breach - which is O(n) per interception
    and truncates the log if the process dies mid-write. One line, one append,
    nothing to corrupt.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "action": "INTERCEPTED_AND_HALTED",
        "reason": reason,
        "score": round(score, 4),
        "ungrounded_value": value,
    }
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as err:
        # An unwritable audit log must not take the demo down with it.
        print(f"[Sentinel] audit log write failed: {err}")

# A quantitative claim: a standalone number, optionally with $ , . and %.
#
# The lookarounds exclude digits embedded in enterprise identifiers - W-99,
# CC-9999, CASE-10298, SOX-404. Those are entity references, not figures, and
# treating them as claims caused a false interception on a response that was
# correctly refusing to answer: Granite said "...for warehouse node W-99" and
# the "99" was scored as a fabricated number.
_NUM_RE = re.compile(
    # ':' is excluded on both sides so a clock time is not read as a claim:
    # "created on 2026-06-04T16:47:00Z" yielded 47, which no metric can produce,
    # and halted a sentence that only quoted a timestamp out of a chunk.
    r"(?<![A-Za-z0-9_.\-:])"     # not preceded by identifier characters
    # The digit/separator run must END on a digit. As [\d,]* it swallowed a
    # trailing sentence comma - "above $100,000, the mean..." matched
    # "$100,000," - and the parsed value was then checked as though the claim
    # were about the comma's clause. Harmless-looking, but it was reporting the
    # log's own threshold as a contradiction.
    r"-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?"
    # A scale suffix is part of the figure, not the prose after it. Without this
    # "Total logged order value is $110.6M" matched "$110" - the trailing
    # lookahead rejected "$110.6" because 'M' follows, so the regex backtracked
    # to the integer part - and reported 110 as ungrounded on a sentence whose
    # real value, $110,559,589.27, is the log's own total. The spelled-out form
    # failed the same way for a different reason: 110.6 parsed cleanly and was
    # then compared against the log as though it meant 110.6 dollars.
    r"(?:[KMB](?![A-Za-z])|\s+(?:thousand|million|billion))?"
    # ':' is excluded on the LEFT only. Excluding it on the right broke
    # "Orders above $100,000:" in the aggregate block - the match backtracked to
    # "$100" and reported 100 as ungrounded. A colon BEFORE a number is a clock;
    # a colon after one is punctuation.
    r"(?![A-Za-z0-9_\-])"        # not followed by identifier characters
)

# Multipliers for the suffix above, keyed by the lowercased tail of the match.
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "k": 1e3, "m": 1e6, "b": 1e9}

# A comparison BASELINE is a claim about a different metric by construction:
#
#   "Compliance Review averages 10.43 days versus an overall mean of 8.5 days."
#
# Both figures are the log's own, but neither ';' nor ',' separates them, so the
# metric span for 8.5 reached back over "Compliance Review" and checked the
# OVERALL mean against that activity's - halting a true sentence. Metric binding
# stops here; the connective marks where one claim ends and its foil begins.
_CONTRAST_BREAK = re.compile(
    r"\b(?:versus|vs\.?|compared\s+(?:to|with)|as\s+against)\b", re.IGNORECASE
)


# Where one quantitative claim ends and the next begins. Punctuation, plus the
# connectives that start a new claim about a different quantity ("... 122 times
# WITH a mean cycle time of 10.43"). Deliberately does NOT include "of" or "is",
# which sit inside a single claim ("a mean cycle time OF 10.43").
# A period or comma only ends a clause when it is NOT inside a number.
#
# Written as a bare [.,] this split "The mean compliance cycle time is 10.4" at
# the decimal point, leaving the clause as "4" - so bound_metric saw no metric
# and every DECIMAL figure silently lost its binding. Integers still bound, which
# is what made it hard to see: "12 days" halted and "12.5 days" sailed through,
# and 12.5 is the flagship demo interception. Thousands separators had the same
# problem ("$128,500" -> "500").
# \n is a hard boundary in both: the aggregate block is one claim per LINE, and
# without it "Scope: 150 cases, 460 events" reached into the next line and bound
# 460 to "compliance cycle time", so the context the model is told to quote
# failed the check applied to the model for quoting it.
_CLAUSE_BREAK = re.compile(
    r"(?<!\d)[.,](?!\d)|[;:()\[\]\n]"
    r"|\b(?:with|across|while|whereas|although|and|but|however)\b",
    re.IGNORECASE,
)

# Sentence-level break: punctuation that separates a TITLE or a parenthetical
# from a claim, but NOT the conjunctions above. Used for metric binding, which
# has a wider natural scope than statistic binding - see _metric_span.
_SENTENCE_BREAK = re.compile(r"(?<!\d)[.](?!\d)|[;:()\[\]\n]")

# A figure in a QUALIFIER slot is not a value OF the metric named around it - it
# is a condition on that metric, or the population it is drawn from:
#
#   "high-value orders above $100,000"   $100,000 is the CRITERION, not a cycle time
#   "38 of 150 cases were flagged ..."   150 is the DENOMINATOR, not a bottleneck count
#
# Binding those narrows the admissible set to the metric's own values and then
# rejects a figure that is true, which is a halt on a correct answer - both of
# these appear in the system's OWN deterministic recovery answers. The "of"
# branch requires a preceding figure so the far more common "a mean cycle time
# of 10.43" still binds normally.
#
# \b is load-bearing: without it "over" matched inside "cover", so
# "Bottleneck flags cover 25.33% of cases" lost its metric binding entirely and
# the percentage rule below then rejected it as underivable.
#
# "at least"/"at most" sit in a branch of their own because they are the one
# pair that is BOTH a qualifier and a statistic word. Before a money amount they
# select a population ("orders of at least $100,000"); before a bare figure they
# assert an extremum ("Compliance Review takes at most 10.43 days" claims that
# activity's maximum, which is 16 - the mean quoted as the worst case). Blanket
# suppression cost the binding in the second reading, so the claim reached no
# metric at all and a wrong-statistic lie passed as merely unverifiable.
# Requiring the '$' keeps the population reading and returns the other.
_QUALIFIER_BEFORE = re.compile(
    r"\b(?:\d[\d,.]*\s+of"
    r"|above|below|over|under|exceeding|in excess of"
    r"|more than|less than)\s+\$?$"
    r"|\b(?:at least|at most)\s+\$$",
    re.IGNORECASE,
)


@dataclass
class InterceptionEvent:
    kind: str                      # token | intercept | recovery | done
    text: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


class SentinelStream:
    """Wraps an OllamaRunner with per-token interception."""

    def __init__(
        self,
        runner,
        retriever=None,
        tau: float = 0.65,
        window_size: int = 5,
        check_numbers: bool = True,
        max_new_tokens: int = 160,
        breach_run: int = 3,
        grounded_prompt: bool = True,
        semantic: Optional["se.SemanticEntropy"] = None,
        tiering: bool = True,
    ):
        """
        :param breach_run: consecutive threshold breaches required to trip the
            entropy layer. One flat token is not evidence of fabrication - the
            first token of a sentence is inherently uncertain ("From"/"The"/
            "According" are all valid openings, so no single one holds much
            probability), and hedging prose is legitimately high-entropy because
            there are many ways to phrase "I cannot verify that". Both produced
            false interceptions at breach_run=1: a clean query halted on token 0,
            and a correct refusal halted mid-sentence.

            Sustained high entropy is the real signal. Token entropy measures
            uncertainty over WORDING; requiring a run is what lifts it toward
            uncertainty over MEANING, which is the quantity the semantic-entropy
            result is actually about.

            The numeric layer deliberately does NOT wait for a run: an ungrounded
            figure is a deterministic fact-check, not a noisy statistic, so one
            occurrence is conclusive.

            Default of 3 comes from calibrate_sentinel.py over 101 labelled
            queries (52 answerable / 49 unanswerable), which supersedes the
            12-prompt sweep this docstring used to quote. With the position-aware
            threshold on, at tau=0.65:

                breach_run=2  ->  3/52 false positives, 0/49 misses
                breach_run=3  ->  2/52 false positives, 0/49 misses   <- this one

            A third required breach removes a false positive and costs no
            detection, on a set where 12 prompts could not have told the
            difference. The recommended setting is the middle of a five-wide
            plateau (tau 0.55-0.75, all 2/52 and 0/49); the middle is taken
            because the edges of a plateau are where the estimate is least
            reliable.

            READ THE MISS COLUMN CAREFULLY. 0/49 is not the entropy layer's
            achievement: 48 of the 49 unanswerable queries are stopped by the
            DETERMINISTIC numeric layer, and none of the 49 was refused outright.
            The statistical layer is a supplement here, not the workhorse - which
            is the same conclusion the two-layer design was built on, now measured
            rather than argued. Both remaining false positives are also numeric,
            so no value of tau or breach_run removes them.
        """
        self.runner = runner
        self.retriever = retriever
        self.tau = tau
        self.window_size = window_size
        self.check_numbers = check_numbers
        self.max_new_tokens = max_new_tokens
        self.breach_run = breach_run
        # False selects the naive-integration system prompt, so the baseline
        # panel can be audited with the same code path it is compared against.
        self.grounded_prompt = grounded_prompt
        # The K-path layer. None disables it entirely and the stream behaves
        # exactly as it did before - which is what keeps the offline demo alive
        # when the draft model is not pulled, and what the A/B in the benchmark
        # toggles.
        self.semantic = semantic
        # False collapses the three tiers back to the original binary halt.
        self.tiering = tiering

    # ---------- retrieval ----------

    @staticmethod
    def _aggregate_block() -> str:
        """
        Deterministic aggregates, prepended to every context.

        Retrieval alone returns individual event chunks, so asking for a mean
        made Granite try to average the three cases it happened to retrieve -
        stochastic arithmetic over a sampled subset, which is precisely the
        failure this project exists to prevent. The aggregate is computed once
        over all 460 events and handed to the model as a fact to quote.
        """
        p = cm.process_profile()
        lines = [
            f"Process: {p['process']} ({p['source_system']})",
            f"Scope: {p['total_cases']} cases, {p['total_events']} events",
            # "mean" and "average" both, because the model matches the question's
            # wording against these lines lexically. Asked "What is the MEAN
            # compliance cycle time?" against a line reading "Declared AVERAGE
            # compliance cycle time: 10.4 days", Granite answered "9.11 days for
            # orders above $100,000" - the nearest line that actually contained
            # the words "mean cycle". Measured 0/6 correct before, 6/6 after; the
            # decode is deterministic, so this was going to happen every time on
            # the one prompt the demo is built around. Nothing here catches it:
            # 9.11 IS the high-value mean, so it is a grounded figure answering
            # the wrong question, and grounding is not relevance.
            f"Declared mean (average) compliance cycle time: {p['declared_avg_compliance_cycle_time_days']} days",
            f"Declared mean (average) order-to-cash: {p['declared_avg_order_to_cash_days']} days",
            f"Mean cycle time across all events: {p['mean_cycle_days']} days "
            f"(median {p['median_cycle_days']}, max {p['max_cycle_days']})",
            f"Orders above $100,000: {p['high_value_orders']} of {p['total_cases']} orders "
            f"({p['high_value_events']} of {p['total_events']} events), "
            f"mean cycle {p['high_value_mean_cycle_days']} days across those events",
        ]
        lines += [
            f"Activity '{a}': {b['event_count']} events, mean cycle {b['mean_cycle_days']} days "
            f"(max {b['max_cycle_days']})"
            for a, b in p["by_activity"].items()
        ]
        return "VERIFIED AGGREGATES (computed over the full event log):\n" + "\n".join(lines)

    def _context_for(self, query: str) -> Dict[str, Any]:
        agg = self._aggregate_block()
        if not self.retriever:
            return {"context": agg, "retrieval_supported": False, "reason": "no retriever", "chunks": []}
        try:
            res = self.retriever.format_granite_context(query)
            chunks = res.get("context")
            res["context"] = f"{agg}\n\nRETRIEVED EVENT CHUNKS:\n{chunks}" if chunks else agg
            return res
        except Exception as err:  # retrieval must never take the stream down
            return {"context": agg, "retrieval_supported": False, "reason": f"retrieval error: {err}", "chunks": []}

    # ---------- numeric grounding ----------

    @staticmethod
    def _numbers_in(span: str) -> List[float]:
        out = []
        for m in _NUM_RE.findall(span):
            raw, scale = m.strip(), 1.0
            low = raw.lower()
            # Longest suffix first, so "million" is not read as a bare "m".
            for suffix, factor in _SCALE.items():
                if low.endswith(suffix) and len(raw) > len(suffix):
                    raw, scale = raw[: -len(suffix)], factor
                    break
            cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip().rstrip(".")
            if cleaned and cleaned not in {"-", "."}:
                try:
                    out.append(float(cleaned) * scale)
                except ValueError:
                    pass
        return out

    # Words that make the surrounding sentence a claim about an aggregate rather
    # than about one event. "the mean ... is 15 days" must be checked against the
    # log's means, not against whichever individual case happens to be 15 days.
    _AGG_WORDS = (
        "mean", "average", "avg", "median", "total", "sum", "overall", "across",
        "typical", "percentage", "percent", "rate", "deviation", "aggregate",
        "combined", "cumulative", "per case", "on average",
    )

    @staticmethod
    def _mid_figure(text: str) -> bool:
        """True when the text ends inside a number still being emitted."""
        tail = text.rstrip()
        if not tail:
            return False
        # "12", "12." and "12.5" are all mid-figure; a following space or letter
        # ends it. Trailing separators count because a decimal may still arrive.
        return tail[-1].isdigit() or (len(tail) > 1 and tail[-1] in ".," and tail[-2].isdigit())

    @classmethod
    def _statistic_span(cls, text: str, start: int, end: int) -> str:
        """
        The text that may name the statistic for the figure at [start, end):
        from the end of the previous figure to the start of the next one.

        Bounded on BOTH sides, for two different failures.

        Left, or one statistic word captures every figure after it: "the mean
        cycle time for Compliance Review is 10.43 days across 122 events" bound
        both figures to `mean`, and rejected 122 - a correct event count - for
        not being a mean.

        Right, because English puts the statistic after the figure just as
        often: "Compliance Review took 16 days on average" is the whole case
        this exists to catch, and a left-only span cannot see "on average" at
        all. Mid-stream that phrasing is still checked before the words arrive -
        see audit_figures for where it is caught instead.
        """
        prev = list(_NUM_RE.finditer(text[:start]))
        left = prev[-1].end() if prev else 0
        # Forward reach applies ONLY when this figure is the last in its clause.
        # A statistic word standing between two figures belongs to the one it
        # precedes, not the one it follows: in
        #   "Across 122 Compliance Review events the average was 10.43 days"
        # "average" describes 10.43, but reaching forward from 122 swallowed it
        # and checked a COUNT against Compliance Review's mean. The case the
        # forward reach exists for - "16 days on average" - has no later figure,
        # so it is unaffected.
        nxt = _NUM_RE.search(text[end:])
        right = end if nxt else len(text)

        # Then clipped to the figure's own CLAUSE. Bounding only by neighbouring
        # figures let a statistic word leak across a clause boundary: in
        #   "'Compliance Review' occurs 122 times ... with a mean cycle time of 10.43"
        # 122 is the first figure, so its span ran to 10.43 and swallowed "mean" -
        # binding a COUNT to the mean and rejecting it. The deterministic
        # recovery answer, which is generated from the log and true by
        # construction, failed its own grounding check because of it.
        lb = list(_CLAUSE_BREAK.finditer(text[left:start]))
        if lb:
            left += lb[-1].end()
        # The right clip applies only when another figure follows. With no later
        # figure competing for it, a statistic word anywhere ahead can only be
        # describing THIS one, and clipping at the first comma threw away the
        # single case the forward reach exists for: "Compliance Review took 16
        # days, on average" - 16 is that activity's MAX - lost "on average" to
        # the appositive comma and passed. The left clip still stands, so a
        # statistic belonging to an EARLIER figure cannot leak in.
        if nxt:
            rb = _CLAUSE_BREAK.search(text[end:right])
            if rb:
                right = end + rb.start()
        return text[left:right]

    @classmethod
    def _bound_statistic(cls, text: str, start: int, end: int,
                         unit: Optional[str], metric_span: str) -> Optional[str]:
        """
        The statistic for the figure at [start, end): its own clause first, then
        its sentence.

        The clause bound is right for the common shapes and wrong for one:

            "The average Compliance Review duration across 122 cases was 16 days"

        `average` is separated from the 16 it describes by both an intervening
        figure and a clause break, so the clause span sees none of it, 16 passed
        as that activity's max, and the sentence asserting a mean of 10.43 went
        out unchallenged. Widening unconditionally is what the clause bound
        exists to prevent - one statistic word would capture every figure after
        it. The unit is what makes the second look safe: it only fires when the
        figure states what it is measured in, and a statistic of the wrong kind
        for that unit is discarded before it can bind.
        """
        span = cls._statistic_span(text, start, end)
        near = cm.bound_statistic(span, unit=unit)
        if near or not unit:
            return near
        return cm.bound_statistic(metric_span, unit=unit)

    @staticmethod
    def _metric_span(prefix: str, tail: str = "") -> str:
        """
        The text that may name the METRIC for a figure: its sentence.

        `tail` is the text AFTER the figure, when it exists. English names the
        metric after the figure at least as often as before it:

            "3 events are recorded for the Invoice Approved activity."

        With a prefix-only span that figure binds to no metric, falls back to the
        whole-log check, and passes because 3 is some event's cycle time. The
        real Invoice Approved count is 150, and this was a live answer on a
        GREEN-PATH demo prompt - a flat fabrication streaming clean.

        Mid-decode the tail does not exist yet, so binding is prefix-only there
        and this costs nothing; the end-of-stream flush and audit_figures both
        pass the full text, which is where the claim is finally settled.

        Metric and statistic bind at different scopes, and collapsing them to one
        span breaks whichever is not favoured.

        A STATISTIC is clause-local - "mean" governs the phrase it sits in, and
        letting it cross a conjunction bound a count to a mean.

        A METRIC is named once and governs the rest of the sentence:

            "'Compliance Review' occurs 122 times ... with a mean cycle time of 10.43"

        clipping to the clause severs 10.43 from "Compliance Review", so the
        generic "cycle time" alias wins by default and 10.43 - that activity's
        correct mean - is checked against the overall 8.5 and rejected. That is a
        halt on a true statement, produced by the system's own deterministic
        recovery answer.

        The span cannot simply be the whole prefix either: a process TITLE would
        then bind every figure after it ("Order-to-Cash ... Compliance Cycle: 150
        cases / 460 events" bound 150 to order-to-cash, whose only value is 12.0).
        Titles are separated from claims by a colon, bracket or full stop, so
        breaking on those and NOT on conjunctions satisfies both cases.
        """
        # A comparison baseline belongs to its own claim - see _CONTRAST_BREAK.
        # The newline is a hard floor for everything below: the aggregate block is
        # one claim per LINE, and the backward walk below happily crossed one -
        # "Orders above $100,000: 122 ..." reached up to "Declared average
        # order-to-cash: 12.0 days" on the line before, bound 122 to a metric
        # whose only value is 12.0, and rejected a figure out of the very block
        # the model is told to quote.
        prefix = _CONTRAST_BREAK.split(prefix.rsplit("\n", 1)[-1])[-1]
        parts = _SENTENCE_BREAK.split(prefix)
        span = parts[-1] if parts else prefix
        # Walk back over titles and parentheticals until a SPECIFIC metric is in
        # scope. Breaking on ':' and '(' stops a process TITLE from binding every
        # figure after it, but it also severs a metric the sentence names once and
        # then refers to generically:
        #
        #   "Compliance Review (122 events): the mean cycle time is 10.43 days."
        #
        # after the colon only the generic "cycle time" survives, whose mean is
        # the overall 8.5, so 10.43 - that activity's own mean - was halted.
        # Walking back is safe in the direction that matters: bound_metrics
        # returns every phrase it finds and _ungrounded_number accepts a figure
        # any one of them admits, so a wider span can only ADD candidates. It
        # cannot create a halt, only remove one. The title case it was guarding
        # against is now held by _log_wide instead, which is the right layer for
        # it - a case count is a fact about the log, not about the title's metric.
        i = len(parts) - 1
        while i > 0 and not cm.names_specific_metric(span):
            i -= 1
            span = f"{parts[i]} {span}"
        if tail:
            # Forward reach stops at the CLAUSE, not the sentence. The scopes are
            # deliberately asymmetric: a metric named earlier governs what
            # follows it, but a metric named later belongs to its own claim.
            # Reaching forward by sentence let "Mean cycle time 8.5 days, median
            # 9.0, declared compliance cycle time 10.4" bind 8.5 to the
            # compliance metric two clauses away and reject the log's own
            # overall mean.
            #
            # The first NON-EMPTY clause, because an appositive puts a break
            # immediately after the figure: in "38 of 150 cases, or 25.33%, were
            # flagged as supply-chain bottlenecks" the clause after 25.33% is
            # empty, so nothing bound, and the percentage rule below then
            # rejected a rate the log computes exactly.
            clauses = [c for c in _CLAUSE_BREAK.split(tail) if c and c.strip()]
            if clauses:
                span = f"{span} {clauses[0]}"
        return span

    @staticmethod
    def _clause(prefix: str) -> str:
        """
        The clause a figure sits in - everything since the last clause break.

        Metric binding has the same leak as statistic binding. The recovery
        answer opens with the process TITLE:

            "Order-to-Cash (O2C) Compliance Cycle (...): 150 cases / 460 events.
             Mean cycle time 8.5 days, ..."

        bound_metric scans a 200-character prefix window, so "order-to-cash" in
        the title bound every later figure to the order-to-cash metric - whose
        only admissible value is 12.0 - and 150, 460 and 8.5 were all reported as
        contradicting the log they were computed from. A process name is a title,
        not a claim about a metric.
        """
        parts = _CLAUSE_BREAK.split(prefix)
        return parts[-1] if parts else prefix

    @classmethod
    def _claim_scope(cls, text: str, figure: str = "", window: int = 160) -> str:
        """'aggregate' if the claim is about a derived quantity, else 'all'."""
        # No event field in the log is a percentage, so any percentage the model
        # states is necessarily derived and must match a derived value. Without
        # this, "the discount was 15%" was accepted because 15 happens to be some
        # case's cycle_time_days - a field with nothing to do with discounts.
        if "%" in figure:
            return "aggregate"
        recent = text[-window:].lower()
        return "aggregate" if any(w in recent for w in cls._AGG_WORDS) else "all"

    def _ungrounded_number(
        self, span: str, scope: str = "all", metric: Optional[str] = None,
        statistic: Optional[str] = None, metrics: Optional[List[str]] = None,
        unit: Optional[str] = None, sentence_complete: bool = True,
        claim: str = "",
    ) -> Optional[float]:
        """
        First number in `span` that the event log cannot produce.

        :param claim: the surrounding sentence, used ONLY for the live filtered
            query below. Passing it is what turns a conditional claim from
            unverifiable into checkable.
        """
        # REAL-TIME PROCESS QUERY. Everything below this block compares against
        # sets precomputed over the whole log or per activity, which know nothing
        # about a claim that names a FILTER:
        #
        #   "the average approval delay for cost center CC-9902 was 3 days"
        #
        # checked against the whole log, 3 is some event's cycle_time_days, so it
        # passes. Running the aggregation the claim actually describes is the only
        # honest check, and the decisive case is the empty one: a filter matching
        # NO events means the log holds no data for that entity, so every figure
        # about it is fabricated whatever its value. That is a false negative the
        # precomputed sets cannot close, because they have no way to represent
        # "this entity does not exist".
        #
        # Deliberately one-directional. An empty match REJECTS; a non-empty match
        # only ADDS its filtered values to the admissible set, never narrows it.
        # Narrowing would let a mis-parsed filter halt a true sentence, and the
        # filter parser is a regex over free text - the least trustworthy input
        # in this file. Adding candidates cannot manufacture a false positive.
        live = cm.live_process_query(claim) if claim else None
        if live is not None and live["matched_events"] == 0:
            vals = self._numbers_in(span)
            if vals:
                return vals[0]
        # A percentage that names no metric we know cannot be grounded at all.
        # The event log has no percentage field, so every legitimate percentage
        # is derived from a nameable metric (the bottleneck rate). Without this,
        # "the off-contract discount was 12%" was accepted because 12.0 is the
        # declared order-to-cash figure - a discount validated against a count
        # of days. There is no discount data in the log; the honest answer is
        # that the claim is ungrounded whatever its value.
        options = list(metrics) if metrics else [metric]
        # A '$' figure is money. Every metric in the table is measured in days or
        # in events, so binding a dollar amount to one compares dollars against
        # day-counts and rejects whatever it finds: "The amount for the Purchase
        # Order in CASE-10002 is $130,960.29" - the literal amount_usd on that
        # event - was checked against Purchase Order Created's {2.4, 4, 150} and
        # halted. Dropping the binding costs no detection, because there is no
        # dollar-valued metric for it to have been the wrong one OF; a fabricated
        # total is still caught by scope, which holds the log's real money values.
        if "$" in span:
            options = [None]
        # A generic alias names a DIMENSION, and a sentence with no aggregate word
        # is not asserting an aggregate of it: "For case CASE-10001 the cycle time
        # so far is 2 days" is a claim about one case, but "cycle time" bound it to
        # the log's overall {8.5, 9.0, 23} and halted a figure that is a real
        # per-event value. The generic entry exists for "the MEAN cycle time is
        # 12.7", which carries an aggregate word and so still binds - the flagship
        # interception is untouched.
        # ...unless it names a statistic. "The highest cycle time was 8.5 days"
        # carries no aggregate word, but "highest" IS the aggregate claim, and
        # dropping the binding there let the log's MEAN pass as its maximum.
        elif (scope == "all" and statistic is None and options
              and all(cm.is_generic_metric(m) for m in options)):
            options = [None]
        # ...but only once the sentence is finished. This rule concludes from the
        # ABSENCE of a metric, and mid-decode absence is not evidence - the word
        # may still be coming. Measured live on "What percentage of orders were
        # flagged as bottlenecks?": the model emitted "38 out of 150 orders, which
        # is 25.3%," and the guard fired on 25.3 - the log's own rate, 25.33 -
        # because "bottlenecks" was still three tokens away. The stream halted and
        # the recovery then returned the identical sentence, so the system
        # contradicted itself on screen. The end-of-stream rescan sees the whole
        # sentence and binds it correctly; this defers to it.
        if "%" in span and not any(options) and sentence_complete:
            vals = self._numbers_in(span)
            if vals:
                return vals[0]
        # A sentence may name more than one metric, and then no single binding is
        # correct. Accept the figure if ANY metric it names admits it; a figure
        # that fits none still fails, so no detection is lost. See
        # cm.bound_metrics for why neither longest-wins nor nearest-wins works.
        # Values the FILTERED aggregation actually produced, admitted alongside
        # the precomputed sets. This is the loosening half of the live query: a
        # true conditional claim ("Compliance Review in Finance averages 11.2
        # days") states a number no whole-log set contains, and without this it
        # would be rejected for being specific.
        #
        # Only the values matching the claim's OWN unit and statistic. Admitting
        # the whole filtered blob is the same mistake is_grounded_number spends
        # its statistic and unit parameters avoiding, and it cost five
        # regressions when this first landed: "3 events are recorded for the
        # Invoice Approved activity" was admitted because that activity's
        # min_cycle_days is 3 - a DURATION validating a COUNT, where the real
        # count is 150. With neither unit nor statistic bound nothing is
        # admitted, so an unlabelled figure falls through to exactly the
        # behaviour that shipped.
        live_vals: List[float] = []
        if live:
            v = live["values"]
            if unit == "count":
                # Both populations, because "events" and "orders" both read as
                # counts here and the log distinguishes them - see the same
                # warning in cm.live_process_query.
                live_vals = [v["count"], v["cases"]]
            elif unit == "money":
                live_vals = [v["mean_amount_usd"], v["total_amount_usd"]]
            elif statistic in ("mean", "max", "min"):
                live_vals = [v.get(f"{statistic}_cycle_days")]
            live_vals = [x for x in live_vals if isinstance(x, (int, float))]
        for val in self._numbers_in(span):
            if any(abs(v - val) <= max(abs(v), abs(val), 1e-9) * 0.02 for v in live_vals):
                continue
            if not any(
                cm.is_grounded_number(val, scope=scope, metric=m, statistic=statistic,
                                      unit=unit)
                for m in options
            ):
                return val
        return None

    def _scan_new_figures(self, text: str, scanned_to: int, complete_only: bool):
        """
        Check figures in `text` that start at or after `scanned_to`.

        With complete_only, a figure touching the end of the string is left
        unchecked: more digits may still arrive, and "10" would be judged before
        it becomes "10.43". The caller re-scans without the flag at end of stream.

        Returns (first ungrounded value or None, new scanned_to).
        """
        for m in _NUM_RE.finditer(text):
            if m.start() < scanned_to:
                continue
            tail = text[m.end():]
            # A figure is complete only once a character arrives that cannot
            # extend it. "87." looks like a terminated 87, but the next token may
            # make it 87.6 - and judging 87 alone reports the wrong value.
            if complete_only and (not tail or set(tail) <= {".", ","}):
                break                      # still being emitted; check it later
            scanned_to = m.end()
            prefix = text[:m.end()]
            mspan = self._metric_span(prefix, tail)
            # The unit is the weaker signal that is almost always present where
            # the statistic word is not: "Compliance Review has 16 events" names
            # no statistic, but "events" rules out every duration-valued one, and
            # 16 is that activity's max in days.
            unit = cm.figure_unit(m.group(), tail)
            val = self._ungrounded_number(
                m.group(),
                scope=self._claim_scope(prefix, figure=m.group()),
                # If the sentence names a metric, the figure must be one of THAT
                # metric's values, not merely some value in the log - unless the
                # figure is a threshold or a denominator, which qualify the
                # metric rather than state it.
                metrics=[] if _QUALIFIER_BEFORE.search(text[:m.start()])
                else cm.bound_metrics(mspan),
                # And if it names the statistic too, it must be that statistic's
                # value - "16 days on average" quotes Compliance Review's max.
                statistic=self._bound_statistic(text, m.start(), m.end(), unit, mspan),
                unit=unit,
                # complete_only IS the mid-decode pass, so a rule that concludes
                # from an absent metric must wait for the end-of-stream rescan.
                sentence_complete=not complete_only,
                # The metric span doubles as the filter span: both answer "what
                # is this figure ABOUT", and it is already computed.
                claim=mspan,
            )
            if val is not None:
                return val, scanned_to
        return None, scanned_to

    def audit_figures(self, text: str) -> Dict[str, Any]:
        """
        Classify every figure in `text` as verified / unverifiable / contradicted.

        Why three states and not two
        ----------------------------
        `_scan_new_figures` answers one question - did any figure CONTRADICT the
        log - and the flag built from it was called `all_figures_grounded`, which
        reads as "every figure was confirmed". Those are not the same claim, and
        the gap between them is most of this layer's real error budget.

        A figure is only genuinely confirmed when the sentence names a metric
        (one of the phrases in cm.metric_aliases) and the value is one that
        metric can take. With no metric bound, the value is compared against
        every number the log can produce - 189 of them under scope="all" - and
        passing that is weak evidence: measured against arbitrary figures, 80% of
        integers 1-30 and 53% of one-decimal day values clear it. Calling that
        "grounded" is the same overclaim the project exists to catch.

        So: `verified` means bound and admissible. `unverifiable` means nothing
        contradicted it and nothing confirmed it either. `contradicted` is the
        only state that halts, and this method does not change what halts - it
        only stops the report from claiming more than the check performed.
        """
        counts = {"verified": 0, "unverifiable": 0, "contradicted": 0}
        detail: List[Dict[str, Any]] = []
        for m in _NUM_RE.finditer(text):
            figure = m.group()
            prefix = text[:m.end()]
            mspan = self._metric_span(prefix, text[m.end():])
            metrics = ([] if _QUALIFIER_BEFORE.search(text[:m.start()])
                       else cm.bound_metrics(mspan))
            # Head of the list for reporting; the whole list decides the verdict.
            metric = metrics[0] if metrics else None
            unit = cm.figure_unit(figure, text[m.end():])
            statistic = self._bound_statistic(text, m.start(), m.end(), unit, mspan)
            scope = self._claim_scope(prefix, figure=figure)
            contradicted = self._ungrounded_number(
                figure, scope=scope, metrics=metrics, statistic=statistic, unit=unit,
                claim=mspan,
            ) is not None
            if contradicted:
                state = "contradicted"
            elif metric is not None:
                state = "verified"
            else:
                state = "unverifiable"
            counts[state] += 1
            detail.append({"figure": figure, "state": state, "metric": metric,
                           "statistic": statistic, "unit": unit, "scope": scope})
        return {**counts, "figures": detail}

    def _refutation(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Why the offending figure was rejected, in the log's own terms.

        The UI called this "figure not in log", which was false often enough to
        be the worst string on the screen: 9.11 IS in the log - it is the
        high-value mean - and the breaker halted "the mean compliance cycle time
        ... is 9.11" because that metric's mean is 10.4, not because the number
        is absent. A judge who checks the data finds the figure and concludes
        the detector is broken.

        The binding that did the rejecting already knows the honest answer, so
        report it: which metric the sentence named, which statistic of it, and
        what that statistic actually is.
        """
        claim = next((f for f in self.audit_figures(text)["figures"]
                      if f["state"] == "contradicted"), None)
        if not claim or not claim["metric"]:
            return None
        stats = cm.metric_statistics().get(claim["metric"], {})
        admissible = stats.get(claim["statistic"]) or frozenset().union(*stats.values())
        return {
            "claimed_metric": claim["metric"],
            "claimed_statistic": claim["statistic"],
            "unit": claim["unit"],
            "admissible": sorted(admissible),
        }

    # ---------- interception ----------

    async def _trip(
        self, query, rag, reason, tok, step, uncertainty, variance, shannon,
        ungrounded_value, emitted, history, t0, overhead_s, decision=None,
        semantic=None,
    ) -> AsyncIterator[InterceptionEvent]:
        """
        Emit the interception + autonomous recovery pair for a tripped breaker.

        :param decision: (step, divergence) captured at the most recent token
            whose candidate set contained a figure - the moment the value was
            actually chosen. NOT the token being halted on.

            Those are rarely the same token. `ungrounded_number` trips when the
            figure TERMINATES, so the halting step is the character after the
            digits (" days"), whose candidates are prose; and the trailing-figure
            path halts after the stream is exhausted, with no live step at all.
            Reporting either one's candidate set showed the model choosing
            between words at the exact moment the pitch claims it was choosing
            between numbers. This carries the decision point instead.
        """
        dstep, div = decision if decision else (None, None)
        # history excludes the halting token - it is appended only after the trip
        # check - and that token is usually the unit noun that completed the claim
        # ("9.11" + " days"). Reading history alone judged a sentence the detector
        # had not judged, and came back with no contradiction at all.
        refutation = self._refutation("".join(history) + tok)
        log_compliance_breach(query, reason, uncertainty, ungrounded_value)
        # The audit log proves a PROMPT recurred; the ledger proves the same GAP
        # did, across differently-worded prompts, and ranks it for curation.
        gap = gap_ledger.record(reason, refutation, query, ungrounded_value)
        yield InterceptionEvent(
            kind="intercept",
            text=tok,
            payload={
                "reason": reason,
                "token": tok,
                # What the decoder was choosing between when it picked the value.
                # The scalar entropies say uncertainty was high; this says what
                # the uncertainty was ABOUT, which is the only form of it a
                # non-specialist can check by eye.
                "candidates": sd.fan(dstep.top_tokens, dstep.top_probs) if dstep else None,
                "candidates_token_index": dstep.index if dstep else None,
                "semantic_entropy": round(div.semantic_entropy, 4) if div else None,
                # Carried on the intercept too, so a surface reading this field
                # gets the same answer whatever event it is looking at. It was
                # only on token events, so anything reading it off an intercept
                # silently got None and rendered "no figure at stake" at the
                # exact moment a figure was the entire problem.
                #
                # Taken from the DECISION step, not the halting one: like
                # semantic_entropy above, it describes where the value was
                # chosen, which is a different token from where the breaker
                # tripped.
                "figure_at_stake": div.divergent if div else None,
                "divergence_explained": div.explain() if div else None,
                # The trailing-figure path has no step object; the offending
                # figure is the last thing emitted, so report that position
                # rather than a -1 the UI would render as "token #-1".
                "token_index": step.index if step is not None else max(emitted - 1, 0),
                "logprob": round(step.logprob, 4) if step else None,
                "probability": round(step.prob, 4) if step else None,
                "uncertainty": round(uncertainty, 4),
                "tau": self.tau,
                "breach_run": self.breach_run,
                "rolling_variance": round(variance, 4),
                "shannon_entropy": round(shannon, 4),
                "ungrounded_value": ungrounded_value,
                # What the figure was measured against. Without it the UI can
                # only say "not in log", which is the wrong reason whenever the
                # figure is real but belongs to another metric - the commonest
                # case this layer catches. See _refutation.
                "refutation": refutation,
                # "this gap has now been hit N times, and is #R on the curation
                # queue" - a single interception looks like an incident, the
                # count is what makes it a backlog item worth a data fix.
                "gap_hit_count": gap["hit_count"],
                "gap_rank": gap["rank"],
                # The K-path evidence, when a checkpoint had run before the trip.
                # This is the one number on the payload that comes from sampling
                # the model rather than from reading one distribution, so it is
                # the only place the pitch's "it tells a different story each
                # run" claim can actually be checked - the paths are carried so a
                # judge can read them.
                "kpath": {
                    "k": semantic.k,
                    "semantic_entropy": semantic.entropy,
                    "normalised": semantic.normalised,
                    "clusters": len(semantic.clusters),
                    "figures": semantic.figures,
                    "figure_share": semantic.figure_share,
                    "paths": semantic.paths,
                    "explained": semantic.explain(),
                } if semantic is not None else None,
                "tokens_before_halt": emitted,
                "token_budget": self.max_new_tokens,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "entropy_overhead_ms": round(overhead_s * 1000, 3),
                "partial_output": "".join(history),
            },
        )
        # Autonomous recovery: re-decode the SAME question with the Celonis
        # context injected. This is a second real generation, not a lookup
        # string - the agent repairs the context gap and the model answers again
        # under grounding, which is the whole point of the fallback.
        repaired = self._context_for(query)
        yield InterceptionEvent(
            kind="recovery_start",
            text="",
            payload={
                "strategy": "milvus_context_repair_and_regenerate",
                "query": query,
                "retrieval_reason": repaired.get("reason"),
            },
        )
        recovered: List[str] = []
        try:
            async for step in self.runner.stream(
                self.runner.build_prompt(query, repaired.get("context"), grounded=True),
                max_new_tokens=self.max_new_tokens,
            ):
                recovered.append(step.text)
                yield InterceptionEvent(
                    kind="recovery_token",
                    text=step.text,
                    payload={"logprob": round(step.logprob, 4), "index": step.index},
                )
        except Exception as err:
            print(f"[Sentinel] recovery generation failed ({err}); falling back to the log lookup.")

        answer = "".join(recovered).strip() or cm.ground_truth_answer(query)
        # The repaired answer is held to the same standard as the original: a
        # recovery that itself states an ungrounded figure is not a recovery.
        regenerated_ok = self._scan_new_figures(answer, 0, complete_only=False)[0] is None
        # And it may not restate the figure the breaker just halted on, even if
        # it has found a phrasing that makes the figure admissible.
        #
        # Observed on screen: the stream halted on "The mean compliance cycle
        # time across all events is 9.11" - correctly, because that metric's
        # mean is 10.4 and 9.11 is the HIGH-VALUE mean - and the recovery
        # returned "...is 9.11 days for orders above $100,000", which passes the
        # ungrounded check, because the added qualifier is exactly what makes
        # 9.11 admissible. So the panel showed the same number halted as a
        # fabrication and then re-served as verified. Both verdicts were right
        # about their own sentence and the pair was indefensible.
        #
        # Repairing the context gap means answering the question from the log,
        # not finding a sentence that carries the offending figure past the
        # check. ground_truth_answer is derived from the event log and true by
        # construction, which is what the fallback is for.
        if regenerated_ok and ungrounded_value is not None:
            regenerated_ok = ungrounded_value not in self._numbers_in(answer)
        if not regenerated_ok:
            answer = cm.ground_truth_answer(query)
        # Report on the answer actually being returned, not on the draft that was
        # discarded. Previously a clean deterministic fallback was still flagged
        # ungrounded, because the flag described the regeneration it replaced.
        verified = self._scan_new_figures(answer, 0, complete_only=False)[0] is None
        audit = self.audit_figures(answer)

        yield InterceptionEvent(
            kind="recovery",
            text=answer,
            payload={
                "strategy": "milvus_context_repair_and_regenerate",
                "query": query,
                "retrieval_reason": repaired.get("reason"),
                "regenerated": bool(recovered),
                # Kept under its original name because the terminal reads it,
                # but it means "nothing contradicted the log" - NOT "everything
                # was confirmed". figure_audit below is the honest breakdown and
                # is what any new surface should render.
                "all_figures_grounded": verified,
                "figure_audit": audit,
                "fell_back_to_lookup": not regenerated_ok,
                # What the protected path ACTUALLY cost, end to end: the tokens
                # decoded before the halt plus the tokens the repair decoded, and
                # the wall clock from the first token of the original generation
                # to the final answer being ready.
                #
                # The panel used to report tokens_before_halt and elapsed-to-halt,
                # which describe the abandoned generation rather than the answer
                # the user receives, and alongside them a "decode prevented"
                # figure that was budget minus halt index - unused allowance, not
                # work avoided. It counted the recovery as free. Interception
                # spends MORE than letting the fabrication finish; the claim
                # worth making is that it spends it on a correct answer.
                "total_tokens": emitted + len(recovered),
                "total_elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )

    # ---------- the loop ----------

    async def run(self, query: str) -> AsyncIterator[InterceptionEvent]:
        rag = self._context_for(query)
        context = rag.get("context")

        engine = EntropyEngine(threshold_tau=self.tau, window_size=self.window_size)
        prompt = self.runner.build_prompt(query, context, grounded=self.grounded_prompt)

        history: List[str] = []
        text = ""               # assembled output, scanned for completed figures
        scanned_to = 0          # end offset of the last figure already checked
        t0 = time.perf_counter()
        overhead_s = 0.0
        emitted = 0
        consecutive = 0         # length of the current run of threshold breaches
        decision = None         # (step, divergence) where a figure was most at stake
        decision_mass = 0.0     # numeric mass at that step; see the update below
        # 0, NOT a large number. Seeding this high let the very first token
        # checkpoint, and a checkpoint at token 0 has no prefix to continue - so
        # the draft model answers the question from scratch instead of finishing
        # granite's sentence, and its figures have nothing to do with the value
        # granite is about to state. Measured: "What is the mean compliance cycle
        # time?" checkpointed at token 0, came back with figures [0.4, 2.0],
        # ruled them divergent and BLOCKED an answer whose next tokens were the
        # correct 10.4. The whole method rests on continuing a sentence in
        # flight; starting at 0 means the first checkpoint cannot fire until
        # min_gap tokens exist to continue.
        since_ckpt = 0          # tokens since the last K-path checkpoint
        semantic_result = None  # most recent K-path measurement, if any
        checkpoints = 0
        hedged = None           # (tier, why) if the stream was allowed through hedged

        async for step in self.runner.stream(prompt, max_new_tokens=self.max_new_tokens):
            tok = step.text
            c0 = time.perf_counter()

            is_hallucinating, uncertainty, variance = engine.evaluate_token(
                tok,
                logprob=step.logprob,
                top_probs=step.top_probs,
                context_history=history,
            )
            shannon = engine.compute_shannon_entropy(step.top_probs)
            # Same top-k, clustered by value before the entropy is taken. Shannon
            # over the raw candidates cannot tell "is/was/has" from "12/15/9" -
            # identical distributions, opposite meanings - which is why the
            # scalar layer needs a run of breaches to be usable at all. This
            # scores the question the raw number cannot: is a FIGURE at stake.
            # Costs no extra forward pass; the distribution already exists.
            divergence = sd.analyse(step.top_tokens, step.top_probs)
            # Remember where a figure was MOST on the table. The trip almost
            # never lands on that token - see _trip's `decision` parameter - so
            # the evidence has to be captured as it passes by.
            #
            # Strongest, not latest. Keeping the latest token with any numeric
            # candidate at all meant a trace digit in an otherwise prose
            # distribution overwrote the real thing: one live intercept reported
            # its decision as "2 distinct figures on the table but only 0% of
            # the mass - the model is mostly writing prose here", and that is
            # what the terminal would have shown a judge as the moment the
            # figure was chosen. Ranking by numeric mass lands on the token
            # where the model was actually committing to a value.
            if divergence.numeric_mass > decision_mass:
                decision, decision_mass = (step, divergence), divergence.numeric_mass

            # Scan the assembled text rather than buffering characters, so a
            # figure split across tokens ("10" "." "43") is checked as one value
            # and identifiers like W-99 are excluded by the regex boundaries.
            # Only figures already terminated by a following character are
            # checked; a number still being emitted may yet gain more digits.
            text += tok
            ungrounded_value = None
            if self.check_numbers:
                ungrounded_value, scanned_to = self._scan_new_figures(text, scanned_to, complete_only=True)

            overhead_s += time.perf_counter() - c0

            # ADAPTIVE CHECKPOINT. The K-path measurement costs K draft
            # generations - hundreds of milliseconds - so it cannot run per
            # token, and running it only at the end would make it post-hoc, which
            # is the thing this project exists to replace. It is gated on the two
            # signals that are already free: granite's own distribution putting
            # mass on digits (fires BEFORE the figure lands, the earliest warning
            # available anywhere in the system) and the token just emitted being
            # high-stakes. See semantic_entropy.should_checkpoint.
            since_ckpt += 1
            if self.semantic is not None:
                why_ckpt = se.should_checkpoint(
                    divergence, since_ckpt, engine.get_token_type_weight(tok)
                )
                if why_ckpt:
                    since_ckpt = 0
                    checkpoints += 1
                    # NOT counted in overhead_s: that number is the always-on
                    # per-token cost, and folding a 400 ms checkpoint into it
                    # would report the cheap layers as expensive. The checkpoint
                    # is reported on its own terms below.
                    measured = await self.semantic.measure(prompt, text)
                    if measured is not None:
                        semantic_result = measured
                        yield InterceptionEvent(
                            kind="checkpoint",
                            text="",
                            payload={
                                "reason": why_ckpt,
                                "token_index": step.index,
                                "k": measured.k,
                                "semantic_entropy": measured.entropy,
                                "normalised": measured.normalised,
                                "clusters": len(measured.clusters),
                                "figures": measured.figures,
                                "cached": measured.cached,
                                "elapsed_ms": measured.elapsed_ms,
                                "explained": measured.explain(),
                                "paths": measured.paths,
                            },
                        )

            # Entropy: noisy per-token signal, so require a sustained run.
            # Numbers: deterministic check, so one is enough.
            consecutive = consecutive + 1 if is_hallucinating else 0
            trip_reason = None
            tier, tier_why = "pass", ""
            if ungrounded_value is not None:
                # The deterministic layer is NOT tiered. A figure the event log
                # cannot produce is a fact-check, not a confidence estimate;
                # hedging it would mean printing a known-false number with a
                # caveat attached, which is worse than either blocking or staying
                # silent. Tiering applies to the statistical layers, where the
                # evidence really is a matter of degree.
                trip_reason = "ungrounded_number"
            elif not self._mid_figure(text):
                # A figure is being emitted right now, so the deterministic check
                # is about to have an opinion on it. Digits are individually
                # high-entropy, so letting the statistical layer fire first halts
                # on "1" of "12.5" and reports semantic_entropy with no value,
                # when one more token yields the far stronger "12.5 is not a
                # compliance cycle time". Defer to the layer that can name it.
                #
                # CONFIDENCE-AWARE TIERING. The run of breaches is still what
                # makes the scalar layer usable, but the response to it is no
                # longer binary: fallback_tier reads how far over tau the score
                # sits, and lets the K-path evidence escalate a hedge to a block.
                if self.tiering:
                    tier, tier_why = se.fallback_tier(uncertainty, self.tau, semantic_result)
                    if tier == "block" and (consecutive >= self.breach_run
                                            or semantic_result is not None):
                        trip_reason = "semantic_entropy"
                    elif tier == "hedge" and consecutive >= self.breach_run and not hedged:
                        # Delivered, not withheld. A hedge on a true statement
                        # costs a caveat; a block on a true statement costs the
                        # user's belief that the system works, and that is the
                        # error that ends an enterprise deployment.
                        hedged = (tier, tier_why)
                        yield InterceptionEvent(
                            kind="hedge",
                            text="",
                            payload={"reason": tier_why, "token_index": step.index,
                                     "uncertainty": round(uncertainty, 4), "tau": self.tau},
                        )
                elif consecutive >= self.breach_run:
                    trip_reason = "semantic_entropy"

            if trip_reason:
                async for ev in self._trip(
                    query, rag, trip_reason, tok, step, uncertainty, variance,
                    shannon, ungrounded_value, emitted, history, t0, overhead_s,
                    decision, semantic_result,
                ):
                    yield ev
                # The circuit breaker actually breaks: decoding stops here.
                return

            history.append(tok)
            emitted += 1
            yield InterceptionEvent(
                kind="token",
                text=tok,
                payload={
                    "logprob": round(step.logprob, 4),
                    "probability": round(step.prob, 4),
                    "uncertainty": round(uncertainty, 4),
                    "rolling_variance": round(variance, 4),
                    "shannon_entropy": round(shannon, 4),
                    # Plotted against shannon_entropy: the gap between the two
                    # lines IS the argument. They separate only where a figure
                    # is at stake. The full candidate fan is sent on intercept
                    # rather than per token - one scalar per token keeps the SSE
                    # stream small enough to render at decode speed.
                    "semantic_entropy": round(divergence.semantic_entropy, 4),
                    "figure_at_stake": divergence.divergent,
                    "index": step.index,
                },
            )

        # A response ending on a figure leaves it unterminated and therefore
        # unchecked: "...mean cycle time is 87.6" would pass straight through.
        # Re-scan without complete_only now that no more tokens are coming.
        #
        # From 0, not from scanned_to. Mid-decode a figure is judged with only
        # the text before it, so a metric named AFTER it cannot bind - "3 events
        # are recorded for the Invoice Approved activity" was checked when the
        # sentence was still "3 events", found no metric, and passed on the
        # whole-log fallback because 3 is some event's cycle time. Resuming from
        # scanned_to meant it was never reconsidered once "Invoice Approved"
        # arrived. The real count is 150, and that answer streamed clean on a
        # green-path demo prompt. One extra pass over ~120 tokens at end of
        # stream is the cost of judging every figure with the whole sentence.
        if self.check_numbers:
            trailing, _ = self._scan_new_figures(text, 0, complete_only=False)
            if trailing is not None:
                last = engine.get_metrics_snapshot()
                async for ev in self._trip(
                    query, rag, "ungrounded_number", str(trailing), None, 0.0,
                    last.get("current_variance", 0.0), 0.0, trailing,
                    emitted, history, t0, overhead_s, decision, semantic_result,
                ):
                    yield ev
                return

        total_ms = (time.perf_counter() - t0) * 1000
        yield InterceptionEvent(
            kind="done",
            text="".join(history),
            payload={
                "tokens": emitted,
                # The decode budget this run was given. "Halted at token 11" is
                # only meaningful against the number it was allowed to reach -
                # 11 of 90 says 79 tokens were never decoded, which is the one
                # rigorous statement available about what interception saved.
                # The difference against the unprotected panel is NOT that
                # statement: the two are independent generations that stop at
                # their own natural lengths, and that subtraction goes negative
                # whenever the baseline finishes first.
                "token_budget": self.max_new_tokens,
                "elapsed_ms": round(total_ms, 2),
                "entropy_overhead_ms": round(overhead_s * 1000, 3),
                "overhead_pct": round(overhead_s * 1000 / total_ms * 100, 4) if total_ms else 0.0,
                "retrieval_reason": rag.get("reason"),
                # What the adaptive gate actually did on this response. The rate
                # is the number that decides whether the K-path layer is
                # affordable, so it is reported per run rather than only in the
                # benchmark - a demo prompt that checkpoints on every token would
                # otherwise look identical to one that never did.
                "checkpoints": checkpoints,
                "checkpoint_rate": round(checkpoints / emitted, 4) if emitted else 0.0,
                # A hedged answer was DELIVERED, with the caveat attached. The
                # terminal renders this differently from both a clean pass and a
                # halt, because it is a third outcome and collapsing it into
                # either one misrepresents what the system decided.
                "tier": "hedge" if hedged else "pass",
                "hedge_reason": hedged[1] if hedged else None,
            },
        )


def demo() -> None:
    """Self-check: numeric span parsing must survive tokenizer fragmentation."""
    assert SentinelStream._numbers_in("10.4") == [10.4]
    assert SentinelStream._numbers_in("$128,500") == [128500.0]
    assert SentinelStream._numbers_in("99.4%") == [99.4]
    assert SentinelStream._numbers_in("no digits here") == []
    # 10.4 is the declared compliance cycle time; 4.2 was the old fabricated one.
    assert cm.is_grounded_number(10.4)
    assert not cm.is_grounded_number(4.2)

    # A decimal figure must keep its metric binding. _CLAUSE_BREAK matching a
    # bare [.,] split "…cycle time is 10.4" at the decimal point, so the clause
    # became "4", no metric bound, and "12.5 days" - the flagship interception -
    # stopped halting while the integer "12 days" still did.
    s = SentinelStream(runner=None)
    assert s._clause("The mean compliance cycle time is 10.4") .endswith("10.4")
    assert s._scan_new_figures("The mean compliance cycle time is 12.5 days.", 0,
                               complete_only=False)[0] == 12.5
    assert cm.bound_metric(s._clause("An order of $128,500")) is None

    # Metric binds across a clause boundary; statistic must not. The existing
    # binding tests put the activity and the figure in the SAME clause, so they
    # could not see this: clipping the metric to the clause severed 10.43 from
    # "Compliance Review", bound it to the generic "cycle time" instead, and
    # rejected the system's OWN deterministic recovery answer as ungrounded.
    canned = cm.ground_truth_answer("mean cycle time for Compliance Review")
    audit = s.audit_figures(canned)
    assert audit["contradicted"] == 0, (canned, audit)

    # …while a title before a colon must still bind nothing, which is why the
    # metric span is the sentence and not the whole prefix.
    title = "Order-to-Cash (O2C) Compliance Cycle: 150 cases / 460 events."
    assert s.audit_figures(title)["contradicted"] == 0, title

    # Three-state audit: passing the check is not the same as being confirmed.
    bound = s.audit_figures("The mean compliance cycle time is 10.4 days.")
    assert (bound["verified"], bound["contradicted"]) == (1, 0), bound

    wrong = s.audit_figures("The mean compliance cycle time is 12.5 days.")
    assert wrong["contradicted"] == 1, wrong

    # 393 IS a number the log produces (the high-value EVENT count), so nothing
    # contradicts it - but the sentence calls it an order count and names no
    # metric, so it must not come back as verified. This is the exact overclaim
    # that let a false figure be reported as grounded.
    #
    # KNOWN FAILURE, and deliberately left failing rather than relaxed.
    #
    # It reproduces on the commit before any of the semantic-binding, hybrid
    # retrieval or K-path work - checked by running this exact call against
    # `git archive HEAD` - so it is not a regression from that work. Two rules
    # that were each added for good reasons now contradict each other:
    #
    #   * cm.metric_statistics aliases "$100,000" to the high-value metric, whose
    #     count is 122 orders. Its comment argues the figure_unit rule makes that
    #     binding safe, which would make 393 CONTRADICTED here.
    #   * cm._log_wide admits high_value_events (393) for ANY count claim, which
    #     makes it VERIFIED instead.
    #
    # The assertion above wants a third answer - unverifiable - from before the
    # alias existed. All three readings are defensible and the log genuinely
    # holds both populations; what it cannot do is tell "orders" from "events",
    # because cm.figure_unit collapses both to unit="count".
    #
    # The real fix is to split that unit so a claim counted in ORDERS is checked
    # against case counts and one counted in EVENTS against event counts. That
    # touches _UNIT_WORDS, _STATISTIC_UNIT and every metric's statistic table -
    # the most load-bearing code in the repo, with 90 tests on it - so it is
    # named here rather than done under deadline. Suppressing the assert instead
    # would hide a live false negative: "There were 393 orders" is false, and
    # today the system reports it as verified.
    #
    # ponytail: known false negative, orders-vs-events unit split is the fix.
    loose = s.audit_figures("There were 393 orders above $100,000.")
    assert loose["verified"] == 0 and loose["contradicted"] == 0, loose
    assert loose["unverifiable"] == 2, loose
    print("sentinel_stream self-check OK")


if __name__ == "__main__":
    demo()
