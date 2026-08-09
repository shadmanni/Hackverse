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

import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import celonis_metrics as cm
import semantic_divergence as sd
from entropy_engine import EntropyEngine

# A quantitative claim: a standalone number, optionally with $ , . and %.
#
# The lookarounds exclude digits embedded in enterprise identifiers - W-99,
# CC-9999, CASE-10298, SOX-404. Those are entity references, not figures, and
# treating them as claims caused a false interception on a response that was
# correctly refusing to answer: Granite said "...for warehouse node W-99" and
# the "99" was scored as a fabricated number.
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])"      # not preceded by identifier characters
    # The digit/separator run must END on a digit. As [\d,]* it swallowed a
    # trailing sentence comma - "above $100,000, the mean..." matched
    # "$100,000," - and the parsed value was then checked as though the claim
    # were about the comma's clause. Harmless-looking, but it was reporting the
    # log's own threshold as a contradiction.
    r"-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?"
    r"(?![A-Za-z0-9_\-])"        # not followed by identifier characters
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
_QUALIFIER_BEFORE = re.compile(
    r"(?:\d[\d,.]*\s+of"
    r"|above|below|over|under|exceeding|more than|less than|at least|at most)"
    r"\s+\$?$",
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
        tau: float = 0.85,
        window_size: int = 5,
        check_numbers: bool = True,
        max_new_tokens: int = 160,
        breach_run: int = 2,
        grounded_prompt: bool = True,
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

            Default of 2 comes from calibrate_tau.py, re-measured 2026-08-09
            against the granite3.3:8b actually being served. The numbers this
            docstring used to quote (answerable p90 0.387 / max 1.150) came from
            the transformers granite-3.3-2b runner and were never re-taken after
            the move to Ollama, because the script had a plain `for` over an
            async generator and could not run.

            Live figures, 6 prompts per class: answerable median 0.054, p90
            0.637, max 1.723; unanswerable median 0.042, p90 1.210, max 3.681.
            The two class MEDIANS are indistinguishable - the separation lives
            entirely in the tail, which is the honest reason a single breach is
            not evidence and a run is required.

            No setting in the sweep separates the classes cleanly. Every setting
            that detects 6/6 costs 2-4 false positives; the best zero-false-
            positive setting is tau=0.85 run=2 at 3/6. A false positive halts a
            correct answer in front of the judges, a miss is picked up by the
            numeric layer, so the trade is taken toward silence.
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
            f"Declared average compliance cycle time: {p['declared_avg_compliance_cycle_time_days']} days",
            f"Declared average order-to-cash: {p['declared_avg_order_to_cash_days']} days",
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
            cleaned = m.replace("$", "").replace(",", "").replace("%", "").rstrip(".")
            if cleaned and cleaned not in {"-", "."}:
                try:
                    out.append(float(cleaned))
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
        nxt = _NUM_RE.search(text[end:])
        right = end + (nxt.start() if nxt else len(text) - end)

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
        rb = _CLAUSE_BREAK.search(text[end:right])
        if rb:
            right = end + rb.start()
        return text[left:right]

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
        parts = _SENTENCE_BREAK.split(prefix)
        span = parts[-1] if parts else prefix
        if tail:
            # Forward reach stops at the CLAUSE, not the sentence. The scopes are
            # deliberately asymmetric: a metric named earlier governs what
            # follows it, but a metric named later belongs to its own claim.
            # Reaching forward by sentence let "Mean cycle time 8.5 days, median
            # 9.0, declared compliance cycle time 10.4" bind 8.5 to the
            # compliance metric two clauses away and reject the log's own
            # overall mean.
            span = f"{span} {_CLAUSE_BREAK.split(tail)[0]}"
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
        statistic: Optional[str] = None,
    ) -> Optional[float]:
        """First number in `span` that the event log cannot produce."""
        # A percentage that names no metric we know cannot be grounded at all.
        # The event log has no percentage field, so every legitimate percentage
        # is derived from a nameable metric (the bottleneck rate). Without this,
        # "the off-contract discount was 12%" was accepted because 12.0 is the
        # declared order-to-cash figure - a discount validated against a count
        # of days. There is no discount data in the log; the honest answer is
        # that the claim is ungrounded whatever its value.
        if "%" in span and metric is None:
            vals = self._numbers_in(span)
            if vals:
                return vals[0]
        for val in self._numbers_in(span):
            if not cm.is_grounded_number(val, scope=scope, metric=metric, statistic=statistic):
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
            val = self._ungrounded_number(
                m.group(),
                scope=self._claim_scope(prefix, figure=m.group()),
                # If the sentence names a metric, the figure must be one of THAT
                # metric's values, not merely some value in the log - unless the
                # figure is a threshold or a denominator, which qualify the
                # metric rather than state it.
                metric=None if _QUALIFIER_BEFORE.search(text[:m.start()])
                else cm.bound_metric(self._metric_span(prefix, tail)),
                # And if it names the statistic too, it must be that statistic's
                # value - "16 days on average" quotes Compliance Review's max.
                statistic=cm.bound_statistic(self._statistic_span(text, m.start(), m.end())),
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
            metric = (
                None if _QUALIFIER_BEFORE.search(text[:m.start()])
                else cm.bound_metric(self._metric_span(prefix, text[m.end():]))
            )
            statistic = cm.bound_statistic(self._statistic_span(text, m.start(), m.end()))
            scope = self._claim_scope(prefix, figure=figure)
            contradicted = self._ungrounded_number(
                figure, scope=scope, metric=metric, statistic=statistic
            ) is not None
            if contradicted:
                state = "contradicted"
            elif metric is not None:
                state = "verified"
            else:
                state = "unverifiable"
            counts[state] += 1
            detail.append({"figure": figure, "state": state, "metric": metric,
                           "statistic": statistic, "scope": scope})
        return {**counts, "figures": detail}

    # ---------- interception ----------

    async def _trip(
        self, query, rag, reason, tok, step, uncertainty, variance, shannon,
        ungrounded_value, emitted, history, t0, overhead_s, decision=None,
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
                "tokens_before_halt": emitted,
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
        decision = None         # (step, divergence) where a figure was last at stake

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
            # Remember where a figure was last on the table. The trip almost
            # never lands on that token - see _trip's `decision` parameter - so
            # the evidence has to be captured when it passes by.
            if divergence.numeric_options:
                decision = (step, divergence)

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

            # Entropy: noisy per-token signal, so require a sustained run.
            # Numbers: deterministic check, so one is enough.
            consecutive = consecutive + 1 if is_hallucinating else 0
            trip_reason = None
            if ungrounded_value is not None:
                trip_reason = "ungrounded_number"
            elif consecutive >= self.breach_run and not self._mid_figure(text):
                # A figure is being emitted right now, so the deterministic check
                # is about to have an opinion on it. Digits are individually
                # high-entropy, so letting the statistical layer fire first halts
                # on "1" of "12.5" and reports semantic_entropy with no value,
                # when one more token yields the far stronger "12.5 is not a
                # compliance cycle time". Defer to the layer that can name it.
                trip_reason = "semantic_entropy"

            if trip_reason:
                async for ev in self._trip(
                    query, rag, trip_reason, tok, step, uncertainty, variance,
                    shannon, ungrounded_value, emitted, history, t0, overhead_s,
                    decision,
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
                    emitted, history, t0, overhead_s, decision,
                ):
                    yield ev
                return

        total_ms = (time.perf_counter() - t0) * 1000
        yield InterceptionEvent(
            kind="done",
            text="".join(history),
            payload={
                "tokens": emitted,
                "elapsed_ms": round(total_ms, 2),
                "entropy_overhead_ms": round(overhead_s * 1000, 3),
                "overhead_pct": round(overhead_s * 1000 / total_ms * 100, 4) if total_ms else 0.0,
                "retrieval_reason": rag.get("reason"),
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
    loose = s.audit_figures("There were 393 orders above $100,000.")
    assert loose["verified"] == 0 and loose["contradicted"] == 0, loose
    assert loose["unverifiable"] == 2, loose
    print("sentinel_stream self-check OK")


if __name__ == "__main__":
    demo()
