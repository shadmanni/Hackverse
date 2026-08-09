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
    r"(?<![A-Za-z0-9_.\-])"      # not preceded by identifier characters
    r"-?\$?\d[\d,]*(?:\.\d+)?%?"
    r"(?![A-Za-z0-9_\-])"        # not followed by identifier characters
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
            return {"context": agg, "is_poison": False, "reason": "no retriever", "chunks": []}
        try:
            res = self.retriever.format_granite_context(query)
            chunks = res.get("context")
            res["context"] = f"{agg}\n\nRETRIEVED EVENT CHUNKS:\n{chunks}" if chunks else agg
            return res
        except Exception as err:  # retrieval must never take the stream down
            return {"context": agg, "is_poison": False, "reason": f"retrieval error: {err}", "chunks": []}

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
        self, span: str, scope: str = "all", metric: Optional[str] = None
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
            if not cm.is_grounded_number(val, scope=scope, metric=metric):
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
                # metric's values, not merely some value in the log.
                metric=cm.bound_metric(prefix),
            )
            if val is not None:
                return val, scanned_to
        return None, scanned_to

    # ---------- interception ----------

    async def _trip(
        self, query, rag, reason, tok, step, uncertainty, variance, shannon,
        ungrounded_value, emitted, history, t0, overhead_s,
    ) -> AsyncIterator[InterceptionEvent]:
        """Emit the interception + autonomous recovery pair for a tripped breaker."""
        log_compliance_breach(query, reason, uncertainty, ungrounded_value)
        yield InterceptionEvent(
            kind="intercept",
            text=tok,
            payload={
                "reason": reason,
                "token": tok,
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

        yield InterceptionEvent(
            kind="recovery",
            text=answer,
            payload={
                "strategy": "milvus_context_repair_and_regenerate",
                "query": query,
                "retrieval_reason": repaired.get("reason"),
                "regenerated": bool(recovered),
                "all_figures_grounded": verified,
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
                    "index": step.index,
                },
            )

        # A response ending on a figure leaves it unterminated and therefore
        # unchecked: "...mean cycle time is 87.6" would pass straight through.
        # Re-scan without complete_only now that no more tokens are coming.
        if self.check_numbers:
            trailing, _ = self._scan_new_figures(text, scanned_to, complete_only=False)
            if trailing is not None:
                last = engine.get_metrics_snapshot()
                async for ev in self._trip(
                    query, rag, "ungrounded_number", str(trailing), None, 0.0,
                    last.get("current_variance", 0.0), 0.0, trailing,
                    emitted, history, t0, overhead_s,
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
    print("sentinel_stream self-check OK")


if __name__ == "__main__":
    demo()
