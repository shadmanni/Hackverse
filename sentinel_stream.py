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
from typing import Any, Dict, Iterator, List, Optional

import celonis_metrics as cm
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
    r"-?\$?\d[\d,]*(?:\.\d+)?%?"
    r"(?![A-Za-z0-9_\-])"        # not followed by identifier characters
)


@dataclass
class InterceptionEvent:
    kind: str                      # token | intercept | recovery | done
    text: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


class SentinelStream:
    """Wraps a GraniteRunner with per-token interception."""

    def __init__(
        self,
        runner,
        retriever=None,
        tau: float = 0.65,
        window_size: int = 5,
        check_numbers: bool = True,
        max_new_tokens: int = 160,
        breach_run: int = 3,
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
        """
        self.runner = runner
        self.retriever = retriever
        self.tau = tau
        self.window_size = window_size
        self.check_numbers = check_numbers
        self.max_new_tokens = max_new_tokens
        self.breach_run = breach_run

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
            f"Orders above $100,000: {p['high_value_orders']}, "
            f"mean cycle {p['high_value_mean_cycle_days']} days",
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

    def _ungrounded_number(self, span: str, scope: str = "all") -> Optional[float]:
        """First number in `span` that the event log cannot produce."""
        for val in self._numbers_in(span):
            if not cm.is_grounded_number(val, scope=scope):
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
            val = self._ungrounded_number(
                m.group(), scope=self._claim_scope(text[:m.end()], figure=m.group())
            )
            if val is not None:
                return val, scanned_to
        return None, scanned_to

    # ---------- interception ----------

    def _trip(
        self, query, rag, reason, tok, step, uncertainty, variance, shannon,
        ungrounded_value, emitted, history, t0, overhead_s,
    ) -> Iterator[InterceptionEvent]:
        """Emit the interception + autonomous recovery pair for a tripped breaker."""
        yield InterceptionEvent(
            kind="intercept",
            text=tok,
            payload={
                "reason": reason,
                "token": tok,
                "token_index": getattr(step, "index", -1),
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
        yield InterceptionEvent(
            kind="recovery",
            text=cm.ground_truth_answer(query),
            payload={
                "strategy": "deterministic_event_log_lookup",
                "query": query,
                "retrieval_reason": rag.get("reason"),
            },
        )

    # ---------- the loop ----------

    def run(self, query: str) -> Iterator[InterceptionEvent]:
        rag = self._context_for(query)
        context = rag.get("context")

        engine = EntropyEngine(threshold_tau=self.tau, window_size=self.window_size)
        prompt = self.runner.build_prompt(query, context)

        history: List[str] = []
        text = ""               # assembled output, scanned for completed figures
        scanned_to = 0          # end offset of the last figure already checked
        t0 = time.perf_counter()
        overhead_s = 0.0
        emitted = 0
        consecutive = 0         # length of the current run of threshold breaches

        for step in self.runner.stream(prompt, max_new_tokens=self.max_new_tokens):
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
            elif consecutive >= self.breach_run:
                trip_reason = "semantic_entropy"

            if trip_reason:
                yield from self._trip(
                    query, rag, trip_reason, tok, step, uncertainty, variance,
                    shannon, ungrounded_value, emitted, history, t0, overhead_s,
                )
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
                yield from self._trip(
                    query, rag, "ungrounded_number", str(trailing), None, 0.0,
                    last.get("current_variance", 0.0), 0.0, trailing,
                    emitted, history, t0, overhead_s,
                )
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
