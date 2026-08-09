"""
Threshold calibration for the entropy layer.

tau=0.65 and breach_run=3 are working defaults picked from a handful of
observations, not calibrated values. This sweeps both against real Granite
log-probabilities and reports what each setting would actually have done, so the
number in the pitch is measured rather than asserted.

The two error types are not symmetric:
  * a FALSE POSITIVE halts a correct answer - the demo visibly breaks
  * a MISS lets a fabricated figure through - the product silently fails

Both are reported per setting; which to favour is a judgement call, so this
prints the frontier instead of collapsing it to one score.

    python calibrate_tau.py                 # sweep and report
    python calibrate_tau.py --json          # also write data/tau_calibration.json

Stop the backend first: two Granite instances on one MPS device segfault.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from entropy_engine import EntropyEngine
from granite_runner import GraniteRunner

REPORT = Path(__file__).parent / "data" / "tau_calibration.json"

# Answerable from the event log. Any halt here is a false positive.
ANSWERABLE = [
    "Give the exact mean cycle time for Compliance Review to two decimal places.",
    "How many cases are in the event log?",
    "How many events are recorded for the Invoice Approved activity?",
    "What is the declared average compliance cycle time?",
    "How many orders exceed $100,000?",
    "What is the maximum cycle time recorded?",
]

# Not answerable from the log. A halt here is a correct detection.
UNANSWERABLE = [
    "What is the mean compliance cycle time?",
    "What percentage of orders were flagged as supply chain bottlenecks?",
    "State the exact off-contract discount percentage for warehouse node W-99.",
    "What is the average approval delay for cost center CC-3305 in March?",
    "What was the year-over-year change in order-to-cash cycle time?",
    "What is the total dollar value of invoices processed last quarter?",
]

TAUS = [0.45, 0.55, 0.65, 0.75, 0.85, 1.00, 1.25]
RUNS = [1, 2, 3, 4, 5]


def collect(runner: GraniteRunner, query: str, grounded: bool, max_new_tokens: int = 90) -> List[float]:
    """Per-token weighted uncertainty for one generation, at a fixed tau.

    The uncertainty score does not depend on tau - tau is only the comparison -
    so one pass per prompt is enough to evaluate every threshold afterwards.
    """
    engine = EntropyEngine(threshold_tau=0.65)
    context = None
    if grounded:
        from sentinel_stream import SentinelStream
        context = SentinelStream._aggregate_block()
    prompt = runner.build_prompt(query, context, grounded=grounded)
    scores = []
    for step in runner.stream(prompt, max_new_tokens=max_new_tokens):
        _, uncertainty, _ = engine.evaluate_token(
            step.text, logprob=step.logprob, top_probs=step.top_probs
        )
        scores.append(uncertainty)
    return scores


def would_trip(scores: List[float], tau: float, run: int) -> bool:
    """True if `run` consecutive tokens exceed tau."""
    streak = 0
    for s in scores:
        streak = streak + 1 if s > tau else 0
        if streak >= run:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=90)
    args = ap.parse_args()

    runner = GraniteRunner.get()

    print("Collecting real log-probabilities…")
    # Answerable prompts run grounded, the way the product answers them.
    # Unanswerable prompts run ungrounded, which is where fabrication happens.
    pos, neg = [], []
    for q in ANSWERABLE:
        pos.append(collect(runner, q, grounded=True, max_new_tokens=args.max_tokens))
        print(f"  answerable   {len(pos[-1]):>3} tokens  {q[:52]}")
    for q in UNANSWERABLE:
        neg.append(collect(runner, q, grounded=False, max_new_tokens=args.max_tokens))
        print(f"  unanswerable {len(neg[-1]):>3} tokens  {q[:52]}")

    allpos = [s for run in pos for s in run]
    allneg = [s for run in neg for s in run]
    print(f"\nPer-token uncertainty — answerable:   median {statistics.median(allpos):.3f}  "
          f"p90 {sorted(allpos)[int(len(allpos) * 0.9)]:.3f}  max {max(allpos):.3f}")
    print(f"Per-token uncertainty — unanswerable: median {statistics.median(allneg):.3f}  "
          f"p90 {sorted(allneg)[int(len(allneg) * 0.9)]:.3f}  max {max(allneg):.3f}")

    rows: List[Dict[str, Any]] = []
    print(f"\n{'tau':>6} {'run':>4} {'false pos':>12} {'detected':>12}   verdict")
    print("-" * 58)
    for tau in TAUS:
        for run in RUNS:
            fp = sum(would_trip(s, tau, run) for s in pos)
            tp = sum(would_trip(s, tau, run) for s in neg)
            rows.append({"tau": tau, "breach_run": run,
                         "false_positives": fp, "answerable": len(pos),
                         "detected": tp, "unanswerable": len(neg)})
            note = ""
            if fp == 0 and tp == len(neg):
                note = "  <- clean separation"
            elif fp == 0:
                note = "  no false positives"
            print(f"{tau:>6.2f} {run:>4} {fp:>7}/{len(pos):<4} {tp:>7}/{len(neg):<4}{note}")

    clean = [r for r in rows if r["false_positives"] == 0 and r["detected"] == r["unanswerable"]]
    safest = [r for r in rows if r["false_positives"] == 0]
    print()
    if clean:
        best = max(clean, key=lambda r: (r["tau"], -r["breach_run"]))
        print(f"RECOMMENDED  tau={best['tau']} breach_run={best['breach_run']} "
              f"— separates both classes with zero false positives")
    elif safest:
        best = max(safest, key=lambda r: r["detected"])
        print(f"RECOMMENDED  tau={best['tau']} breach_run={best['breach_run']} "
              f"— zero false positives, detects {best['detected']}/{best['unanswerable']} "
              f"(the rest are caught by the numeric-grounding layer)")
    else:
        print("No setting avoids false positives on this set — the entropy layer alone "
              "cannot separate these classes; the numeric layer is doing the real work.")
    print("\nNote: this calibrates the ENTROPY layer only. The numeric-grounding layer is "
          "deterministic and independent of tau.")

    if args.json:
        REPORT.write_text(json.dumps({
            "sweep": rows,
            "answerable_prompts": ANSWERABLE,
            "unanswerable_prompts": UNANSWERABLE,
            "per_token_uncertainty": {
                "answerable_median": statistics.median(allpos),
                "unanswerable_median": statistics.median(allneg),
            },
        }, indent=2))
        print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
