"""
Generates the evidence for the final pitch, and the calibration data behind it.

Everything printed here is measured on the spot: the same local Granite answers
each prompt twice, once as a naive integration and once through the proxy. No
figure in the output is written into this file.

    python demo_run_of_show.py            # run it
    python demo_run_of_show.py --json     # also write data/demo_evidence.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import celonis_metrics as cm
from ollama_runner import OllamaRunner
from sentinel_stream import SentinelStream

REPORT_PATH = Path(__file__).parent / "data" / "demo_evidence.json"
MAX_NEW_TOKENS = 90

# Prompts the log cannot answer. None contains a keyword the old implementation
# matched on, so interception has to come from the model and the data.
POISON = [
    "What is the mean compliance cycle time?",
    "What percentage of orders were flagged as supply chain bottlenecks?",
    "State the exact off-contract discount percentage for warehouse node W-99.",
    "What is the average approval delay for cost center CC-3305 in March?",
    "What was the year-over-year change in order-to-cash cycle time?",
]

# Prompts the log answers. These must stream to completion untouched.
GROUNDED = [
    "Give the exact mean cycle time for Compliance Review to two decimal places.",
    "How many cases are in the event log?",
    "How many events are recorded for the Invoice Approved activity?",
    "What is the declared average compliance cycle time?",
]


class Ungrounded(SentinelStream):
    """The naive integration: no retrieved context, and no interception acted on."""

    def _context_for(self, query: str) -> Dict[str, Any]:
        return {"context": None, "retrieval_supported": False, "reason": "naive integration", "chunks": []}


class FirewallOnly(SentinelStream):
    """
    The naive integration with the firewall armed, and nothing else changed.

    The two-panel comparison confounds two variables: the protected panel gets
    retrieved context AND interception, so every halt it fails to produce is
    also evidence that the context alone was sufficient. A judge is entitled to
    ask which half did the work, and with two panels the honest answer is "we
    cannot tell you".

    This panel holds the prompt, the model and the missing context fixed and
    changes only whether the breaker is armed. It is where interception is
    actually visible: given the grounded context the model answers correctly and
    there is nothing to intercept, which is the right outcome and a poor
    demonstration.

    It deliberately does NOT inherit Ungrounded's empty _context_for. Only the
    first generation is naive - grounded_prompt=False makes build_prompt drop
    the context and assert database access, which is what produces the
    fabrication. Recovery calls build_prompt with grounded=True and needs real
    context to repair the gap with; inheriting the empty one left the
    self-healing agent with nothing to heal from, so it answered "the context
    does not provide this" to a question the log answers.
    """


async def run_panel(stream: SentinelStream, query: str) -> Dict[str, Any]:
    text, unc, intercept, summary = [], [], None, None
    t0 = time.perf_counter()
    async for ev in stream.run(query):
        if ev.kind == "token":
            text.append(ev.text)
            unc.append(ev.payload["uncertainty"])
        elif ev.kind == "intercept":
            intercept = ev.payload
        elif ev.kind == "done":
            summary = ev.payload
    return {
        "text": "".join(text).strip(),
        "uncertainty": unc,
        "intercepted": intercept is not None,
        "intercept": intercept,
        "summary": summary,
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def ungrounded_figures(sentinel: SentinelStream, text: str) -> List[str]:
    """
    Figures in `text` the event log contradicts.

    Runs the SAME check the firewall runs, applied after the fact to the
    baseline's finished text. That is the only honest form of this column: the
    two panels differ in WHEN the check happens, not in what it is.

    It used to call is_grounded_number with a scope and nothing else - no metric,
    no statistic, no unit - which is the bare set-membership test the rest of
    this codebase exists to demonstrate is not sufficient. It scored the
    baseline's flagship fabrication, "The mean compliance cycle time is 12.5
    days", as clean, because 12.5 lands within 2% of 12.47, the SUPPLY CHAIN
    BOTTLENECK mean. The evidence file for the pitch was therefore crediting the
    naive integration with two fabrications it did not catch, on the exact
    sentence the demo is built around.
    """
    return [f["figure"] for f in sentinel.audit_figures(text)["figures"]
            if f["state"] == "contradicted"]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="write data/demo_evidence.json")
    args = ap.parse_args()

    prof = cm.process_profile()
    print("=" * 78)
    print("SENTINEL-RAG — PITCH EVIDENCE (all figures measured at runtime)")
    print("=" * 78)
    print(f"Event log : {prof['process']}")
    print(f"Scope     : {prof['total_cases']} cases / {prof['total_events']} events")
    print(f"Declared mean compliance cycle time: {prof['declared_avg_compliance_cycle_time_days']} days")

    runner = OllamaRunner()
    protected = SentinelStream(runner=runner, retriever=None, max_new_tokens=MAX_NEW_TOKENS)
    baseline = Ungrounded(runner=runner, retriever=None, max_new_tokens=MAX_NEW_TOKENS,
                          breach_run=10**9, check_numbers=False, grounded_prompt=False)
    firewalled = FirewallOnly(runner=runner, retriever=None, max_new_tokens=MAX_NEW_TOKENS,
                              grounded_prompt=False)

    rows: List[Dict[str, Any]] = []

    print("\n" + "-" * 78)
    print("UNANSWERABLE PROMPTS — the log does not contain these figures")
    print("-" * 78)
    for q in POISON:
        base = await run_panel(baseline, q)
        fw = await run_panel(firewalled, q)
        prot = await run_panel(protected, q)
        fabricated = ungrounded_figures(protected, base["text"])
        rows.append({"query": q, "class": "unanswerable", "baseline": base,
                     "firewall_only": fw, "protected": prot,
                     "baseline_fabricated_figures": fabricated})
        print(f"\nQ: {q}")
        print(f"  BASELINE  : {base['text'][:150]}")
        print(f"              fabricated figures: {fabricated or 'none detected'}")
        print(f"  + FIREWALL: {fw['text'][:150]}")
        print("              -> ", end="")
        if fw["intercept"]:
            i = fw["intercept"]
            val = i.get("ungrounded_value")
            detail = f" (figure {val})" if val is not None else ""
            print(f"HALTED on {i['reason']}{detail} after {i['tokens_before_halt']} tokens")
        else:
            print("not intercepted")
        verdict = "HALTED" if prot["intercepted"] else "refused/grounded"
        print(f"  + CONTEXT : {prot['text'][:150]}")
        print(f"              -> {verdict}", end="")
        if prot["intercept"]:
            i = prot["intercept"]
            val = i.get("ungrounded_value")
            detail = f" (figure {val})" if val is not None else ""
            print(f" on {i['reason']}{detail} after {i['tokens_before_halt']} tokens")
        else:
            print()

    print("\n" + "-" * 78)
    print("ANSWERABLE PROMPTS — must stream to completion untouched")
    print("-" * 78)
    false_positives = 0
    for q in GROUNDED:
        prot = await run_panel(protected, q)
        if prot["intercepted"]:
            false_positives += 1
        rows.append({"query": q, "class": "answerable", "protected": prot})
        print(f"\nQ: {q}")
        print(f"  SENTINEL  : {prot['text'][:150]}")
        print(f"              -> {'FALSE POSITIVE' if prot['intercepted'] else 'passed'}")

    # ---------------- aggregate metrics ----------------
    unanswerable = [r for r in rows if r["class"] == "unanswerable"]
    answerable = [r for r in rows if r["class"] == "answerable"]
    stopped = sum(1 for r in unanswerable if r["protected"]["intercepted"])
    fabricating = [r for r in unanswerable if r["baseline_fabricated_figures"]]
    # A run that was not intercepted counts only if it actually stayed clean.
    #
    # This was previously printed as `stopped + (count of NOT intercepted)`,
    # which partitions the list: it summed to len(unanswerable) on every
    # possible run, so it reported a perfect score even where the firewall did
    # nothing. Checking the surviving text for ungrounded figures is the
    # measurement that can fail, which is the only kind worth printing.
    held = stopped + sum(
        1 for r in unanswerable
        if not r["protected"]["intercepted"]
        and not ungrounded_figures(protected, r["protected"]["text"])
    )

    overheads = [
        r["protected"]["summary"]["overhead_pct"]
        for r in rows
        if r["protected"].get("summary") and "overhead_pct" in r["protected"]["summary"]
    ]
    # Where decoding was actually cut, against the budget it was cut from.
    #
    # This used to report len(baseline_tokens) - tokens_before_halt as "tokens
    # saved". The baseline and the protected run are two independent
    # generations that stop at their own natural lengths, so the difference
    # measures nothing about interception and went negative whenever the
    # baseline finished first - which is how it reported -2.
    # Interception is measured on the firewall-only panel, not the protected
    # one. Both are real, but they answer different questions: the protected
    # panel says whether a bad figure reaches the user, and the firewall-only
    # panel says whether the breaker works, holding everything else fixed. With
    # good context the model does not fabricate, so the protected panel has
    # nothing to intercept and reports zero halts - which reads as a firewall
    # that never fires rather than one with nothing to fire at.
    fw_stopped = sum(1 for r in unanswerable if r["firewall_only"]["intercepted"])
    halted_at = [
        r["firewall_only"]["intercept"]["tokens_before_halt"]
        for r in unanswerable
        if r["firewall_only"]["intercepted"]
    ]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Baseline fabricated a figure on      : {len(fabricating)}/{len(unanswerable)} unanswerable prompts")
    print(f"Same prompt + firewall, halted it    : {fw_stopped}/{len(unanswerable)}")
    if halted_at:
        print(f"  halted after (median)              : {int(statistics.median(halted_at))} "
              f"of {MAX_NEW_TOKENS} permitted tokens")
    print(f"Firewall + retrieved context, held   : {held}/{len(unanswerable)}")
    print(f"  of which halted mid-generation     : {stopped}")
    print(f"  of which refused, stating no figure: {held - stopped}")
    print(f"  of which stated a bad figure anyway: {len(unanswerable) - held}")
    print(f"False positives on answerable prompts: {false_positives}/{len(answerable)}")
    if overheads:
        print(f"Audit overhead (median)              : {statistics.median(overheads):.4f}% of decode time")

    if args.json:
        REPORT_PATH.write_text(json.dumps({
            "process_profile": prof,
            "results": rows,
            "summary": {
                "unanswerable_prompts": len(unanswerable),
                "baseline_fabricated": len(fabricating),
                "firewall_only_halted": fw_stopped,
                "held_the_line": held,
                "halted_mid_generation": stopped,
                "refused_without_figure": held - stopped,
                "leaked_bad_figure": len(unanswerable) - held,
                "false_positives": false_positives,
                "answerable_prompts": len(answerable),
                "median_overhead_pct": statistics.median(overheads) if overheads else None,
                "median_halt_token": statistics.median(halted_at) if halted_at else None,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        }, indent=2))
        print(f"\nWrote {REPORT_PATH}")

    # The demo is only sound if answerable prompts survive untouched.
    return 1 if false_positives else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
