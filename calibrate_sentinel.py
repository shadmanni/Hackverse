"""
The calibration harness: 101 labelled queries, every threshold in the system
swept against them, three separate reports out.

    python calibrate_sentinel.py                  # collect (slow, cached) then report
    python calibrate_sentinel.py --limit 20       # smoke run
    python calibrate_sentinel.py --report-only    # re-sweep the cached collection
    python calibrate_sentinel.py --no-paths       # skip K-path sampling (tau sweep only)

Stop the backend first. Ollama serves one slot per model, so a live backend and
this script take turns and every latency number comes out wrong.

What it answers
---------------
1. TAU / BREACH_RUN. False-positive and false-negative rates, reported
   SEPARATELY and never averaged. Averaging them into one score hides the only
   thing that matters about the trade-off: they cost different amounts.
2. SIM_TAU, the clustering threshold. Swept, with the numeric override toggled
   on and off, so the choice is defended by a table rather than asserted.
3. LATENCY. Per-token overhead of each layer, the cost of one K-path checkpoint,
   the rate at which checkpoints actually fire, and the amortised per-token
   figure that follows from those two.

How a query is scored
---------------------
ANSWERABLE  -> blocked = FALSE POSITIVE. Unambiguous: the log holds the value.
UNANSWERABLE -> the failure is a specific figure reaching the user un-halted.
   So: blocked = correct; not blocked and the answer states NO figure = also
   correct, because refusing is not hallucinating; not blocked and a figure was
   stated = FALSE NEGATIVE. Scoring a refusal as a miss would drive tau down
   until the system halted on honest answers, which is the opposite of the goal.
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import calibration_set
import semantic_divergence as sd
import semantic_entropy as se
from entropy_engine import EntropyEngine
from sentinel_stream import SentinelStream

RAW_PATH = Path(__file__).parent / "data" / "calibration_raw.json"
REPORT_PATH = Path(__file__).parent / "data" / "calibration_report.json"

TAUS = [0.45, 0.55, 0.65, 0.75, 0.85, 1.00, 1.25, 1.50]
RUNS = [1, 2, 3, 4]
SIM_TAUS = [0.50, 0.60, 0.65, 0.72, 0.80, 0.90]
MAX_TOKENS = 90


# --------------------------------------------------------------------------- #
# Collection - the slow part, cached
# --------------------------------------------------------------------------- #

async def collect_one(runner, sem, entry: Dict[str, str], want_paths: bool) -> Dict[str, Any]:
    """
    One generation, fully instrumented.

    The per-token uncertainty score does NOT depend on tau - tau is only the
    comparison - so one pass per prompt is enough to evaluate every threshold
    afterwards. That is what makes a 101-query sweep affordable at all.
    """
    query, label = entry["prompt"], entry["label"]
    grounded = label == "answerable"
    context = SentinelStream._aggregate_block() if grounded else None
    prompt = runner.build_prompt(query, context, grounded=grounded)

    stream = SentinelStream(runner=None)          # for its numeric-grounding methods
    engine = EntropyEngine(threshold_tau=0.65)

    scores: List[float] = []
    eff_taus: List[float] = []
    weights: List[float] = []
    numeric_mass: List[float] = []
    text = ""
    scanned = 0
    numeric_trip_at: Optional[int] = None
    checkpoint_at: List[int] = []
    entropy_us: List[float] = []       # per-token cost of the always-on layers
    numeric_us: List[float] = []
    since_ckpt = 999
    t0 = time.perf_counter()

    async for step in runner.stream(prompt, max_new_tokens=MAX_TOKENS):
        c0 = time.perf_counter()
        _, unc, _ = engine.evaluate_token(step.text, logprob=step.logprob,
                                          top_probs=step.top_probs)
        div = sd.analyse(step.top_tokens, step.top_probs)
        entropy_us.append((time.perf_counter() - c0) * 1e6)

        scores.append(unc)
        eff_taus.append(engine.last_effective_tau)
        w = engine.get_token_type_weight(step.text)
        weights.append(w)
        numeric_mass.append(div.numeric_mass)

        n0 = time.perf_counter()
        text += step.text
        val, scanned = stream._scan_new_figures(text, scanned, complete_only=True)
        numeric_us.append((time.perf_counter() - n0) * 1e6)
        if val is not None and numeric_trip_at is None:
            numeric_trip_at = len(scores) - 1

        # Record where the adaptive policy WOULD fire, without paying for it -
        # the checkpoint rate is a property of the policy and the token stream,
        # not of whether the draft model ran.
        if se.should_checkpoint(div, since_ckpt, w):
            checkpoint_at.append(len(scores) - 1)
            since_ckpt = 0
        else:
            since_ckpt += 1

    gen_ms = (time.perf_counter() - t0) * 1000

    # A figure stated at the very end is never terminated, so re-scan the whole
    # text now that no more tokens are coming - same reason SentinelStream does.
    trailing, _ = stream._scan_new_figures(text, 0, complete_only=False)
    audit = stream.audit_figures(text)

    row: Dict[str, Any] = {
        "prompt": query,
        "label": label,
        "why": entry["why"],
        "tokens": len(scores),
        "scores": [round(s, 5) for s in scores],
        "effective_taus": [round(t, 4) for t in eff_taus],
        "weights": [round(w, 2) for w in weights],
        "numeric_mass": [round(m, 4) for m in numeric_mass],
        "numeric_trip_at": numeric_trip_at,
        "trailing_ungrounded": trailing,
        "figures_stated": audit["verified"] + audit["unverifiable"] + audit["contradicted"],
        "figure_audit": {k: audit[k] for k in ("verified", "unverifiable", "contradicted")},
        "answer": text.strip(),
        "gen_ms": round(gen_ms, 1),
        "checkpoint_at": checkpoint_at,
        "entropy_us_per_token": round(statistics.mean(entropy_us), 2) if entropy_us else 0.0,
        "numeric_us_per_token": round(statistics.mean(numeric_us), 2) if numeric_us else 0.0,
    }

    if want_paths:
        # K speculative paths at the FIRST point the adaptive policy would have
        # fired, which is where a live checkpoint would actually have sampled.
        # Falling back to a short prefix keeps a row for queries that never
        # trigger one, so the ablation is not silently restricted to the
        # figure-heavy half of the set.
        cut = checkpoint_at[0] + 1 if checkpoint_at else min(8, len(scores))
        prefix = text_prefix(text, cut)
        p0 = time.perf_counter()
        result = await sem.measure(prompt, prefix)
        row["path_ms"] = round((time.perf_counter() - p0) * 1000, 1)
        row["paths"] = result.paths if result else []
        row["path_prefix"] = prefix
    return row


def text_prefix(text: str, n_tokens: int) -> str:
    """
    Approximate the first n decoded tokens by character budget.

    The exact token boundaries are not needed: the prefix only has to land the
    draft model in the same sentence the checkpoint fired in. ~4 characters per
    token is close enough for that and avoids carrying the token list around.
    """
    return text[: max(12, n_tokens * 4)]


async def collect(limit: Optional[int], want_paths: bool) -> List[Dict[str, Any]]:
    from ollama_runner import OllamaRunner

    runner = OllamaRunner()
    sem = se.SemanticEntropy(k=5, lookahead=10)

    # Warm the semantic-binding index BEFORE anything is timed. It builds a
    # MiniLM on first use - 12.5 s measured cold - and that landed inside the
    # first query's per-token numeric timing, reporting the numeric layer at
    # 243 ms/token when its steady-state cost is 13 ms per unbound figure and
    # 0.1 ms per token. main._load_models pays the same cost at startup for the
    # same reason; a benchmark that does not is measuring a model load.
    #
    # It has to ENCODE a span, not merely build the index - the same finding
    # main._load_models records. Building the alias index leaves a further ~6 s
    # of torch warm-up owed on the first real encode, and calling _alias_index()
    # alone left a 689 ms/token outlier on query 1 of this very sweep.
    try:
        import celonis_metrics as cm
        cm._semantic_metrics("warm the encoder with one span")
    except Exception as err:
        print(f"  (semantic binding warm-up skipped: {err})")
    sem._embed(["warm"])          # and the clustering encoder

    entries = calibration_set.full_set()
    if limit:
        # Interleave the classes so a truncated run still has both.
        ans = [e for e in entries if e["label"] == "answerable"]
        una = [e for e in entries if e["label"] == "unanswerable"]
        entries = [x for pair in zip(ans, una) for x in pair][:limit]

    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries, 1):
        try:
            row = await collect_one(runner, sem, entry, want_paths)
        except Exception as err:
            print(f"  [{i:>3}/{len(entries)}] FAILED {entry['prompt'][:50]}: {err}")
            continue
        rows.append(row)
        flag = "A" if row["label"] == "answerable" else "U"
        print(f"  [{i:>3}/{len(entries)}] {flag} {row['tokens']:>3}tok "
              f"{row['gen_ms']:>6.0f}ms ckpt={len(row['checkpoint_at'])} "
              f"{entry['prompt'][:46]}")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(rows, indent=1))     # resumable
    return rows


# --------------------------------------------------------------------------- #
# Sweep 1 - tau and breach_run
# --------------------------------------------------------------------------- #

def would_trip_entropy(row: Dict[str, Any], tau: float, run: int, dynamic: bool) -> bool:
    """
    `run` consecutive tokens above threshold.

    With dynamic=True the per-token effective threshold is used, scaled to the
    tau being swept - which is how the position-aware rule is evaluated against
    the flat one on identical data rather than on a separate run.
    """
    streak = 0
    base = 0.65      # the tau the effective values were recorded at
    for i, s in enumerate(row["scores"]):
        limit = tau * (row["effective_taus"][i] / base) if dynamic else tau
        streak = streak + 1 if s > limit else 0
        if streak >= run:
            return True
    return False


def sweep_tau(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ans = [r for r in rows if r["label"] == "answerable"]
    una = [r for r in rows if r["label"] == "unanswerable"]

    def numeric_trips(r) -> bool:
        return r["numeric_trip_at"] is not None or r["trailing_ungrounded"] is not None

    # An unanswerable query is only MISSED if a figure actually reached the user.
    # A refusal is correct behaviour and must not be scored as a miss.
    def missed(r, entropy_trip: bool) -> bool:
        return not entropy_trip and not numeric_trips(r) and r["figures_stated"] > 0

    out = []
    for dynamic in (False, True):
        for tau in TAUS:
            for run in RUNS:
                fp_e = sum(would_trip_entropy(r, tau, run, dynamic) for r in ans)
                fp_n = sum(numeric_trips(r) for r in ans)
                fp_any = sum(would_trip_entropy(r, tau, run, dynamic) or numeric_trips(r)
                             for r in ans)
                fn = sum(missed(r, would_trip_entropy(r, tau, run, dynamic)) for r in una)
                caught = len(una) - fn
                out.append({
                    "dynamic_tau": dynamic, "tau": tau, "breach_run": run,
                    "answerable": len(ans), "unanswerable": len(una),
                    "fp_entropy": fp_e, "fp_numeric": fp_n, "fp_combined": fp_any,
                    "fp_rate": round(fp_any / len(ans), 4) if ans else 0.0,
                    "fn": fn, "fn_rate": round(fn / len(una), 4) if una else 0.0,
                    "caught": caught,
                })
    return {"grid": out, "answerable_n": len(ans), "unanswerable_n": len(una)}


# --------------------------------------------------------------------------- #
# Sweep 2 - clustering threshold
# --------------------------------------------------------------------------- #

def sweep_clustering(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    How sim_tau and the numeric override change the entropy the clusterer reports.

    Two failure directions, and the table has to show both:
      * too LOOSE - everything lands in one cluster, entropy collapses toward 0
        on genuine fabrications, and the layer reports false confidence.
      * too STRICT - paraphrases of one claim split, entropy rises on correct
        answers, and the layer raises false alarms.

    The useful column is neither of those alone but the SEPARATION between the
    classes: mean normalised entropy on unanswerable minus answerable. A
    threshold that raises both equally has bought nothing.
    """
    with_paths = [r for r in rows if r.get("paths")]
    if not with_paths:
        return {"note": "no K-path samples collected; re-run without --no-paths"}

    engine = se.SemanticEntropy(k=5)
    grid = []
    for numeric_override in (True, False):
        for sim in SIM_TAUS:
            engine.sim_tau = sim
            per_class: Dict[str, List[float]] = {"answerable": [], "unanswerable": []}
            clusters: Dict[str, List[int]] = {"answerable": [], "unanswerable": []}
            for r in with_paths:
                paths = r["paths"]
                cl = _cluster_with(engine, paths, numeric_override)
                n = len(paths)
                h = abs(-sum((len(c) / n) * math.log2(len(c) / n) for c in cl)) if n else 0.0
                ceiling = math.log2(n) if n > 1 else 1.0
                per_class[r["label"]].append(h / ceiling if ceiling else 0.0)
                clusters[r["label"]].append(len(cl))
            a = per_class["answerable"] or [0.0]
            u = per_class["unanswerable"] or [0.0]
            grid.append({
                "numeric_override": numeric_override,
                "sim_tau": sim,
                "answerable_mean_norm_entropy": round(statistics.mean(a), 4),
                "unanswerable_mean_norm_entropy": round(statistics.mean(u), 4),
                "separation": round(statistics.mean(u) - statistics.mean(a), 4),
                "answerable_mean_clusters": round(statistics.mean(
                    clusters["answerable"] or [0]), 2),
                "unanswerable_mean_clusters": round(statistics.mean(
                    clusters["unanswerable"] or [0]), 2),
            })
    best = max(grid, key=lambda g: g["separation"])
    return {"grid": grid, "n_sampled_queries": len(with_paths), "best": best}


def _cluster_with(engine, paths, numeric_override: bool):
    """Cluster `paths`, optionally disabling the same-figure/different-figure rule."""
    if numeric_override:
        return engine._cluster(paths)[0]
    # Embedding-only, to show what the override is worth. Monkey-patching the
    # figure extractor is the smallest way to ablate one rule without a second
    # copy of the clustering code that could drift from the real one.
    original = se._figures
    se._figures = lambda _t: frozenset()
    try:
        return engine._cluster(paths)[0]
    finally:
        se._figures = original


# --------------------------------------------------------------------------- #
# Sweep 3 - latency
# --------------------------------------------------------------------------- #

def latency_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The per-token overhead budget, measured rather than asserted.

    The budget is 15 ms per token. Two of the three layers are far under it and
    one is far over it, so the honest number is the AMORTISED one: a checkpoint
    that costs C milliseconds and fires on a fraction f of tokens adds C x f per
    token. Reporting the checkpoint cost alone would overstate the overhead by
    the checkpoint rate; reporting only the always-on layers would understate it
    by everything the expensive layer costs.
    """
    ent = [r["entropy_us_per_token"] for r in rows]
    num = [r["numeric_us_per_token"] for r in rows]
    tokens = sum(r["tokens"] for r in rows)
    path_ms = [r["path_ms"] for r in rows if r.get("path_ms")]

    knobs = checkpoint_knobs(rows, statistics.median(path_ms) if path_ms else 0.0)
    # Recomputed from the policy against the cached per-token distributions, NOT
    # read off the cached checkpoint_at. Those were recorded before the token-0
    # checkpoint was removed, so they describe a policy the code no longer runs;
    # the numeric_mass and weights columns are exactly what makes re-deriving it
    # possible without re-generating 101 answers.
    rate = next(g["checkpoint_rate"] for g in knobs["grid"]
                if g["min_gap"] == 6 and g["numeric_mass_floor"] == 0.25)
    ckpts = round(rate * tokens)
    ckpt_median = statistics.median(path_ms) if path_ms else 0.0
    # MEDIAN, not mean. The mean over this set is 20,905 us/token against a
    # median of 83 - a 250x gap produced by ONE row, the first query, which paid
    # a 689 ms lazy torch warm-up that no later query pays and that the server
    # pays at startup instead (main._load_models). Reporting the mean would put
    # the numeric layer at 21 ms/token and declare a 15 ms budget blown by a cost
    # the product does not actually incur; reporting the median alone would hide
    # a genuine tail, so p95 is carried beside it.
    always_on_ms = (statistics.median(ent) + statistics.median(num)) / 1000 if ent else 0.0
    amortised = always_on_ms + ckpt_median * rate

    def p(xs, q):
        return round(sorted(xs)[min(int(len(xs) * q), len(xs) - 1)], 2) if xs else 0.0

    return {
        "budget_ms_per_token": 15.0,
        "always_on": {
            "entropy_us_per_token_median": round(statistics.median(ent), 2) if ent else 0.0,
            "entropy_us_per_token_p95": p(ent, 0.95),
            "numeric_grounding_us_per_token_median": round(statistics.median(num), 2) if num else 0.0,
            "numeric_grounding_us_per_token_p95": p(num, 0.95),
            "numeric_grounding_us_per_token_max": round(max(num), 2) if num else 0.0,
            # Kept so the gap between mean and median stays visible rather than
            # being quietly replaced by the flattering number.
            "numeric_grounding_us_per_token_mean": round(statistics.mean(num), 2) if num else 0.0,
            "combined_ms_per_token": round(always_on_ms, 4),
        },
        "k_path_checkpoint": {
            "median_ms": round(ckpt_median, 1),
            "p90_ms": round(sorted(path_ms)[int(len(path_ms) * .9)], 1) if path_ms else 0.0,
            "n_measured": len(path_ms),
        },
        "adaptive_checkpointing": {
            "total_tokens": tokens,
            "total_checkpoints": ckpts,
            "checkpoint_rate": round(rate, 4),
            "tokens_per_checkpoint": round(1 / rate, 1) if rate else None,
            # What it would have cost with no gate at all. The ratio between this
            # and the amortised figure IS the value of adaptive checkpointing.
            "naive_every_token_ms_per_token": round(ckpt_median, 1),
            "saving_factor": round(1 / rate, 1) if rate else None,
        },
        "amortised_ms_per_token": round(amortised, 3),
        "within_budget": amortised <= 15.0,
        "knobs": knobs,
    }


# Measured on an idle machine, qwen2.5:0.5b, median of 3 runs each. Ollama runs
# `-np 1`, so K concurrent requests only partly overlap; K=5 costs 3.1x K=1
# rather than 5x, and OLLAMA_NUM_PARALLEL is the daemon setting that would close
# the rest of that gap. These are the alternatives to the per-checkpoint cost
# measured in-sweep, and they are what the projection below prices.
CHECKPOINT_COST_MS = {
    "K=5 lookahead=10 (current)": 1010.0,
    "K=5 lookahead=6": 624.0,
    "K=3 lookahead=10": 640.0,
    "K=1 lookahead=10 (= perfect parallelism ceiling for K=5)": 323.0,
}


def checkpoint_knobs(rows: List[Dict[str, Any]], measured_cost_ms: float) -> Dict[str, Any]:
    """
    What it would take to get the K-path layer under 15 ms/token.

    The amortised cost is (checkpoint cost x checkpoint rate), so there are
    exactly two levers. The RATE is recomputed here from the cached per-token
    numeric_mass rather than re-run, which is the whole reason that column is
    stored: the adaptive policy is a pure function of the token stream, so every
    (min_gap, floor) pair can be priced offline against the same 101 generations.
    """
    def rate_for(min_gap: int, floor: float) -> float:
        ck = tok = 0
        for r in rows:
            # 0, matching SentinelStream.run - a checkpoint at token 0 has no
            # prefix to continue and was removed. Seeding this at 999 (as the
            # collection did) put one spurious checkpoint at the head of EVERY
            # response, which is what pinned the projected rate at a 4.5% floor
            # no min_gap could get under.
            since = 0
            for i, mass in enumerate(r["numeric_mass"]):
                w = r["weights"][i]
                tok += 1
                since += 1
                if since >= min_gap and (mass >= floor or w >= se.HIGH_STAKES_WEIGHT):
                    ck += 1
                    since = 0
        return ck / tok if tok else 0.0

    grid = []
    for min_gap in (6, 10, 15, 20, 30, 40, 60):
        for floor in (0.25, 0.50):
            r = rate_for(min_gap, floor)
            grid.append({
                "min_gap": min_gap, "numeric_mass_floor": floor,
                "checkpoint_rate": round(r, 4),
                "tokens_per_checkpoint": round(1 / r, 1) if r else None,
                "amortised_ms_per_token": {
                    k: round(c * r, 2) for k, c in CHECKPOINT_COST_MS.items()
                },
                "within_budget_at_measured_cost": measured_cost_ms * r <= 15.0,
            })
    # Clearing the budget by raising min_gap until the layer never runs is not
    # engineering, it is gaming the metric - so the setting is only reported as a
    # solution alongside the rate it achieves it at, and a rate below ~2% on
    # these responses (median 20 tokens) means most answers get zero checkpoints.
    at_current = [g for g in grid
                  if g["amortised_ms_per_token"]["K=5 lookahead=10 (current)"] <= 15.0]
    cheapest = min(at_current, key=lambda g: g["min_gap"]) if at_current else None
    return {
        "note": "amortised = checkpoint cost x checkpoint rate; both levers priced "
                "against the same 101 cached generations",
        "grid": grid,
        "cheapest_setting_that_clears_15ms_at_current_cost": cheapest,
        "verdict": (
            "No setting clears 15 ms/token while the layer still runs on a "
            "meaningful share of tokens. The settings that clear it do so by "
            "checkpointing on under 1% of tokens, i.e. by not running - which is "
            "not a fix, it is a disabled feature reported as a passing benchmark. "
            "The binding constraint is the per-checkpoint cost (943 ms for K=5 on "
            "a laptop serving one slot), not the gate."
            if not at_current or cheapest["checkpoint_rate"] < 0.02 else
            f"min_gap={cheapest['min_gap']} clears the budget at a "
            f"{cheapest['checkpoint_rate']:.2%} checkpoint rate."
        ),
    }


# --------------------------------------------------------------------------- #

def print_report(rep: Dict[str, Any]) -> None:
    t = rep["tau_sweep"]
    print("\n" + "=" * 78)
    print(f"THRESHOLD SWEEP — {t['answerable_n']} answerable, {t['unanswerable_n']} unanswerable")
    print("=" * 78)
    print(f"{'dyn':>4} {'tau':>5} {'run':>4} {'FP':>8} {'FP rate':>8} {'FN':>7} {'FN rate':>8}  note")
    print("-" * 78)
    for r in t["grid"]:
        if r["breach_run"] not in (2, 3):
            continue
        note = ""
        if r["fp_combined"] == 0 and r["fn"] == 0:
            note = "<- clean separation"
        elif r["fp_combined"] == 0:
            note = "no false positives"
        print(f"{'Y' if r['dynamic_tau'] else 'N':>4} {r['tau']:>5.2f} {r['breach_run']:>4} "
              f"{r['fp_combined']:>4}/{r['answerable']:<3} {r['fp_rate']:>7.1%} "
              f"{r['fn']:>3}/{r['unanswerable']:<3} {r['fn_rate']:>7.1%}  {note}")

    # Eliminate false positives FIRST, then minimise misses among what is left.
    # The order encodes the asymmetry: falsely blocking a correct answer costs
    # the user's trust in the system, while a miss is caught downstream or not at
    # all - and a system nobody trusts catches nothing either way.
    zero_fp = [r for r in t["grid"] if r["fp_combined"] == 0]
    pool = zero_fp or t["grid"]
    floor = min((r["fp_rate"], r["fn_rate"]) for r in pool)
    tied = [r for r in pool if (r["fp_rate"], r["fn_rate"]) == floor]
    # Among equally-scoring settings take the MIDDLE tau, not an extreme. Every
    # tie is a plateau whose edges are where the estimate is least reliable - a
    # setting that only just avoids a false positive on 52 queries will not avoid
    # it on the 53rd. The middle is the one furthest from both cliffs.
    tied.sort(key=lambda r: (r["tau"], r["breach_run"]))
    best = tied[len(tied) // 2]
    print(f"\nRECOMMENDED  dynamic_tau={best['dynamic_tau']} tau={best['tau']} "
          f"breach_run={best['breach_run']}")
    print(f"  false positives {best['fp_combined']}/{best['answerable']} ({best['fp_rate']:.1%})"
          f" | misses {best['fn']}/{best['unanswerable']} ({best['fn_rate']:.1%})")
    print(f"  chosen from a plateau of {len(tied)} equally-scoring settings "
          f"(tau {tied[0]['tau']}-{tied[-1]['tau']}); the middle is furthest from both cliffs")
    if not zero_fp:
        print("  NOTE: no setting reaches zero false positives on this set.")
    # Which layer is actually doing the work. If the numeric layer alone accounts
    # for every false positive, no tau fixes it and the sweep is the wrong tool.
    fp_num = t["grid"][0]["fp_numeric"]
    print(f"  false positives attributable to the numeric layer alone: {fp_num}"
          f"/{best['answerable']} (independent of tau)")

    c = rep.get("clustering_ablation", {})
    if "grid" in c:
        print("\n" + "=" * 78)
        print(f"CLUSTERING ABLATION — sim_tau, over {c['n_sampled_queries']} sampled queries")
        print("=" * 78)
        print(f"{'numeric':>8} {'sim_tau':>8} {'H(ans)':>8} {'H(unans)':>9} {'separation':>11} "
              f"{'clusters a/u':>14}")
        print("-" * 78)
        for g in c["grid"]:
            print(f"{'ON' if g['numeric_override'] else 'OFF':>8} {g['sim_tau']:>8.2f} "
                  f"{g['answerable_mean_norm_entropy']:>8.3f} "
                  f"{g['unanswerable_mean_norm_entropy']:>9.3f} "
                  f"{g['separation']:>11.3f} "
                  f"{g['answerable_mean_clusters']:>6.2f}/{g['unanswerable_mean_clusters']:<7.2f}")
        b = c["best"]
        print(f"\nBEST separation: sim_tau={b['sim_tau']} numeric_override="
              f"{'ON' if b['numeric_override'] else 'OFF'} -> {b['separation']:.3f}")

    l = rep["latency"]
    print("\n" + "=" * 78)
    print("LATENCY — budget 15.000 ms/token")
    print("=" * 78)
    ao = l["always_on"]
    print(f"  always-on layers      {ao['combined_ms_per_token']:>9.4f} ms/token  median "
          f"(entropy {ao['entropy_us_per_token_median']:.1f} us + "
          f"numeric {ao['numeric_grounding_us_per_token_median']:.1f} us)")
    print(f"    numeric tail        p95 {ao['numeric_grounding_us_per_token_p95']/1000:.2f} ms/token, "
          f"max {ao['numeric_grounding_us_per_token_max']/1000:.2f} ms/token "
          f"(mean {ao['numeric_grounding_us_per_token_mean']/1000:.2f} - one lazy warm-up, see report)")
    print(f"  one K-path checkpoint {l['k_path_checkpoint']['median_ms']:>9.1f} ms "
          f"(p90 {l['k_path_checkpoint']['p90_ms']:.0f} ms, n={l['k_path_checkpoint']['n_measured']})")
    a = l["adaptive_checkpointing"]
    print(f"  checkpoint rate       {a['checkpoint_rate']:>9.2%} "
          f"({a['total_checkpoints']} over {a['total_tokens']} tokens, "
          f"1 per {a['tokens_per_checkpoint']} tokens)")
    print(f"  AMORTISED             {l['amortised_ms_per_token']:>9.3f} ms/token   "
          f"{'WITHIN BUDGET' if l['within_budget'] else '*** OVER BUDGET ***'}")
    if not l["within_budget"]:
        print(f"  Without the adaptive gate this layer would cost "
              f"{a['naive_every_token_ms_per_token']:.0f} ms/token; the gate divides that by "
              f"{a['saving_factor']:.0f}x and it is STILL over budget.")
        k = l["knobs"]
        print(f"\n  {'min_gap':>8} {'floor':>6} {'rate':>7} " +
              " ".join(f"{n.split(' ')[0]:>9}" for n in CHECKPOINT_COST_MS))
        print("  " + "-" * 66)
        for g in k["grid"]:
            if g["numeric_mass_floor"] != 0.25:
                continue
            cells = " ".join(f"{v:>9.1f}" for v in g["amortised_ms_per_token"].values())
            print(f"  {g['min_gap']:>8} {g['numeric_mass_floor']:>6.2f} "
                  f"{g['checkpoint_rate']:>6.2%} {cells}")
        best = k["cheapest_setting_that_clears_15ms_at_current_cost"]
        if best:
            print(f"\n  Clears 15 ms/token at the CURRENT checkpoint cost from min_gap="
                  f"{best['min_gap']} — but at a {best['checkpoint_rate']:.2%} rate "
                  f"(1 per {best['tokens_per_checkpoint']} tokens).")
        print(f"\n  VERDICT: {k['verdict']}")
        print("\n  Columns are per-checkpoint costs measured on an idle machine; the last is the\n"
              "  ceiling K=5 would reach with OLLAMA_NUM_PARALLEL raised (Ollama runs -np 1).")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-paths", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        if not RAW_PATH.exists():
            print(f"No cached collection at {RAW_PATH}; run without --report-only first.")
            return 1
        rows = json.loads(RAW_PATH.read_text())
        print(f"Re-sweeping {len(rows)} cached rows from {RAW_PATH}")
    else:
        print(f"Collecting {args.limit or len(calibration_set.full_set())} labelled queries…")
        rows = await collect(args.limit, want_paths=not args.no_paths)

    report = {
        "n_queries": len(rows),
        "max_new_tokens": MAX_TOKENS,
        "tau_sweep": sweep_tau(rows),
        "clustering_ablation": sweep_clustering(rows),
        "latency": latency_report(rows),
    }
    print_report(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
