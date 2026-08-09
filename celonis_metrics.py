"""
Deterministic ground-truth metrics computed from the Celonis event log.

This module is the single authoritative source for every aggregate number the
system is allowed to state. mock_celonis_data*.json says it directly:

    "Any model-generated aggregate number not derivable from these events
     is a hallucination."

So we derive them here, once, and check the model against them. Nothing in the
UI or the API may hardcode a metric that this module can compute.
"""

import json
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data" / "mock_celonis_data_large.json"
HIGH_VALUE_USD = 100_000.0


@lru_cache(maxsize=4)
def load_events(path: str = str(DATA_PATH)) -> tuple:
    with open(path) as f:
        return tuple(json.load(f)["events"])


@lru_cache(maxsize=4)
def _raw(path: str = str(DATA_PATH)) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=4)
def process_profile(path: str = str(DATA_PATH)) -> Dict[str, Any]:
    """Every aggregate the system is permitted to assert, derived from the log."""
    doc = _raw(path)
    events = list(doc["events"])
    cycles = [e["cycle_time_days"] for e in events if e.get("cycle_time_days") is not None]
    amounts = [e["amount_usd"] for e in events if e.get("amount_usd") is not None]
    # An ORDER is a case, not an event. Counting the events kept saying "393
    # orders above $100,000" over a log of 150 cases - a figure the system fed
    # to Granite inside its own VERIFIED AGGREGATES block, and then accepted as
    # grounded when the model repeated it, because 393 was in the admissible
    # set. Both counts are exposed so a claim about either can be checked.
    high_value = [e for e in events if (e.get("amount_usd") or 0) >= HIGH_VALUE_USD]
    high_value_cases = {e["case_id"] for e in high_value if e.get("case_id")}
    hv_cycles = [e["cycle_time_days"] for e in high_value if e.get("cycle_time_days") is not None]

    by_activity: Dict[str, Dict[str, float]] = {}
    for ev in events:
        act = ev.get("activity")
        if act and ev.get("cycle_time_days") is not None:
            by_activity.setdefault(act, {"values": []})["values"].append(ev["cycle_time_days"])
    for act, blob in by_activity.items():
        vals = blob.pop("values")
        blob["mean_cycle_days"] = round(statistics.mean(vals), 2)
        blob["max_cycle_days"] = max(vals)
        # The minimum was withheld until the unit rule existed. Exposing it is
        # what lets a range bind both endpoints ("Compliance Review cycle times
        # range from 4 to 16 days" used to halt on 4), but on its own it also
        # made every activity's smallest cycle time admissible as ANY unqualified
        # figure for that activity, because metric_aliases is the union of a
        # metric's statistics. Measured then: Invoice Approved's minimum is 3,
        # which let "3 events are recorded for the Invoice Approved activity" -
        # a fabrication caught live on a green-path prompt, real count 150 - pass
        # again. figure_unit closes that hole from the other side: "3 events" is
        # a COUNT claim, so only count-valued statistics are admissible and a
        # duration can no longer stand in for one. See _STATISTIC_UNIT.
        blob["min_cycle_days"] = min(vals)
        blob["event_count"] = len(vals)

    # The declared block wins where it exists: it is the stated ground truth.
    declared = doc.get("summary_metrics", {})

    return {
        "process": doc.get("process", "Unknown Process"),
        "source_system": doc.get("source_system", "Celonis EMS"),
        "total_events": len(events),
        "total_cases": len({e["case_id"] for e in events if e.get("case_id")}),
        "mean_cycle_days": round(statistics.mean(cycles), 2) if cycles else 0.0,
        "median_cycle_days": round(statistics.median(cycles), 2) if cycles else 0.0,
        "max_cycle_days": max(cycles) if cycles else 0,
        "mean_amount_usd": round(statistics.mean(amounts), 2) if amounts else 0.0,
        "total_amount_usd": round(sum(amounts), 2) if amounts else 0.0,
        "high_value_orders": len(high_value_cases),
        "high_value_events": len(high_value),
        "high_value_mean_cycle_days": round(statistics.mean(hv_cycles), 2) if hv_cycles else 0.0,
        "by_activity": by_activity,
        "declared_avg_compliance_cycle_time_days": declared.get("avg_compliance_cycle_time_days"),
        "declared_avg_order_to_cash_days": declared.get("declared_avg_order_to_cash_days")
        or declared.get("avg_order_to_cash_days"),
        "declared_orders_flagged_for_bottleneck": declared.get("orders_flagged_for_bottleneck"),
        "declared_total_orders": declared.get("total_orders"),
    }


@lru_cache(maxsize=8)
def groundable_numbers(path: str = str(DATA_PATH), scope: str = "all") -> Set[float]:
    """
    Numbers a truthful answer may contain.

    scope="aggregate" - only derived and declared aggregates.
    scope="all"       - aggregates plus every raw per-event field value.

    The distinction is what makes the check useful. Cycle times run 1..23 days,
    so under scope="all" almost any small integer is "in the data" and the claim
    "the mean compliance cycle time is approximately 15 days" passes - 15 is some
    individual case's cycle time. But a claim ABOUT A MEAN is only true if it
    matches a mean. Scoping the comparison to aggregates is what catches it:
    15 is not any aggregate the log produces (the real value is 10.4).
    """
    prof = process_profile(path)
    nums: Set[float] = set()

    for key, val in prof.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            nums.add(round(float(val), 2))
    for blob in prof["by_activity"].values():
        nums.add(round(float(blob["mean_cycle_days"]), 2))
        nums.add(round(float(blob["max_cycle_days"]), 2))
        nums.add(round(float(blob["event_count"]), 2))

    # The high-value threshold itself. It is not a measurement, it is the
    # criterion - but SentinelStream._aggregate_block writes "Orders above
    # $100,000: 122 of 150" straight into the context, so the model quoting it
    # back is faithful reproduction of what we handed it. Leaving it out meant
    # the deterministic recovery answer, which also states it, was rejected by
    # its own checker. Any figure this system puts in front of the model has to
    # be groundable, or the check punishes accurate quotation.
    nums.add(round(float(HIGH_VALUE_USD), 2))

    # Percentages the log supports, e.g. share of orders flagged as bottlenecked.
    if prof["total_cases"]:
        flagged = prof["by_activity"].get("Supply Chain Bottleneck Flagged", {}).get("event_count", 0)
        nums.add(round(flagged / prof["total_cases"] * 100, 2))
        nums.add(round(100 - flagged / prof["total_cases"] * 100, 2))

    if scope == "aggregate":
        return nums

    for ev in load_events(path):
        for key in ("cycle_time_days", "amount_usd"):
            if ev.get(key) is not None:
                nums.add(round(float(ev[key]), 2))
    return nums


@lru_cache(maxsize=4)
def metric_statistics(path: str = str(DATA_PATH)) -> Dict[str, Dict[str, frozenset]]:
    """
    Phrase -> statistic -> the values that statistic may take.

    The single source of truth for metric binding; metric_aliases is the union
    of these, kept because most callers only need "is this value admissible for
    this metric at all".

    Why the statistic label is kept
    -------------------------------
    Binding a figure to a metric still cannot reject a claim about the WRONG
    STATISTIC of the RIGHT metric. "Compliance Review took 16 days on average"
    passed set-membership because 16 is that activity's MAX. The sentence says
    "on average"; the mean is 10.43. Both figures are real, one of them is a
    lie, and a set cannot tell them apart because it has forgotten which value
    was which.

    A statistic maps to a SET, not a scalar: "compliance cycle time" has both a
    declared mean (10.4) and a computed one (10.43), and a truthful answer may
    quote either.
    """
    p = process_profile(path)
    act = p["by_activity"]
    out: Dict[str, Dict[str, frozenset]] = {}

    def add(phrase: str, **stats):
        clean: Dict[str, frozenset] = {}
        for stat, v in stats.items():
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            s = {round(float(x), 2) for x in vals if x is not None}
            if s:
                clean[stat] = frozenset(s)
        if clean:
            out[phrase] = clean

    def activity(name: str) -> Dict[str, Any]:
        a = act.get(name, {})
        return {"mean": a.get("mean_cycle_days"), "max": a.get("max_cycle_days"),
                "min": a.get("min_cycle_days"), "count": a.get("event_count")}

    add("cycle time", mean=p["mean_cycle_days"], median=p["median_cycle_days"],
        max=p["max_cycle_days"])
    add("compliance cycle time",
        mean=[p["declared_avg_compliance_cycle_time_days"],
              act.get("Compliance Review", {}).get("mean_cycle_days")])
    add("compliance review", **activity("Compliance Review"))
    add("order-to-cash", mean=p["declared_avg_order_to_cash_days"])
    add("order to cash", mean=p["declared_avg_order_to_cash_days"])
    add("bottleneck",
        mean=act.get("Supply Chain Bottleneck Flagged", {}).get("mean_cycle_days"),
        # The one activity phrase that was missing its own maximum. The aggregate
        # block writes "Activity 'Supply Chain Bottleneck Flagged': 38 events,
        # mean cycle 12.47 days (max 20)", so the check rejected 20 - a value it
        # had just handed the model as fact - the moment anything bound that line
        # to this metric.
        max=act.get("Supply Chain Bottleneck Flagged", {}).get("max_cycle_days"),
        count=[act.get("Supply Chain Bottleneck Flagged", {}).get("event_count"),
               p["declared_orders_flagged_for_bottleneck"]],
        rate=round(act.get("Supply Chain Bottleneck Flagged", {}).get("event_count", 0)
                   / max(p["total_cases"], 1) * 100, 2))
    add("invoice approved", **activity("Invoice Approved"))
    add("purchase order", **activity("Purchase Order Created"))
    # `threshold` is the criterion the phrase names, not a measurement - but
    # "orders above $100,000" states it, and binding the figure to this metric
    # then checked $100,000 against {122, 9.11} and rejected the log's own
    # recovery answer. A metric's admissible set has to include the constant
    # that defines it, or naming the criterion becomes a hallucination.
    add("high-value", count=p["high_value_orders"],
        mean=p["high_value_mean_cycle_days"], threshold=HIGH_VALUE_USD)
    add("high value", count=p["high_value_orders"],
        mean=p["high_value_mean_cycle_days"], threshold=HIGH_VALUE_USD)
    # Longer forms so the specific metric beats the generic "cycle time" under
    # longest match - "high-value" is exactly as long as "cycle time" and the tie
    # was being resolved by dict iteration order, which is not a decision rule.
    add("high-value orders", count=p["high_value_orders"],
        mean=p["high_value_mean_cycle_days"], threshold=HIGH_VALUE_USD)
    add("high value orders", count=p["high_value_orders"],
        mean=p["high_value_mean_cycle_days"], threshold=HIGH_VALUE_USD)
    # Aliased by its defining criterion, which was held back until the unit rule
    # landed. "$100,000" names this metric and nothing else in the log, so
    # "The 122 orders in excess of $100,000 have a mean cycle time of 9.11 days"
    # - true in every figure - now binds instead of being checked against the
    # generic whole-log cycle time and halted on all three.
    #
    # The objection to adding it was "There were 393 orders above $100,000",
    # where 393 is the EVENT count and the sentence says orders: with only a
    # criterion to go on, the alias could not say which population was meant, so
    # binding it was guessing. figure_unit answers that: "orders" is a count, so
    # the claim is checked against count-valued statistics only, and the count
    # of that population is a thing the log actually knows.
    for phrase in ("$100,000", "100,000"):
        add(phrase, count=p["high_value_orders"],
            mean=p["high_value_mean_cycle_days"], threshold=HIGH_VALUE_USD)
    return out


# Aliases that name the DIMENSION rather than a metric. They exist so that
# "what is the mean cycle time?" binds to something at all, but every specific
# metric in the table is also a cycle time, so they must lose to any specific
# phrase in the same span. See bound_metric.
_GENERIC_ALIASES = frozenset({"cycle time"})

# Words that name WHICH statistic of a metric a sentence is claiming. Kept
# deliberately narrow: a phrase only lands here when it is unambiguous, because
# the cost of a wrong statistic binding is halting a correct answer, while the
# cost of missing one is only falling back to the behaviour that already exists.
_STATISTIC_WORDS: Dict[str, Tuple[str, ...]] = {
    "mean": ("mean", "average", "avg", "on average", "typically",
             # Adverbs of central tendency. They are unambiguous - none of them
             # can mark a count or an extremum - and without them "Compliance
             # Review usually takes 16 days" bound no statistic, so 16 passed on
             # set membership as that activity's MAX while the sentence asserts
             # its mean of 10.43.
             "usually", "normally", "generally", "typical"),
    "median": ("median",),
    "max": ("maximum", "longest", "slowest", "worst", "at most", "peak", "up to",
            # "highest" is the ordinary superlative for a duration and was absent,
            # so "The highest cycle time was 8.5 days" quoted the MEAN unchallenged.
            "highest"),
    # Bare "count" earns its place by creating AMBIGUITY, not by binding. In
    # "Compliance Review has the highest event count at 122" the superlative
    # belongs to the count, not to a duration - but only "highest" matched, so
    # 122 was checked against that activity's max of 16 and a true sentence
    # halted. With both words present bound_statistic sees two candidates and
    # returns None, which restores the wider whole-metric check. Declining to
    # guess is the correct answer to a genuinely ambiguous sentence.
    # "times" is the one predicate form of a count narrow enough to be safe:
    # "'Compliance Review' occurs 16 times" asserts a count, and 16 is that
    # activity's max. "events" was tried and rejected - it appears in almost
    # every statistic span in this domain, so it bound counts to means.
    # "times" was here and has moved to _UNIT_WORDS. As a statistic word it was
    # matched anywhere in the span, so "cycle times" - a plural DURATION - bound
    # the count statistic, and "Compliance Review cycle times range from 4 to 16
    # days" checked 4 against that activity's event count of 122. As a unit word
    # it only matches directly after a figure, where "occurs 16 times" still
    # reads as a count and "cycle times range from" no longer does.
    "count": ("number of", "how many", "count of", "total of", "count"),
    "rate": ("percentage", "percent", "proportion", "share of", "rate of"),
}

# Matched on WORD boundaries. Substring matching put "count" inside "accounts",
# so "Compliance Review accounts for 122 of the 460 events in the log" bound the
# count statistic to 460 - the log's total event count, stated correctly - and
# halted on it, because that activity's own count is 122. The same class of bug
# had already been fixed once in _QUALIFIER_BEFORE, where "over" matched inside
# "cover"; "discount" against "count" is the next one waiting in this domain.
_STATISTIC_PATTERNS: Dict[str, "re.Pattern"] = {
    stat: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b",
                     re.IGNORECASE)
    for stat, words in _STATISTIC_WORDS.items()
}

# What KIND of quantity each statistic produces. Two statistics of the same
# metric can hold the same number while meaning different things - Compliance
# Review has a max of 16 days and a count of 122 events - so a set that has
# forgotten which was which accepts "16 events" by quoting the max.
_STATISTIC_UNIT: Dict[str, str] = {
    "mean": "duration", "median": "duration", "max": "duration",
    "min": "duration", "count": "count", "rate": "rate", "threshold": "money",
}

# The noun that follows a figure names the unit it is measured in. This is a
# deliberately LOCAL signal - it reads only the few characters after the digits,
# unlike the statistic words, which are searched across the whole clause. That
# is why "events" works here and was rejected there: in a span search it appears
# in almost every sentence in this domain, but directly after a figure it is
# unambiguous.
_UNIT_WORDS: Dict[str, Tuple[str, ...]] = {
    "duration": ("days", "day", "hours", "hour", "hrs", "weeks", "week",
                 "months", "month", "minutes", "business days"),
    "count": ("cases", "case", "orders", "order", "events", "event", "times",
              "records", "record", "invoices", "invoice", "activities",
              "instances", "occurrences", "steps"),
}
_UNIT_AFTER = re.compile(
    r"^\s*(?:%\s*)?(" + "|".join(
        re.escape(w) for ws in _UNIT_WORDS.values() for w in sorted(ws, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)
_UNIT_OF = {w: unit for unit, ws in _UNIT_WORDS.items() for w in ws}


def figure_unit(figure: str, tail: str) -> Optional[str]:
    """
    The unit a figure is stated in: 'rate', 'money', 'duration', 'count', or
    None when the sentence does not say.

    `figure` is the matched numeral (which carries its own '%' or '$'), `tail`
    the text immediately after it. Only the unit NOUN is read, not the wider
    sentence - see _UNIT_WORDS for why the scope has to stay this tight.
    """
    if "%" in figure:
        return "rate"
    if "$" in figure:
        return "money"
    m = _UNIT_AFTER.match(tail or "")
    if m:
        return _UNIT_OF[m.group(1).lower()]
    # "38 percent of orders" - the unit is spelled rather than punctuated.
    if re.match(r"^\s*per\s?cent", tail or "", re.IGNORECASE):
        return "rate"
    return None


def bound_statistic(text: str, window: int = 200, unit: Optional[str] = None) -> Optional[str]:
    """
    Which statistic a claim names, if exactly one is unambiguous.

    Returns None when nothing matches OR when several do ("the mean and the
    maximum both..."), because an ambiguous binding would be enforced as if it
    were certain. None restores the pre-existing whole-metric check, so the
    fallback is never stricter than what shipped before.

    `unit` discards statistic words that cannot describe a figure of that kind
    before the ambiguity test is applied - a count is never a mean. That both
    resolves sentences carrying one statistic word per unit ("the mean cycle
    time ... across 122 events" names `mean`, but not for 122) and lets the
    caller search a wider span than the figure's own clause, since the words it
    might wrongly pick up out there are largely ones this filter removes.
    """
    recent = text[-window:].lower()
    hits = {s for s, pat in _STATISTIC_PATTERNS.items() if pat.search(recent)}
    if unit:
        hits = {s for s in hits if _STATISTIC_UNIT.get(s) == unit}
    return hits.pop() if len(hits) == 1 else None


def metric_aliases(path: str = str(DATA_PATH)) -> Dict[str, frozenset]:
    """
    Phrase -> the values that phrase may legitimately take.

    Checking a figure against the whole aggregate set is too weak, because the
    aggregates are densely packed. Granite claimed "the mean compliance cycle
    time is 12.5 days"; the real value is 10.4, but 12.5 is a valid rounding of
    12.47 - the SUPPLY CHAIN BOTTLENECK mean - so a set-membership test accepted
    it. Tolerance cannot separate those: 12.5 rounds from 12.47 exactly as
    legitimately as 10.4 rounds from 10.43.

    The claim has to be bound to the metric it names. When a sentence names one
    of these phrases, only that metric's values are admissible.

    The unqualified "cycle time" entry matters: without it "the mean cycle time
    is 12.7 days" bound to nothing, so 12.7 was accepted for landing within 2%
    of 12.47 - the SUPPLY CHAIN BOTTLENECK mean - while the real overall mean is
    8.5. bound_metric takes the LONGEST match, so a sentence naming a specific
    activity still binds to that activity rather than to the generic phrase.

    Derived from metric_statistics so the two cannot drift apart. This is the
    "any value for this metric" view; pass a statistic to is_grounded_number
    when the sentence says which one it means.
    """
    return {
        phrase: frozenset().union(*stats.values())
        for phrase, stats in metric_statistics(path).items()
    }


@lru_cache(maxsize=8)
def _log_wide(path: str = str(DATA_PATH), statistic: Optional[str] = None) -> frozenset:
    """
    Facts about the LOG rather than about any metric: how many cases it holds,
    how many events, and the span its cycle times cover.

    These are admissible whatever metric a sentence happens to name, and binding
    made them inadmissible. "150 cases are in the order-to-cash process" halted
    on 150 - the true case count - because "order-to-cash" narrowed the set to
    that metric's single declared value of 12.0. Measured live: it is what
    Granite answers to "How many cases are in the order-to-cash process?", which
    is a question a judge asks in the first minute, and the halt landed on a
    correct answer.

    The same shape covers ranges ("cycle times range between 1 and 23 days") -
    the minimum is a real property of the log that no metric's value set holds.

    Withheld when the sentence claims a mean, median or max, because there the
    figure IS being asserted as that statistic of that metric and the narrow set
    is the whole point: "the mean compliance cycle time is 150 days" must still
    fail. Counts and unqualified statements get the exemption; statistics do not.
    """
    if statistic in {"mean", "median", "max", "rate"}:
        return frozenset()
    p = process_profile(path)
    cycles = [e["cycle_time_days"] for e in load_events(path)
              if e.get("cycle_time_days") is not None]
    vals = {p["total_cases"], p["total_events"], p["high_value_events"]}
    if cycles:
        vals |= {min(cycles), max(cycles)}
    return frozenset(round(float(v), 2) for v in vals if v is not None)


def bound_metric(text: str, window: int = 200) -> Optional[str]:
    """
    The metric phrase a claim names, if any. Longest match wins; on a tie, the
    phrase nearest the figure wins.

    The tie-break is not cosmetic. "Across 122 high-value orders above $100,000,
    the mean cycle time is 9.11 days" matches both "high-value" and "cycle time"
    at exactly ten characters, and the two admit disjoint values - 9.11 is the
    high-value mean, while the generic "cycle time" admits only the overall 8.5.
    With max() breaking the tie by dict order, whether the log's own recovery
    answer passed its own check depended on the insertion order of an alias
    table. Nearest-wins resolves it the way the sentence reads: the qualifier
    closest to the figure is the one modifying it.
    """
    recent = text[-window:].lower()
    hits = [(len(k), recent.rfind(k), k) for k in metric_aliases() if k in recent]
    if not hits:
        return None
    # A generic phrase only binds when nothing specific is named. "cycle time"
    # is a DIMENSION, not a metric - every activity metric in the table is a
    # cycle time - so it co-occurs with the specific phrase in almost every real
    # sentence and neither length nor position separates them:
    #
    #   "'Supply Chain Bottleneck Flagged' is the biggest bottleneck with a mean
    #    cycle time of 12.47"
    #
    # "bottleneck" and "cycle time" are both ten characters, and "cycle time"
    # sits nearer the figure, so it won on both rules and 12.47 - that
    # activity's correct mean - was checked against the overall 8.5 and halted.
    # The recovery then answered with 12.47, so the system contradicted itself
    # on screen. Tiering resolves it by specificity, which is what the sentence
    # actually means.
    specific = [h for h in hits if h[2] not in _GENERIC_ALIASES]
    return max(specific or hits)[2]


def bound_metrics(text: str, window: int = 200) -> list:
    """
    EVERY metric phrase the claim names, best-first (same ordering as
    bound_metric, whose answer is this list's head).

    One sentence can name two metrics, and then no single binding is correct:

        "Compliance Review averages 10.43 days, while the bottleneck step
         averages 12.47 days."

    Binding both figures to the winner rejects 12.47 - the bottleneck mean,
    true - for not being a Compliance Review value. Picking the nearest metric
    instead breaks the opposite case ("...occurs 122 times with a mean cycle
    time of 10.43", where the nearest phrase is the generic "cycle time" and the
    correct binding is the activity named earlier). Neither rule is right,
    because the ambiguity is real.

    So the caller checks the figure against every metric the sentence names and
    accepts it if ANY of them admits it. A figure that fits none still fails -
    12.5 is not a value of any metric in "the mean compliance cycle time is
    12.5 days" - so this loses no detection. It only declines to halt on an
    ambiguity the sentence genuinely contains.
    """
    recent = text[-window:].lower()
    hits = [(len(k), recent.rfind(k), k) for k in metric_aliases() if k in recent]
    if not hits:
        return []
    specific = [h for h in hits if h[2] not in _GENERIC_ALIASES]
    # Specific phrases first, each ordered as bound_metric would rank them.
    return [h[2] for h in sorted(specific, reverse=True)] + [
        h[2] for h in sorted((h for h in hits if h[2] in _GENERIC_ALIASES), reverse=True)
    ]


def names_specific_metric(text: str, window: int = 200) -> bool:
    """
    True when `text` names a metric that is more than a bare dimension.

    Used to decide how far back a metric span has to reach: a clause that names
    only "cycle time" has not really said which metric it means, so the title or
    parenthetical before it is still in play.
    """
    recent = text[-window:].lower()
    return any(k in recent for k in metric_aliases() if k not in _GENERIC_ALIASES)


def is_generic_metric(metric: Optional[str]) -> bool:
    """True when the phrase names a dimension rather than a specific metric."""
    return metric in _GENERIC_ALIASES


def is_grounded_number(
    value: float,
    path: str = str(DATA_PATH),
    rel_tol: float = 0.02,
    scope: str = "all",
    metric: Optional[str] = None,
    statistic: Optional[str] = None,
    unit: Optional[str] = None,
) -> bool:
    """
    True when `value` matches a number the event log can produce, within 2%.

    The tolerance absorbs the model's rounding ("8.5" for 8.50), not fabrication:
    a hallucinated figure is wrong by far more than 2% essentially always.

    Pass scope="aggregate" when the surrounding sentence is asserting a mean,
    total, rate or percentage. See groundable_numbers for why that matters.

    Pass metric=<phrase from metric_aliases> when the sentence names a specific
    metric. That narrows the admissible set to that metric's own values, which
    is the only way to reject a figure that is real but belongs to a different
    metric - see metric_aliases for the case that motivated it.

    Pass statistic=<key from metric_statistics> when the sentence also says
    WHICH statistic of that metric it means ("on average", "the longest"). That
    is the only way to reject a figure that is real, and belongs to the right
    metric, and is still a lie because it is the wrong statistic of it:
    "Compliance Review took 16 days on average" quotes that activity's MAX.

    Pass unit=<figure_unit(...)> when the sentence says what the figure is
    MEASURED IN. A statistic word is often absent, and then the union of a
    metric's statistics is the admissible set - which is how "Compliance Review
    has 16 events" passed by matching that activity's max of 16 days. The unit
    is the weaker but far more available signal: it never says which statistic
    is meant, but it does say which ones are impossible, and dropping the
    duration-valued statistics for a claim counted in events is enough. Where
    both are bound the statistic wins, being the more precise of the two.

    The narrowing applies only when a metric is bound and it actually carries a
    statistic of the named kind. Any other combination falls back to the wider
    check, so this can reject claims the old code accepted but never accepts
    fewer - no threshold moved, and nothing that passed before on a correctly
    stated figure passes differently now.
    """
    target = round(float(value), 2)
    by_stat = metric_statistics(path).get(metric) if metric else None
    same_unit = ({s: v for s, v in by_stat.items() if _STATISTIC_UNIT.get(s) == unit}
                 if by_stat and unit else {})
    if by_stat and statistic and statistic in by_stat:
        candidates = by_stat[statistic]
    elif same_unit:
        # _log_wide still applies: the log's own case and event counts are
        # admissible for a count claim whatever metric the sentence names.
        candidates = frozenset().union(*same_unit.values()) | _log_wide(path, statistic)
    elif metric and metric in metric_aliases(path):
        candidates = metric_aliases(path)[metric] | _log_wide(path, statistic)
    else:
        candidates = groundable_numbers(path, scope)
    for known in candidates:
        if known == target:
            return True
        scale = max(abs(known), abs(target), 1e-9)
        if abs(known - target) / scale <= rel_tol:
            return True
    return False


def ground_truth_answer(query: str, path: str = str(DATA_PATH)) -> str:
    """Deterministic answer assembled from the log, used for autonomous recovery."""
    p = process_profile(path)
    q = (query or "").lower()

    if "high-value" in q or "high value" in q or "100,000" in q or "100000" in q:
        # Deliberately does NOT name the process here. The title is
        # "Order-to-Cash (O2C) Compliance Cycle", so "in the <process> log"
        # dropped the phrase "order-to-cash" into the same sentence as the
        # figure - and metric binding then checked 9.11, the high-value mean,
        # against order-to-cash's only value of 12.0 and rejected it. The
        # system's own deterministic answer failed its own checker.
        return (
            f"Across {p['high_value_orders']} high-value orders above "
            f"${HIGH_VALUE_USD:,.0f}, the mean cycle time is "
            f"{p['high_value_mean_cycle_days']} days."
        )
    if "bottleneck" in q or "delay" in q:
        blob = p["by_activity"].get("Supply Chain Bottleneck Flagged", {})
        return (
            f"{blob.get('event_count', 0)} of {p['total_cases']} cases were flagged as supply-chain "
            f"bottlenecks, adding a mean {blob.get('mean_cycle_days', 0)} days."
        )
    for activity, blob in p["by_activity"].items():
        if activity.lower().split()[0] in q:
            return (
                f"'{activity}' occurs {blob['event_count']} times in the log with a mean "
                f"cycle time of {blob['mean_cycle_days']} days (max {blob['max_cycle_days']})."
            )
    return (
        f"{p['process']} ({p['source_system']}): {p['total_cases']} cases / {p['total_events']} events. "
        f"Mean cycle time {p['mean_cycle_days']} days, median {p['median_cycle_days']}, "
        f"declared compliance cycle time {p['declared_avg_compliance_cycle_time_days']} days."
    )


if __name__ == "__main__":
    p = process_profile()
    print(json.dumps(p, indent=2))
    # The number the demo used to hardcode is NOT in the data; the declared one is.
    assert not is_grounded_number(4.2), "4.2 should not be groundable"
    assert is_grounded_number(p["declared_avg_compliance_cycle_time_days"])
    assert is_grounded_number(p["mean_cycle_days"])
    assert not is_grounded_number(42.8), "fabricated demo figure should not be groundable"

    # The unit rule. Compliance Review's max is 16 DAYS and its count is 122
    # EVENTS; without a unit both numbers are admissible for either claim.
    assert figure_unit("16", " events in the log") == "count"
    assert figure_unit("16", " days on average") == "duration"
    assert figure_unit("38%", " of orders") == "rate"
    assert figure_unit("$100,000", "") == "money"
    assert figure_unit("16", " and rising") is None
    assert is_grounded_number(16, metric="compliance review", unit="duration")
    assert not is_grounded_number(16, metric="compliance review", unit="count"), \
        "a count claim must not be satisfied by that metric's maximum in days"
    assert is_grounded_number(122, metric="compliance review", unit="count")
    # Word boundaries: "count" must not match inside "accounts".
    assert bound_statistic("Compliance Review accounts for 460 events") is None

    print("\nself-check OK: 4.2 and 42.8 ungroundable; declared + derived metrics "
          "groundable; unit narrowing separates 16 days from 16 events")
