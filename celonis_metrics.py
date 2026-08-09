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
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

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
    high_value = [e for e in events if (e.get("amount_usd") or 0) >= HIGH_VALUE_USD]
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
        "high_value_orders": len(high_value),
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


def is_grounded_number(
    value: float,
    path: str = str(DATA_PATH),
    rel_tol: float = 0.02,
    scope: str = "all",
) -> bool:
    """
    True when `value` matches a number the event log can produce, within 2%.

    The tolerance absorbs the model's rounding ("8.5" for 8.50), not fabrication:
    a hallucinated figure is wrong by far more than 2% essentially always.

    Pass scope="aggregate" when the surrounding sentence is asserting a mean,
    total, rate or percentage. See groundable_numbers for why that matters.
    """
    target = round(float(value), 2)
    for known in groundable_numbers(path, scope):
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
        return (
            f"Across {p['high_value_orders']} orders above ${HIGH_VALUE_USD:,.0f} in the "
            f"{p['process']} log, the mean cycle time is {p['high_value_mean_cycle_days']} days."
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
    print("\nself-check OK: 4.2 and 42.8 ungroundable; declared + derived metrics groundable")
