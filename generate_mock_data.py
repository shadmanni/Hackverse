"""
Generates a larger, more realistic synthetic Celonis-style event log for
load/integration testing, instead of the small 9-event demo file.

Key design choice: summary_metrics is COMPUTED from the generated events,
not hand-typed. A project whose entire premise is "don't state numbers
that aren't true" would look pretty bad shipping a ground-truth file
whose own aggregate stats don't match its own events - so this script
guarantees they always match.

Usage:
    python generate_mock_data.py                     # 150 cases (default)
    python generate_mock_data.py --cases 500          # bigger
    python generate_mock_data.py --out data/mock_celonis_data_large.json
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)  # reproducible across teammates' machines

RESOURCES = [
    ("Anita Rao", "anita.rao@acmecorp.com", "+91-98765-43210", "Procurement"),
    ("Deepak Menon", "deepak.menon@acmecorp.com", "+91-99887-11223", "Legal & Compliance"),
    ("Priya Sharma", "priya.sharma@acmecorp.com", "+91-97654-32109", "Finance"),
    ("Karan Mehta", "karan.mehta@acmecorp.com", "+91-90000-11122", "Procurement"),
    ("Sneha Iyer", "sneha.iyer@acmecorp.com", "+91-98123-45678", "Supply Chain"),
    ("Rahul Verma", "rahul.verma@acmecorp.com", "+91-96543-21098", "Finance"),
    ("Meera Nair", "meera.nair@acmecorp.com", "+91-95432-10987", "Legal & Compliance"),
    ("Arjun Kapoor", "arjun.kapoor@acmecorp.com", "+91-94321-09876", "Supply Chain"),
]

COST_CENTERS = ["CC-4471", "CC-2210", "CC-3305", "CC-1188", "CC-5560"]

BOTTLENECK_NOTES = [
    "Vendor delay at warehouse node W-14 added {d} days to cycle time.",
    "Customs clearance hold added {d} days before release.",
    "Awaiting secondary approval from {name} ({email}) - added {d} days.",
    "Escalated to {name} (contact {phone}) due to missing documentation.",
]

COMPLIANCE_NOTE = (
    "Compliance cycle time is {avg} days on average for orders above $100,000."
)


def gen_case(case_id: int):
    resource_pool = random.sample(RESOURCES, k=3)
    cost_center = random.choice(COST_CENTERS)
    amount = round(random.uniform(8_000, 450_000), 2)
    events = []

    # 1. Purchase Order Created
    r = resource_pool[0]
    po_days = random.randint(1, 4)
    events.append({
        "case_id": f"CASE-{case_id}",
        "activity": "Purchase Order Created",
        "timestamp": f"2026-0{random.randint(1,7)}-{random.randint(1,28):02d}T{random.randint(6,18):02d}:{random.randint(0,59):02d}:00Z",
        "resource": r[0], "resource_email": r[1], "resource_phone": r[2],
        "department": r[3], "cost_center": cost_center,
        "amount_usd": amount, "cycle_time_days": po_days,
    })

    running_days = po_days

    # 2. Compliance Review (only for orders > $100k, mirrors real business rule)
    if amount > 100_000:
        r = resource_pool[1]
        review_days = running_days + random.randint(3, 12)
        note = COMPLIANCE_NOTE.format(avg="{avg}")  # filled in after computing global avg
        events.append({
            "case_id": f"CASE-{case_id}",
            "activity": "Compliance Review",
            "timestamp": f"2026-0{random.randint(1,7)}-{random.randint(1,28):02d}T{random.randint(6,18):02d}:{random.randint(0,59):02d}:00Z",
            "resource": r[0], "resource_email": r[1], "resource_phone": r[2],
            "department": r[3], "cost_center": cost_center,
            "amount_usd": amount, "cycle_time_days": review_days,
            "note": note,
            "_is_compliance": True,
        })
        running_days = review_days

    # 3. Occasional bottleneck flag (~25% of cases) - deliberately embeds
    #    PII in the note so redaction has something real to catch.
    flagged = random.random() < 0.25
    if flagged:
        r = resource_pool[2]
        delay = random.randint(2, 6)
        running_days += delay
        template = random.choice(BOTTLENECK_NOTES)
        note = template.format(d=delay, name=r[0], email=r[1], phone=r[2])
        events.append({
            "case_id": f"CASE-{case_id}",
            "activity": "Supply Chain Bottleneck Flagged",
            "timestamp": f"2026-0{random.randint(1,7)}-{random.randint(1,28):02d}T{random.randint(6,18):02d}:{random.randint(0,59):02d}:00Z",
            "resource": r[0], "resource_email": r[1], "resource_phone": r[2],
            "department": r[3], "cost_center": cost_center,
            "amount_usd": amount, "cycle_time_days": running_days,
            "note": note,
        })

    # 4. Invoice Approved (final step, always present)
    r = resource_pool[0] if resource_pool[0][3] == "Finance" else next(
        (x for x in resource_pool if x[3] == "Finance"), RESOURCES[2]
    )
    running_days += random.randint(1, 3)
    events.append({
        "case_id": f"CASE-{case_id}",
        "activity": "Invoice Approved",
        "timestamp": f"2026-0{random.randint(1,7)}-{random.randint(1,28):02d}T{random.randint(6,18):02d}:{random.randint(0,59):02d}:00Z",
        "resource": r[0], "resource_email": r[1], "resource_phone": r[2],
        "department": r[3], "cost_center": cost_center,
        "amount_usd": amount, "cycle_time_days": running_days,
    })

    return events, flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=150)
    parser.add_argument("--out", type=str, default="data/mock_celonis_data_large.json")
    args = parser.parse_args()

    all_events = []
    flagged_count = 0
    compliance_days = []
    o2c_days = []  # final cycle_time_days per case = order-to-cash time

    for i in range(1, args.cases + 1):
        case_events, flagged = gen_case(10000 + i)
        all_events.extend(case_events)
        if flagged:
            flagged_count += 1
        for e in case_events:
            if e.get("_is_compliance"):
                compliance_days.append(e["cycle_time_days"])
        o2c_days.append(case_events[-1]["cycle_time_days"])

    avg_compliance = round(sum(compliance_days) / len(compliance_days), 1) if compliance_days else 0
    avg_o2c = round(sum(o2c_days) / len(o2c_days), 1)

    # backfill the compliance note with the real computed average, and
    # strip the internal-only "_is_compliance" flag
    for e in all_events:
        if e.pop("_is_compliance", False):
            e["note"] = e["note"].format(avg=avg_compliance)

    payload = {
        "process": "Order-to-Cash (O2C) Compliance Cycle",
        "source_system": "Celonis EMS - Simulated Export (scaled/generated)",
        "extraction_timestamp": "2026-08-09T00:00:00Z",
        "events": all_events,
        "summary_metrics": {
            "avg_compliance_cycle_time_days": avg_compliance,
            "avg_order_to_cash_days": avg_o2c,
            "orders_flagged_for_bottleneck": flagged_count,
            "total_orders": args.cases,
            "note_ground_truth": (
                "This block is the ONLY authoritative source for aggregate "
                "metrics. Any model-generated aggregate number not derivable "
                "from these events is a hallucination."
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(all_events)} events across {args.cases} cases -> {out_path}")
    print(f"avg_compliance_cycle_time_days = {avg_compliance}")
    print(f"avg_order_to_cash_days = {avg_o2c}")
    print(f"orders_flagged_for_bottleneck = {flagged_count} / {args.cases}")


if __name__ == "__main__":
    main()