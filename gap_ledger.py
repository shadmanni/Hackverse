"""
Frequency-weighted knowledge-gap queue: what the firewall keeps having to catch.

An interception is currently logged and forgotten. `data/audit_log.jsonl` proves
that SOME prompt recurred - it stores a SHA-256 of the query - but it cannot
prove that the same underlying GAP recurred, and those are different questions.
"What is the mean compliance cycle time?" and "give me the average compliance
cycle time" are one missing-data problem and two hashes, so counting hashes
reports 40 unrelated incidents where there is one ticket worth 40 hits. An
operator with limited curation time needs the second number.

The signature
-------------
Key on what the DETECTOR concluded, not on what the user typed.

When a metric bound, `SentinelStream._refutation` already carries the whole
finding: which metric the sentence named, which statistic of it, and the unit.
(metric, statistic, unit) is the gap - two differently-worded questions that
provoke the same wrong claim about the same statistic collapse onto one entry,
which is exactly the aggregation the query hash cannot do.

When nothing bound, the model was guessing about something the log has no
handle for at all - the genuinely-missing-data case, "what is the throughput of
warehouse node W-99". There is no detector-side key to use, so fall back to the
query's content words: lowercased, stopwords and digits dropped, sorted, joined.
Sorting makes word order irrelevant; dropping digits makes W-99 and W-42 the
same ticket, which is right - the gap is "no warehouse node data", and curating
one node without the others is not a fix.

Within one key the merge is lossy only in the safe direction - two gaps can
share a ticket, one gap cannot be split.

ponytail: ACROSS the two keys it CAN split, and does. Whether a metric binds
depends on whether the halted span happened to name it, so one recurring gap
lands under `metric:compliance cycle time|mean|duration` when the model wrote
the metric out, under `metric:cycle time|mean|duration` when it wrote a shorter
alias, and under `terms:compliance cycle mean time` when it wrote none - three
tickets, one problem, counts that individually understate it. Ceiling accepted
because the fix is alias canonicalisation in celonis_metrics (map every alias
of a metric to one id, then key on the id), which is that module's job, not
this one's. Until then read hit_count as a floor.
"""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGER_PATH = Path(__file__).parent / "data" / "knowledge_gaps.json"


def _path() -> Path:
    # The scripted interception tests replay one canned query dozens of times per
    # run: a single `pytest` took `terms:compliance cycle mean time` from 13 hits
    # to 26 in the real ledger. hit_count is shown to an operator as how often
    # this gap actually fires and /gaps ranks on it, so test traffic landing here
    # makes the queue order an artefact of how often the suite was run.
    # PYTEST_CURRENT_TEST is set by pytest for the duration of each test.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return Path(tempfile.gettempdir()) / "sentinel_gap_ledger_test.json"
    return LEDGER_PATH

# Function words only. Anything that could name a metric, an entity or an
# activity has to survive, or two different gaps normalise onto one signature.
_STOPWORDS = frozenset("""
a an the of for in on at to from by with and or is are was were be been being
what which who whom whose how why when where do does did can could should would
me my our us we you your i it its this that these those there here as if than
then so such please tell show give list any all some much many number
""".split())

_WORD_RE = re.compile(r"[a-z]+")


def _content_signature(query: str) -> str:
    """Word-order- and digit-insensitive form of a query. See module docstring."""
    # Single letters are what survives an identifier once digits are stripped
    # ("W-99" -> "w"), so they carry no meaning and would split the ticket by
    # whichever node was asked about.
    words = {w for w in _WORD_RE.findall(query.lower())
             if len(w) > 1 and w not in _STOPWORDS}
    return "terms:" + " ".join(sorted(words))


def signature_for(refutation: Optional[Dict[str, Any]], query: str) -> str:
    if refutation and refutation.get("claimed_metric"):
        # "any" rather than None, so an unbound statistic still prints as a
        # readable ticket line instead of "metric:cycle time|None|None".
        return "metric:{}|{}|{}".format(
            refutation["claimed_metric"],
            refutation.get("claimed_statistic") or "any",
            refutation.get("unit") or "any",
        )
    return _content_signature(query)


def _suggested_fix(refutation: Optional[Dict[str, Any]], reason: str) -> str:
    if refutation and refutation.get("claimed_metric"):
        stat = refutation.get("claimed_statistic") or "value"
        return (f"Publish the {stat} of '{refutation['claimed_metric']}' explicitly in the "
                f"verified-aggregate block; the log admits "
                f"{refutation.get('admissible')} and the model keeps asserting otherwise.")
    if reason == "semantic_entropy":
        return ("No grounded content for these terms - the model was guessing. Add source "
                "data or a retrieval chunk covering them, or an explicit refusal fact.")
    return ("A figure was refuted but bound to no metric. Add the entity or measure named "
            "here to celonis_metrics.metric_aliases so the claim can be checked by name.")


def _load() -> Dict[str, Dict[str, Any]]:
    try:
        return json.loads(_path().read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        # A corrupt or unreadable ledger must not take the stream down, and the
        # next record() legitimately overwrites it: an operator queue that cannot
        # be parsed has no value to preserve.
        print(f"[Sentinel] gap ledger unreadable ({err}); starting a fresh one.")
        return {}


def _save(gaps: Dict[str, Dict[str, Any]]) -> None:
    """Write via a temp file + os.replace, so a crash cannot truncate the ledger."""
    # Unlike the audit log this file IS rewritten whole on every breach, which is
    # the pattern log_compliance_breach avoids precisely because a half-written
    # rewrite loses everything. os.replace is atomic on the same filesystem, so a
    # reader sees either the old ledger or the new one.
    #
    # ponytail: atomic per writer, but read-modify-write by ONE writer, and the
    # ledger never prunes. Both are fine while the backend is the only process
    # (Milvus Lite already forces that - see CLAUDE.md). A second writer would
    # silently lose hits; move to SQLite before adding one.
    path = _path()
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(gaps, indent=2))
        os.replace(tmp, path)
    except OSError as err:
        print(f"[Sentinel] gap ledger write failed: {err}")


def record(reason: str, refutation: Optional[Dict[str, Any]], query: str,
           ungrounded_value: Optional[float] = None) -> Dict[str, Any]:
    """Count one interception against its gap. Returns the entry plus its rank."""
    sig = signature_for(refutation, query)
    now = datetime.now(timezone.utc).isoformat()
    gaps = _load()
    entry = gaps.get(sig) or {
        "signature": sig, "hit_count": 0, "first_seen": now, "example_queries": []
    }
    entry["hit_count"] += 1
    entry["last_seen"] = now
    entry["reason"] = reason
    entry["claimed_metric"] = (refutation or {}).get("claimed_metric")
    entry["claimed_statistic"] = (refutation or {}).get("claimed_statistic")
    entry["admissible_values"] = (refutation or {}).get("admissible")
    entry["ungrounded_value"] = ungrounded_value
    entry["suggested_fix"] = _suggested_fix(refutation, reason)

    # The SAME SHA-256 the audit log stores, not a truncated query. Truncating
    # leaks the head of the prompt, which is where the identifying detail usually
    # sits, and the audit log's docstring is explicit that reproducing the
    # questions people asked is itself a disclosure risk. Nothing readable is
    # lost: the signature above already says what the gap IS in plain words, and
    # the hash is the join key back to audit_log.jsonl for the timeline.
    qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if qhash not in entry["example_queries"]:
        entry["example_queries"] = (entry["example_queries"] + [qhash])[-3:]

    gaps[sig] = entry
    _save(gaps)
    return {**entry, "rank": [g["signature"] for g in _ranked(gaps)].index(sig) + 1}


def _ranked(gaps: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Plain hit_count desc: the claim is "spend curation effort where the firewall
    # fires most", and any weighted score would need a decay constant nobody here
    # can calibrate. Recency only breaks ties, via a stable second sort.
    by_recency = sorted(gaps.values(), key=lambda g: g["last_seen"], reverse=True)
    return sorted(by_recency, key=lambda g: -g["hit_count"])


def ranked(limit: int = 20) -> List[Dict[str, Any]]:
    """The priority queue: gaps the firewall hit most, worst first."""
    return _ranked(_load())[:limit]


def demo() -> None:
    """Self-check: two wordings of one question must be ONE gap with hit_count 2."""
    global LEDGER_PATH
    from sentinel_stream import SentinelStream

    real, LEDGER_PATH = LEDGER_PATH, Path(tempfile.mkdtemp()) / "knowledge_gaps.json"
    try:
        s = SentinelStream(runner=None)
        # Real refutations off the real detector, from two sentences that share
        # no wording beyond the metric name. If the signature keyed on the query
        # (or on the sentence) these would be two tickets of one hit each, which
        # is the failure this module exists to remove.
        a = s._refutation("The mean compliance cycle time is 12.5 days.")
        b = s._refutation("On average, the compliance cycle time comes to 14.2 days.")
        assert a == b, (a, b)          # same finding, different sentence and value
        record("ungrounded_number", a, "What is the mean compliance cycle time?", 12.5)
        hit = record("ungrounded_number", b, "give me the average compliance cycle time", 14.2)
        assert hit["hit_count"] == 2 and hit["rank"] == 1, hit
        assert len(ranked()) == 1, ranked()
        assert ranked()[0]["admissible_values"] == [10.4, 10.43], ranked()[0]
        # Two distinct queries, so both hashes are kept - and neither is readable.
        assert len(hit["example_queries"]) == 2, hit
        assert "compliance" not in str(hit["example_queries"]), hit

        # A different metric is a different ticket: merging must come from the
        # gap being the same, not from the signature being coarse.
        other = s._refutation("Compliance Review takes 16 days on average.")
        assert record("ungrounded_number", other, "how long is Compliance Review?", 16.0
                      )["hit_count"] == 1

        # Nothing bound: the genuinely-missing-data case. Word order and the node
        # number must not split the ticket.
        record("semantic_entropy", None, "What is the throughput of warehouse node W-99?")
        n = record("semantic_entropy", None, "warehouse node W-42 throughput - what is it?")
        assert n["hit_count"] == 2, n
        assert [g["hit_count"] for g in ranked()] == [2, 2, 1], ranked()

        # The guard that keeps the displayed count honest: a test run must not
        # be able to write the operator's ledger. Without this the suite added
        # 13 hits per run to a queue that claims to count real interceptions.
        os.environ["PYTEST_CURRENT_TEST"] = "probe"
        try:
            assert _path() != LEDGER_PATH, _path()
        finally:
            del os.environ["PYTEST_CURRENT_TEST"]
    finally:
        LEDGER_PATH = real
    print("gap_ledger self-check OK")


if __name__ == "__main__":
    demo()
