"""
Semantic entropy at token granularity: is the model unsure about WORDING, or
about WHAT IS TRUE?

Why this exists
---------------
The entropy layer scored Shannon entropy over the raw top-k probabilities. That
number conflates two situations that could not be more different:

    candidates: "is" 0.31  "was" 0.28  "has" 0.22  "remains" 0.11  "'s" 0.08
        -> H = 2.16 bits, and it means nothing. Every candidate expresses the
           same fact. The model is picking a verb form.

    candidates: "12" 0.31  "15" 0.28  "9" 0.22  "10" 0.11  "4" 0.08
        -> H = 2.16 bits, identical number, and it is a fabrication in progress.
           The candidates are mutually exclusive claims about the world. The
           model does not know the figure; it is about to pick one and state it
           with the fluency of a fact.

Token entropy cannot tell those apart, which is why it needed a run of
consecutive breaches to be usable at all, and why it still misses the confident
single-token fabrication. Clustering the candidates BY MEANING before taking the
entropy separates them: case one collapses to a single cluster (H_sem = 0), case
two stays five clusters (H_sem = 2.16).

What this actually decides, and what it does not
------------------------------------------------
Deciding whether "has" and "remains" express the same proposition is
context-dependent entailment. It needs an NLI model over full generations -
exactly the expensive machinery this module exists to avoid - and no lookup
table approximates it. An earlier version tried, by treating a stopword list as
the equivalence relation; it scored the wording case above at 1.4 bits instead
of 0, because "has" and "remains" are not stopwords. The claim was false and the
self-check caught it.

So the question is narrowed to the one the candidate set can actually answer:
**is the model choosing between mutually exclusive VALUES, or between ways of
writing prose?** Figures cluster by value, everything else is one cluster. That
makes the wording case collapse to H = 0 by construction rather than by luck,
and it is the only reading the numbers below support.

Measured on live granite3.3:8b. On the "12.5" interception, frame tokens
averaged H = 0.022 bits against 0.994 for digit tokens - a 45x separation, with
token 10 close to a coin flip between 12.5 and 12.7.

But do not read that as "the model rolls dice on every digit", which a second
live run refutes: on the W-99 discount prompt it sat at 2:0.90 5:0.08, flatly
confident, and the figure was still one the log cannot produce. Uncertainty is
CONCENTRATED at figure positions and is NOT uniform across them.

That asymmetry is the argument for two layers rather than one. Where the model
splits, this signal sees it a token before the digit lands. Where the model is
confident and wrong, no entropy threshold can see it at all, and only the
deterministic check against the event log will - which is the enterprise case,
because a fabricated figure stated at p=0.90 is exactly the one that gets
believed.

Relation to the published result
--------------------------------
Farquhar et al. (doi:10.1038/s41586-024-07421-0) cluster whole sampled responses
by bidirectional entailment and take the entropy over those clusters. That needs
N full generations per answer and an NLI model to compare them. This applies the
same idea one decode step lower down: the candidate set is the top-k the decoder
already produced, and the equivalence relation is numeric rather than
entailment-based. Cheaper and much coarser, and honest about which it is - the
win is that the distribution is a BYPRODUCT of the generation that already
happened, so it costs no extra sample, no extra model and no extra forward pass.

Two distinct numbers in the candidate set, with the mass split between them, is
a dice roll on a figure - visible BEFORE the digit is emitted, which is one
token earlier than any check that inspects the number after the fact, and works
even where the event log holds no ground truth to compare against.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

_STRIP = ".,;:!?()[]{}\"'`$%"

PROSE = "<prose>"        # every candidate that is not a figure


def cluster_key(token: str) -> str:
    """
    Equivalence key for one candidate token, with respect to a NUMERIC claim.

    Two classes:
      * a figure     -> "#<value>", so two different numbers never share a cluster
      * anything else -> PROSE

    ponytail: all non-figures collapse into one cluster, so a fabricated
    *name* ("Munich" vs "Berlin" as candidates) reads as H = 0 here and is
    invisible to this signal. Catching that needs entailment over sampled
    generations; the numeric-grounding layer in sentinel_stream covers the
    figures, which is where the enterprise damage is. Upgrade path is an NLI
    clusterer over the top-k, at the cost of a model call per token.
    """
    t = (token or "").strip()
    if not t:
        return PROSE
    core = t.strip(_STRIP)
    if core and any(c.isdigit() for c in core) and all(c.isdigit() or c in ".," for c in core):
        return "#" + core.rstrip(".,").replace(",", "")
    return PROSE


@dataclass
class Divergence:
    """What the model was actually choosing between at one decoding step."""
    token_entropy: float                       # H over raw candidates (the old measure)
    semantic_entropy: float                    # H over value clusters
    clusters: List[Tuple[str, float]]          # (label, mass), heaviest first
    numeric_options: List[Tuple[str, float]] = field(default_factory=list)
    numeric_mass: float = 0.0
    divergent: bool = False                    # rolling dice on a figure

    def explain(self) -> str:
        """One sentence a non-specialist can read off the screen."""
        if self.divergent:
            opts = ", ".join(f"{v} ({p:.0%})" for v, p in self.numeric_options[:4])
            return (
                f"the model was choosing between {len(self.numeric_options)} different "
                f"figures - {opts} - so it did not know the value; it was about to pick one"
            )
        if self.numeric_options and self.numeric_mass >= 0.55:
            # The common case for a real fabrication, and the one worth saying
            # out loud: the model is not hedging at all. It is certain, and
            # certain of a figure the log cannot produce. This is precisely what
            # an entropy threshold cannot catch, and why the numeric-grounding
            # layer is not redundant with it.
            top, share = self.numeric_options[0]
            return (
                f"the model was {share / self.numeric_mass:.0%} committed to {top} - it was "
                f"not hedging, it was confident. Confidence is not grounding, so the figure "
                f"is checked against the event log rather than against its own probability"
            )
        if not self.numeric_options:
            return (
                f"token entropy {self.token_entropy:.2f} bits, semantic entropy "
                f"{self.semantic_entropy:.2f}: no candidate is a figure, so the model "
                f"is choosing how to word this, not what value to state"
            )
        figures = len(self.numeric_options)
        return (
            f"{figures} distinct figure{'' if figures == 1 else 's'} on the table but only "
            f"{self.numeric_mass:.0%} of the mass - the model is mostly writing prose here"
        )


def analyse(
    top_tokens: Sequence[str],
    top_probs: Sequence[float],
    min_numeric_mass: float = 0.55,
    max_top_share: float = 0.75,
) -> Divergence:
    """
    Cluster the decoder's top-k candidates by meaning and measure the split.

    :param min_numeric_mass: the model must be MORE LIKELY THAN NOT about to emit
        a figure before a numeric split counts. Below this the digits are a
        minority of the candidate set and the model is probably writing prose.
    :param max_top_share: within the numeric mass, the leading figure must hold
        no more than this share. A grounded number quoted from context comes back
        with the whole numeric mass on one value; a guess is spread across
        several.

        Both defaults are hand-set, not swept: the one live interception these
        were checked against ("12.5", top-5 `1:0.85 2:0.07 7:0.03 4:0.02 3:0.02`
        then `5:0.69 7:0.28`) is a single sample. Treat them as a starting
        point, not a calibrated threshold, and say so if asked.
    """
    pairs = [(t, float(p)) for t, p in zip(top_tokens, top_probs) if p > 0]
    if not pairs:
        return Divergence(token_entropy=0.0, semantic_entropy=0.0, clusters=[])

    # abs(): a single certain candidate gives -(1.0 * log2(1.0)) == -0.0, which
    # formats as "-0.00" on screen. Entropy is never negative, so the sign is
    # pure float artefact.
    token_h = abs(-sum(p * math.log2(p) for _, p in pairs))

    masses: Dict[str, float] = {}
    for tok, p in pairs:
        masses[cluster_key(tok)] = masses.get(cluster_key(tok), 0.0) + p
    total = sum(masses.values()) or 1.0
    sem_h = abs(-sum((m / total) * math.log2(m / total) for m in masses.values() if m > 0))

    numeric = sorted(
        ((k[1:], m / total) for k, m in masses.items() if k.startswith("#")),
        key=lambda kv: kv[1],
        reverse=True,
    )
    num_mass = sum(m for _, m in numeric)
    divergent = (
        len(numeric) >= 2
        and num_mass >= min_numeric_mass
        and numeric[0][1] / num_mass <= max_top_share
    )

    return Divergence(
        token_entropy=token_h,
        semantic_entropy=sem_h,
        clusters=sorted(((k, m / total) for k, m in masses.items()), key=lambda kv: kv[1], reverse=True),
        numeric_options=numeric,
        numeric_mass=num_mass,
        divergent=divergent,
    )


def fan(top_tokens: Sequence[str], top_probs: Sequence[float], k: int = 5) -> List[Dict[str, object]]:
    """The candidate set in the shape the terminal renders it: what the model saw."""
    return [
        {"token": t, "p": round(float(p), 4), "cluster": cluster_key(t)}
        for t, p in list(zip(top_tokens, top_probs))[:k]
    ]


def demo() -> None:
    """Self-check: wording variation and fact variation must NOT score alike."""
    wording = analyse(["is", " was", " has", " remains", "'s"], [0.31, 0.28, 0.22, 0.11, 0.08])
    figures = analyse(["12", " 15", " 9", " 10", " 4"], [0.31, 0.28, 0.22, 0.11, 0.08])

    # Identical distributions, so the old measure cannot separate them at all.
    assert abs(wording.token_entropy - figures.token_entropy) < 1e-9

    assert wording.semantic_entropy == 0.0, wording.clusters   # no figure at stake
    assert figures.semantic_entropy > 2.0, figures.clusters    # five rival values
    assert not wording.divergent
    assert figures.divergent

    # Regression: the stopword-lookup version scored this at 1.4 bits, because
    # "has"/"remains" are content words. Nothing non-numeric may split a cluster.
    assert cluster_key(" has") == cluster_key(" remains") == PROSE

    # A figure quoted from context: all the numeric mass sits on one value.
    quoted = analyse(["10", " 1", " the", " ten", " approximately"], [0.94, 0.02, 0.02, 0.01, 0.01])
    assert not quoted.divergent, quoted.numeric_options

    # Numbers present but the model is mostly writing prose - not a dice roll.
    prose = analyse([" days", " 12", " 15", " and", " for"], [0.72, 0.12, 0.09, 0.04, 0.03])
    assert not prose.divergent, prose.numeric_mass

    # Same figure spelled two ways is one claim, not two.
    same = analyse(["10", " 10", " 10.", " ten", " x"], [0.4, 0.3, 0.2, 0.05, 0.05])
    assert [v for v, _ in same.numeric_options] == ["10"], same.numeric_options

    assert cluster_key(" $128,500") == "#128500"
    assert cluster_key(" the") == PROSE and cluster_key(",") == PROSE
    print("semantic_divergence self-check OK")
    print("  wording:", wording.explain())
    print("  figures:", figures.explain())


if __name__ == "__main__":
    demo()
