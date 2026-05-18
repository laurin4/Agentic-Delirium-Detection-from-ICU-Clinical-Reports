"""
Exploratory delirium plausibility score (0–100). Not used for final klasse.
"""

from __future__ import annotations

from typing import Any, Dict


def delirium_probability_estimate(
    signalstaerke: str,
    klasse: int,
    *,
    manual_review_candidate: bool = False,
    decision_rule_applied: str = "",
    has_direct_delir_evidence: bool = False,
) -> int:
    """
    Map signal strength + guardrail context to a deterministic 0–100 score.
    """
    signal = str(signalstaerke or "niedrig").strip().lower()
    if signal not in ("niedrig", "mittel", "hoch"):
        signal = "niedrig"

    if int(klasse) == 0:
        mapping = {"niedrig": 5, "mittel": 22, "hoch": 35}
    else:
        mapping = {"niedrig": 25, "mittel": 58, "hoch": 90}

    score = mapping[signal]

    rule = str(decision_rule_applied or "")
    if rule in ("prophylaxis_only_not_positive", "negated_delir_not_positive", "no_evidence_prefilter_skip"):
        score = min(score, 12)
    elif rule == "direct_delir_positive" or has_direct_delir_evidence:
        score = max(score, 78)
    if manual_review_candidate:
        score = min(max(score, 45), 72)

    return int(max(0, min(100, score)))
