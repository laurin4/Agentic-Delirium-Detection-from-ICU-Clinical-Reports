"""
Deterministic post-LLM clinical decision guardrails.

Hard-excludes no evidence, prophylaxis-only, negation-only, and isolated weak
indirect symptoms. Supports delirium-compatible symptom clusters; downgrades
isolated indirect positives with dominant alternative explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

SIGNAL_KEYS = (
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
)

INDIRECT_DIMENSION_KEYS: Tuple[str, ...] = (
    "desorientierung",
    "hyperaktivitaet_agitation",
    "vigilanz",
)


def _safe_list(signals: Dict[str, Any], key: str) -> List[str]:
    value = signals.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _bool_meta(evidence_metadata: Dict[str, Any], key: str) -> bool:
    raw = evidence_metadata.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes")
    return bool(raw)


def _has_explicit_delir_signals(signals: Dict[str, Any]) -> bool:
    return bool(_safe_list(signals, "delir_explizit"))


def _has_delir_therapy(signals: Dict[str, Any]) -> bool:
    return bool(_safe_list(signals, "delir_therapie"))


def _has_indirect_signals(signals: Dict[str, Any], evidence_metadata: Dict[str, Any]) -> bool:
    if _bool_meta(evidence_metadata, "has_indirect_delir_evidence"):
        return True
    return bool(any(_safe_list(signals, k) for k in INDIRECT_DIMENSION_KEYS))


def _indirect_dimension_count(signals: Dict[str, Any]) -> int:
    return sum(1 for k in INDIRECT_DIMENSION_KEYS if _safe_list(signals, k))


def _has_symptom_cluster(signals: Dict[str, Any], evidence_metadata: Dict[str, Any]) -> bool:
    """Two+ indirect dimensions, or delir therapy with compatible symptoms."""
    if _has_delir_therapy(signals) and (
        _has_indirect_signals(signals, evidence_metadata) or _has_explicit_delir_signals(signals)
    ):
        return True
    return _indirect_dimension_count(signals) >= 2


def _is_isolated_indirect_only(signals: Dict[str, Any], evidence_metadata: Dict[str, Any]) -> bool:
    """Exactly one indirect symptom dimension and no direct delir evidence."""
    if not _has_indirect_signals(signals, evidence_metadata):
        return False
    if _has_explicit_delir_signals(signals) or _bool_meta(evidence_metadata, "has_direct_delir_evidence"):
        return False
    return _indirect_dimension_count(signals) == 1


def _has_alternative_explanation(
    interpretation: Dict[str, Any],
    evidence_metadata: Dict[str, Any],
) -> bool:
    if bool(interpretation.get("alternative_erklaerung", False)):
        return True
    return _bool_meta(evidence_metadata, "has_alternative_explanation")


def _llm_suggests_delirium(signal: str) -> bool:
    """mittel/hoch = LLM supports possible/documented delirium (subject to hard excludes)."""
    return signal in ("mittel", "hoch")


def _cap_signal_for_alt_downgrade(signal: str) -> str:
    """After alternative-explanation downgrade, keep signal at niedrig or mittel."""
    if signal == "hoch":
        return "mittel"
    if signal in ("niedrig", "mittel"):
        return signal
    return "niedrig"


def apply_clinical_decision_guardrails(
    interpretation: Dict[str, Any],
    extraction_signals: Dict[str, Any],
    evidence_metadata: Dict[str, Any],
    *,
    llm_skipped: bool = False,
    prefilter_klasse: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Apply transparent rules after Agent 2.

    Symptom clusters without dominant alternatives may remain klasse=1 (flagged).
    Isolated indirect symptoms are not auto-positive; clear alternative without
    cluster is downgraded to klasse=0 with manual review.
    """
    signals = {k: _safe_list(extraction_signals, k) for k in SIGNAL_KEYS}

    has_direct = _bool_meta(evidence_metadata, "has_direct_delir_evidence") or _has_explicit_delir_signals(
        signals
    )
    has_indirect = _has_indirect_signals(signals, evidence_metadata)
    has_cluster = _has_symptom_cluster(signals, evidence_metadata)
    isolated_indirect = _is_isolated_indirect_only(signals, evidence_metadata)
    has_negated = _bool_meta(evidence_metadata, "has_negated_delir_evidence")
    prophy_only = _bool_meta(evidence_metadata, "has_prophylaxis_or_risk_only") and not has_direct and not has_indirect

    has_alt = _has_alternative_explanation(interpretation, evidence_metadata)
    signal = str(interpretation.get("signalstaerke", "niedrig") or "niedrig").strip().lower()
    if signal not in ("niedrig", "mittel", "hoch"):
        signal = "niedrig"

    begruendung: List[str] = list(interpretation.get("begruendung", []) or [])
    kontext = str(interpretation.get("kontext", "") or "")

    # --- Hard exclude: no evidence ---
    if llm_skipped or evidence_metadata.get("llm_text_reduction_method") == "no_evidence_prefilter_skip":
        return _finalize(
            signalstaerke="niedrig",
            klasse=0,
            kontext=kontext or "Keine regelbasierten Delir-Hinweise.",
            begruendung=begruendung,
            manual_review=False,
            rule="no_evidence_prefilter_skip",
            alt=has_alt,
        )

    # --- Hard exclude: prophylaxis / screening / risk / conditional only ---
    if prophy_only and not has_direct:
        return _finalize(
            signalstaerke="niedrig",
            klasse=0,
            kontext="Nur Delirprophylaxe/Screening/Risiko/Bei-Delir ohne dokumentiertes Delir.",
            begruendung=begruendung + ["Nur Prophylaxe/Risiko/Bei-Delir — kein Delirnachweis."],
            manual_review=False,
            rule="prophylaxis_only_not_positive",
            alt=has_alt,
        )

    # --- Hard exclude: negation only (no separate explicit positive) ---
    if has_negated and not has_direct and not _has_explicit_delir_signals(signals):
        return _finalize(
            signalstaerke="niedrig",
            klasse=0,
            kontext=kontext or "Delir ausgeschlossen bzw. negiert.",
            begruendung=begruendung + ["Negierter Delirhinweis — nicht als Delir gewertet."],
            manual_review=False,
            rule="negated_delir_not_positive",
            alt=has_alt,
        )

    therapy_with_context = _has_delir_therapy(signals) and (
        has_direct or has_indirect or _has_explicit_delir_signals(signals)
    )

    # --- Strong positive: direct delir (kept unless clearly negated without explicit term) ---
    if has_direct and not (has_negated and not _has_explicit_delir_signals(signals)):
        new_signal = signal if signal in ("hoch", "mittel") else "hoch"
        return _finalize(
            signalstaerke=new_signal,
            klasse=1,
            kontext=kontext or "Explizite Delirdokumentation in den Evidenz-Snippets.",
            begruendung=begruendung + ["Expliziter Delirnachweis (Guardrail)."],
            manual_review=False,
            rule="direct_delir_positive",
            alt=has_alt,
        )

    if therapy_with_context:
        new_signal = signal if _llm_suggests_delirium(signal) else "mittel"
        return _finalize(
            signalstaerke=new_signal,
            klasse=1,
            kontext=kontext or "Delirtherapie mit kompatiblem klinischem Kontext.",
            begruendung=begruendung + ["Delirtherapie + Symptomkontext (Guardrail)."],
            manual_review=has_alt,
            rule="delir_therapy_with_compatible_symptoms",
            alt=has_alt,
        )

    # --- Isolated single indirect dimension: do not auto-call positive ---
    if isolated_indirect and _llm_suggests_delirium(signal):
        down_signal = _cap_signal_for_alt_downgrade(signal)
        return _finalize(
            signalstaerke=down_signal,
            klasse=0,
            kontext=kontext or "Isoliertes indirektes Symptom ohne Delir-Cluster.",
            begruendung=begruendung
            + ["Isoliertes schwaches indirektes Symptom — nicht als Delir gewertet; manuelle Prüfung."],
            manual_review=True,
            rule="isolated_indirect_not_positive",
            alt=has_alt,
        )

    # --- Alternative explanation without coherent cluster: downgrade LLM positive ---
    if has_indirect and not has_direct and has_alt and not has_cluster and _llm_suggests_delirium(signal):
        down_signal = _cap_signal_for_alt_downgrade(signal)
        return _finalize(
            signalstaerke=down_signal,
            klasse=0,
            kontext=kontext
            or "Indirekte Symptome mit dominanter alternativer Erklärung ohne Delir-Cluster.",
            begruendung=begruendung
            + [
                "Alternative Erklärung ohne Delir-Cluster — nicht als Delir gewertet; "
                "manuelle Prüfung empfohlen."
            ],
            manual_review=True,
            rule="alternative_explanation_downgrade",
            alt=has_alt,
        )

    # --- Symptom cluster with alternative (plausible but not definitive): keep positive, review ---
    if has_cluster and has_alt and _llm_suggests_delirium(signal):
        return _finalize(
            signalstaerke=signal,
            klasse=1,
            kontext=kontext or "Delir-kompatibles Symptomcluster trotz alternativer Erklärung.",
            begruendung=begruendung
            + [
                "Symptomcluster mit alternativer Erklärung — Delir möglich; "
                "manuelle Prüfung empfohlen."
            ],
            manual_review=True,
            rule="symptom_cluster_with_alternative_review_needed",
            alt=has_alt,
        )

    # --- Symptom cluster without dominant alternative: positive with review ---
    if has_cluster and not has_direct and _llm_suggests_delirium(signal):
        return _finalize(
            signalstaerke=signal,
            klasse=1,
            kontext=kontext or "Delir-kompatibles Symptomcluster in den Evidenz-Snippets.",
            begruendung=begruendung
            + ["Symptomcluster mit LLM-positiver Bewertung — manuelle Prüfung empfohlen."],
            manual_review=True,
            rule="symptom_cluster_positive_review_needed",
            alt=has_alt,
        )

    # --- Residual indirect LLM positive (multi-dimension but not caught above) ---
    if has_indirect and not has_direct and _llm_suggests_delirium(signal):
        return _finalize(
            signalstaerke=signal,
            klasse=1,
            kontext=kontext,
            begruendung=begruendung
            + ["Indirekte Symptome mit LLM-positiver Bewertung — manuelle Prüfung empfohlen."],
            manual_review=True,
            rule="indirect_symptoms_positive_review_needed",
            alt=has_alt,
        )

    # --- Other LLM positives (mittel/hoch) ---
    if _llm_suggests_delirium(signal):
        manual_review = signal == "mittel"
        rule = "llm_classification"
        extra_begr: List[str] = []
        if signal == "mittel":
            extra_begr = ["Signalstärke mittel — manuelle Prüfung empfohlen."]
        return _finalize(
            signalstaerke=signal,
            klasse=1,
            kontext=kontext,
            begruendung=begruendung + extra_begr,
            manual_review=manual_review,
            rule=rule,
            alt=has_alt,
        )

    # --- Default: LLM niedrig or no positive support ---
    return _finalize(
        signalstaerke="niedrig",
        klasse=0,
        kontext=kontext or "Keine ausreichenden Hinweise für ein dokumentiertes Delir.",
        begruendung=begruendung,
        manual_review=isolated_indirect or has_alt,
        rule="llm_classification",
        alt=has_alt,
    )


def _finalize(
    *,
    signalstaerke: str,
    klasse: int,
    kontext: str,
    begruendung: List[str],
    manual_review: bool,
    rule: str,
    alt: bool,
) -> Dict[str, Any]:
    klasse = int(klasse)
    if klasse not in (0, 1):
        klasse = 1 if signalstaerke in ("mittel", "hoch") else 0
    klassifikation = "delir" if klasse == 1 else "kein_delir"
    return {
        "signalstaerke": signalstaerke,
        "klasse": klasse,
        "klassifikation": klassifikation,
        "kontext": kontext,
        "begruendung": begruendung,
        "alternative_erklaerung": alt,
        "manual_review_candidate": manual_review,
        "decision_rule_applied": rule,
        "has_alternative_explanation": alt,
    }
