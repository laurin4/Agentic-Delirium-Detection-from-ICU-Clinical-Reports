"""
Deterministic post-LLM clinical decision guardrails.

Hard-excludes only clear non-delirium cases (no evidence, prophylaxis-only,
negation-only). Uncertain LLM positives (indirect symptoms, alternative
explanations) are preserved as klasse=1 and flagged for manual review.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SIGNAL_KEYS = (
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
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
    return bool(
        _safe_list(signals, "desorientierung")
        or _safe_list(signals, "hyperaktivitaet_agitation")
        or _safe_list(signals, "vigilanz")
    )


def _llm_suggests_delirium(signal: str) -> bool:
    """mittel/hoch = LLM supports possible/documented delirium (subject to hard excludes)."""
    return signal in ("mittel", "hoch")


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

    Only forces klasse=0 for no evidence, prophylaxis-only, or negation-only.
    Other LLM positives are kept and flagged for manual review when uncertain.
    """
    signals = {k: _safe_list(extraction_signals, k) for k in SIGNAL_KEYS}

    has_direct = _bool_meta(evidence_metadata, "has_direct_delir_evidence") or _has_explicit_delir_signals(
        signals
    )
    has_indirect = _has_indirect_signals(signals, evidence_metadata)
    has_negated = _bool_meta(evidence_metadata, "has_negated_delir_evidence")
    prophy_only = _bool_meta(evidence_metadata, "has_prophylaxis_or_risk_only") and not has_direct and not has_indirect

    alt = bool(interpretation.get("alternative_erklaerung", False))
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
            alt=alt,
        )

    # --- Hard exclude: prophylaxis / screening / risk only ---
    if prophy_only and not has_direct:
        return _finalize(
            signalstaerke="niedrig",
            klasse=0,
            kontext="Nur Delirprophylaxe/Screening/Risiko ohne dokumentiertes Delir.",
            begruendung=begruendung + ["Nur Prophylaxe/Risiko — kein Delirnachweis."],
            manual_review=False,
            rule="prophylaxis_only_not_positive",
            alt=alt,
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
            alt=alt,
        )

    therapy_with_context = _has_delir_therapy(signals) and (
        has_direct or has_indirect or _has_explicit_delir_signals(signals)
    )

    # --- Strong positive: direct delir (not negated without explicit counter-evidence) ---
    if has_direct and not (has_negated and not _has_explicit_delir_signals(signals)):
        new_signal = signal if signal in ("hoch", "mittel") else "hoch"
        return _finalize(
            signalstaerke=new_signal,
            klasse=1,
            kontext=kontext or "Explizite Delirdokumentation in den Evidenz-Snippets.",
            begruendung=begruendung + ["Expliziter Delirnachweis (Guardrail)."],
            manual_review=False,
            rule="direct_delir_positive",
            alt=alt,
        )

    if therapy_with_context:
        new_signal = signal if _llm_suggests_delirium(signal) else "mittel"
        return _finalize(
            signalstaerke=new_signal,
            klasse=1,
            kontext=kontext or "Delirtherapie mit kompatiblem klinischem Kontext.",
            begruendung=begruendung + ["Delirtherapie + Symptomkontext (Guardrail)."],
            manual_review=False,
            rule="direct_delir_positive",
            alt=alt,
        )

    # --- Preserve LLM positives; flag uncertain cases for manual review ---
    if _llm_suggests_delirium(signal):
        manual_review = True
        rule = "llm_classification"

        if alt:
            rule = "positive_with_alternative_explanation_review_needed"
            begruendung = begruendung + [
                "Positiv trotz alternativer Erklärung — manuelle Prüfung empfohlen."
            ]
        elif has_indirect and not has_direct:
            rule = "indirect_symptoms_positive_review_needed"
            begruendung = begruendung + [
                "Indirekte Symptome mit LLM-positiver Bewertung — manuelle Prüfung empfohlen."
            ]

        if signal == "mittel":
            manual_review = True
            if rule == "llm_classification":
                begruendung = begruendung + [
                    "Signalstärke mittel — manuelle Prüfung empfohlen."
                ]

        return _finalize(
            signalstaerke=signal,
            klasse=1,
            kontext=kontext,
            begruendung=begruendung,
            manual_review=manual_review,
            rule=rule,
            alt=alt,
        )

    # --- Default: LLM niedrig or no positive support ---
    return _finalize(
        signalstaerke="niedrig",
        klasse=0,
        kontext=kontext or "Keine ausreichenden Hinweise für ein dokumentiertes Delir.",
        begruendung=begruendung,
        manual_review=False,
        rule="llm_classification",
        alt=alt,
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
