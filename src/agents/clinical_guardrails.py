"""
Deterministic post-LLM clinical decision guardrails.

Hard-excludes clear non-delirium cases (no evidence, prophylaxis-only,
negation-only). Downgrades indirect-symptom positives when a strong
alternative explanation is present; other uncertain indirect positives stay
klasse=1 with manual review.
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

    Forces klasse=0 for no evidence, prophylaxis-only, negation-only, and
    indirect-only positives with alternative explanation. Other indirect LLM
    positives remain klasse=1 with manual review.
    """
    signals = {k: _safe_list(extraction_signals, k) for k in SIGNAL_KEYS}

    has_direct = _bool_meta(evidence_metadata, "has_direct_delir_evidence") or _has_explicit_delir_signals(
        signals
    )
    has_indirect = _has_indirect_signals(signals, evidence_metadata)
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

    # --- Hard exclude: prophylaxis / screening / risk only ---
    if prophy_only and not has_direct:
        return _finalize(
            signalstaerke="niedrig",
            klasse=0,
            kontext="Nur Delirprophylaxe/Screening/Risiko ohne dokumentiertes Delir.",
            begruendung=begruendung + ["Nur Prophylaxe/Risiko — kein Delirnachweis."],
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

    # --- Strong positive: direct delir (kept unless explicitly negated without explicit term) ---
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
            manual_review=False,
            rule="direct_delir_positive",
            alt=has_alt,
        )

    # --- Indirect only + alternative explanation: downgrade LLM positive to klasse=0 ---
    if (
        has_indirect
        and not has_direct
        and has_alt
        and _llm_suggests_delirium(signal)
    ):
        down_signal = _cap_signal_for_alt_downgrade(signal)
        return _finalize(
            signalstaerke=down_signal,
            klasse=0,
            kontext=kontext
            or "Indirekte Symptome mit plausibler alternativer Erklärung — nicht als Delir gewertet.",
            begruendung=begruendung
            + [
                "Indirekte Symptome mit alternativer Erklärung (z. B. psychiatrisch, "
                "Intoxikation, Sedierung) — manuelle Prüfung empfohlen."
            ],
            manual_review=True,
            rule="alternative_explanation_downgrade",
            alt=has_alt,
        )

    # --- Indirect only, LLM positive, no alternative explanation: keep positive, flag review ---
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
        manual_review=False,
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
