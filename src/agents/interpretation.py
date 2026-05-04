from typing import Dict, List, Any

SIGNAL_KEYS = [
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
]

ALTERNATIVE_EXPLANATION_KEYWORDS = [
    "hyperkapn",
    "co2",
    "sedier",
    "sedation",
    "intub",
    "intrazerebrale blutung",
    "hirnblutung",
    "neuro",
    "delir bei neurologischer grunderkrankung",
    "sopor",
    "somnol",
    "bewusstseinsstörung",
    "hydrozephalus",
]


def _safe_list(signals: Dict[str, Any], key: str) -> List[str]:
    value = signals.get(key, [])
    return value if isinstance(value, list) else []


def _has_alternative_explanation(report_text: str) -> List[str]:
    text_lower = report_text.lower()
    found = []
    for keyword in ALTERNATIVE_EXPLANATION_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return found


def interpret_signals(report_text: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 2: interpretiert die von Agent 1 extrahierten Signale.

    Dieser Agent klassifiziert noch NICHT final in 0/1,
    sondern bewertet die Signalstärke und den klinischen Kontext.
    """
    normalized = {key: _safe_list(signals, key) for key in SIGNAL_KEYS}

    explicit_delir = normalized["delir_explizit"]
    disorientation = normalized["desorientierung"]
    agitation = normalized["hyperaktivitaet_agitation"]
    vigilance = normalized["vigilanz"]
    therapy = normalized["delir_therapie"]
    prophylaxis = normalized["delir_prophylaxe"]

    alternative_explanations = _has_alternative_explanation(report_text)
    has_alternative_explanation = len(alternative_explanations) > 0

    reasoning: List[str] = []

    has_indirect_signals = bool(disorientation or vigilance or agitation or (therapy and not prophylaxis))
    has_only_prophylaxis = bool(prophylaxis) and not (explicit_delir or has_indirect_signals)

    if explicit_delir:
        signal_strength = "hoch"
        context = "explizite Delirdiagnose dokumentiert"
        reasoning.append("explizite Delirdiagnose vorhanden")
        if agitation:
            reasoning.append("zusätzliche Hinweise auf hyperaktives/agitiertes Verhalten vorhanden")
        if therapy:
            reasoning.append("Delir-spezifische Therapie dokumentiert")
    elif has_indirect_signals:
        signal_strength = "mittel"
        context = "indirekte Delir-Signale vorhanden"
        if disorientation:
            reasoning.append("Desorientierungs-Signale vorhanden")
        if vigilance:
            reasoning.append("Vigilanzveränderung vorhanden")
        if agitation:
            reasoning.append("Agitation/Hyperaktivität vorhanden")
        if therapy:
            reasoning.append("mögliche Delir-Therapie dokumentiert")
    elif has_only_prophylaxis:
        signal_strength = "niedrig"
        context = "nur Delirprophylaxe dokumentiert"
        reasoning.append("nur Delirprophylaxe vorhanden, kein Beweis für Delir")
    else:
        signal_strength = "niedrig"
        context = "keine relevanten Delir-Signale dokumentiert"
        reasoning.append("keine relevanten Delir-Signale vorhanden")

    if has_alternative_explanation:
        reasoning.append(
            "mögliche alternative Erklärung vorhanden: " + ", ".join(alternative_explanations)
        )

    return {
        "signalstaerke": signal_strength,
        "kontext": context,
        "alternative_erklaerung": has_alternative_explanation,
        "alternative_erklaerung_keywords": alternative_explanations,
        "begruendung": reasoning,
        "signale": normalized,
    }