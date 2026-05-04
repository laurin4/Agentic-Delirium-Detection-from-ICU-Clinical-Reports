from typing import Dict, Any, List


def classify_delirium(interpretation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 3: finale Klassifikation in 0 / 1.

    0 = kein Delir
    1 = Delir
    """
    signalstaerke = interpretation.get("signalstaerke", "niedrig")
    kontext = interpretation.get("kontext", "")
    alternative_erklaerung = bool(interpretation.get("alternative_erklaerung", False))
    begruendung: List[str] = list(interpretation.get("begruendung", []))

    if signalstaerke == "hoch":
        klasse = 1
        finale_begruendung = [
            "Explizite oder sehr starke Delir-Signale vorhanden.",
            *begruendung,
        ]
    elif signalstaerke == "mittel":
        klasse = 1
        finale_begruendung = [
            "Indirekte Delir-Signale vorhanden.",
            *begruendung,
        ]
    else:
        klasse = 0
        finale_begruendung = [
            "Keine ausreichenden Hinweise für ein dokumentiertes Delir.",
            *begruendung,
        ]

    return {
        "klasse": klasse,
        "klassifikation": {
            0: "kein_delir",
            1: "delir",
        }[klasse],
        "kontext": kontext,
        "alternative_erklaerung": alternative_erklaerung,
        "begruendung": finale_begruendung,
    }
