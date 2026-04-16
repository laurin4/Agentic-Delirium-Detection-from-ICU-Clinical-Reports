import csv
from src.agents.extraction import extract_passages
from src.agents.interpretation import interpret_signals
from src.agents.interpretation_llm import interpret_signals_llm
from src.agents.classification import classify_delirium
from src.pipeline.paths import ANONYMIZED_DIR, PREDICTIONS_DIR
from src.preprocessing.diagnosis_mapper import build_patient_level_report_records

SIGNAL_KEYS = [
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
]

INTERPRETATION_MODE = "prompt"  # "rule" oder "prompt"
INPUT_MODE = "diagnosis"  # "diagnosis" oder "txt"


def load_report(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
def _load_txt_reports():
    reports_dir = ANONYMIZED_DIR / "generische Arztberichte"
    txt_files = sorted(reports_dir.glob("*.txt"))
    rows = []
    for report_path in txt_files:
        rows.append(
            {
                "PatientenID": report_path.stem,
                "bericht": report_path.name,
                "report_text": load_report(str(report_path)),
            }
        )
    return rows


if __name__ == "__main__":
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = PREDICTIONS_DIR / f"agent1_agent2_agent3_results_{INTERPRETATION_MODE}.csv"
    if INPUT_MODE == "diagnosis":
        report_records = build_patient_level_report_records()
    elif INPUT_MODE == "txt":
        report_records = _load_txt_reports()
    else:
        raise ValueError(f"Ungültiger INPUT_MODE: {INPUT_MODE}")
    rows = []

    print(f"\n=== Agent 1 + Agent 2 + Agent 3: Delir-Pipeline ({INTERPRETATION_MODE}) ===")
    print(f"Anzahl Berichte: {len(report_records)}\n")

    for report in report_records:
        text = report.get("report_text", "")
        patient_id = str(report.get("PatientenID", "")).strip()
        report_name = str(report.get("bericht", f"diagnosis_{patient_id}.txt"))
        result = extract_passages(text)
        if INTERPRETATION_MODE == "rule":
            interpretation = interpret_signals(text, result)
        elif INTERPRETATION_MODE == "prompt":
            interpretation = interpret_signals_llm(text, result)
        else:
            raise ValueError(f"Ungültiger INTERPRETATION_MODE: {INTERPRETATION_MODE}")
        classification = classify_delirium(interpretation)
        hits = []
        for key in SIGNAL_KEYS:
            values = result.get(key, [])
            if isinstance(values, list):
                hits.extend(values)

        print(f"[{report_name}] PatientenID={patient_id} | Treffer gesamt: {len(hits)}")
        if hits:
            for key in SIGNAL_KEYS:
                values = result.get(key, [])
                if isinstance(values, list) and values:
                    print(f"  [{key}]")
                    for idx, hit in enumerate(values, start=1):
                        print(f"    {idx}. {hit}")
        else:
            print("  - Keine Delir-Signale gefunden")

        print("  [interpretation]")
        print(f"    signalstaerke: {interpretation['signalstaerke']}")
        print(f"    kontext: {interpretation['kontext']}")
        print(f"    alternative_erklaerung: {interpretation['alternative_erklaerung']}")
        if interpretation["alternative_erklaerung_keywords"]:
            print(f"    alternative_erklaerung_keywords: {', '.join(interpretation['alternative_erklaerung_keywords'])}")
        if interpretation["begruendung"]:
            print("    begruendung:")
            for reason in interpretation["begruendung"]:
                print(f"      - {reason}")

        print("  [classification]")
        print(f"    klasse: {classification['klasse']}")
        print(f"    klassifikation: {classification['klassifikation']}")
        if classification["begruendung"]:
            print("    begruendung:")
            for reason in classification["begruendung"]:
                print(f"      - {reason}")

        print()

        rows.append({
            "PatientenID": patient_id,
            "bericht": report_name,
            "anzahl_treffer": len(hits),
            "delir_signale": " | ".join(hits),
            "signalstaerke": interpretation["signalstaerke"],
            "kontext": interpretation["kontext"],
            "alternative_erklaerung": interpretation["alternative_erklaerung"],
            "alternative_erklaerung_keywords": " | ".join(interpretation["alternative_erklaerung_keywords"]),
            "begruendung": " | ".join(interpretation["begruendung"]),
            "klasse": classification["klasse"],
            "klassifikation": classification["klassifikation"],
            "klassifikation_begruendung": " | ".join(classification["begruendung"]),
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "PatientenID",
                "bericht",
                "anzahl_treffer",
                "delir_signale",
                "signalstaerke",
                "kontext",
                "alternative_erklaerung",
                "alternative_erklaerung_keywords",
                "begruendung",
                "klasse",
                "klassifikation",
                "klassifikation_begruendung",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ergebnisse gespeichert in: {output_csv}")