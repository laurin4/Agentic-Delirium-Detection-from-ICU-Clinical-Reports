import csv
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

from src.agents.classification import classify_delirium
from src.agents.extraction import extract_passages
from src.agents.interpretation import interpret_signals
from src.agents.interpretation_llm import interpret_signals_llm
from src.models.model_config import LLM_MODEL_LABEL, LLM_PROVIDER
from src.pipeline.paths import ANONYMIZED_DIR, BERICHTE_INPUT_PATH, PREDICTIONS_DIR, MAX_REPORTS
from src.preprocessing.berichte_mapper import build_patient_level_berichte_report_records
from src.preprocessing.diagnosis_mapper import build_patient_level_report_records
from src.preprocessing.delirium_hint_keywords import haystack_contains_delirium_hint
from src.preprocessing.report_text_llm_reduction import reduce_report_text_for_llm


SIGNAL_KEYS = [
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
]

INTERPRETATION_MODE = "prompt"  # "rule" oder "prompt"
# PRIMARY: anonymized hospital reports CSV (Berichte.csv). Fallback: Diagnosenliste.
INPUT_MODE = "berichte"  # "berichte" | "diagnosis" | "txt"

LOGGER = logging.getLogger(__name__)

PREFILTER_SKIP_BE = "LLM übersprungen: keine Delir-Hinweisbegriffe im Bericht."
PREFILTER_SKIP_KONTEXT = "Kein regelbasierter Delir-Hinweis im Bericht gefunden."

_KLASSE_NULL_BE = "Keine ausreichenden Hinweise für ein dokumentiertes Delir."


def _report_contains_delirium_hint(full_text: str, reduced_text: str) -> bool:
    """Hints in either original or reduced text trigger the LLM (avoids truncation false negatives)."""
    return haystack_contains_delirium_hint(full_text) or haystack_contains_delirium_hint(reduced_text)


def _prediction_row_prefilter_skip(
    patient_id: str,
    report_name: str,
    reduction,
) -> Dict[str, object]:
    return {
        "PatientenID": patient_id,
        "bericht": report_name,
        "original_report_text_length": reduction.original_report_text_length,
        "llm_report_text_length": reduction.llm_report_text_length,
        "llm_text_reduction_method": reduction.llm_text_reduction_method,
        "delir_keyword_hits_count": reduction.delir_keyword_hits_count,
        "llm_skipped_by_prefilter": True,
        "anzahl_treffer": 0,
        "delir_signale": "",
        "signalstaerke": "niedrig",
        "kontext": PREFILTER_SKIP_KONTEXT,
        "alternative_erklaerung": False,
        "alternative_erklaerung_keywords": "",
        "begruendung": PREFILTER_SKIP_BE,
        "klasse": 0,
        "klassifikation": "kein_delir",
        "klassifikation_begruendung": _KLASSE_NULL_BE + " | " + PREFILTER_SKIP_BE,
    }


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


def _get_report_records():
    if INPUT_MODE == "berichte":
        if not BERICHTE_INPUT_PATH.exists():
            raise FileNotFoundError(
                f"Primary report input missing: {BERICHTE_INPUT_PATH}. "
                "Expected external CSV (semicolon-separated) with anonymized texts. "
                "Place Berichte.csv on the server under data/raw/, or set INPUT_MODE='diagnosis' "
                "to fall back to Diagnosenliste.csv."
            )
        report_records = build_patient_level_berichte_report_records()
    elif INPUT_MODE == "diagnosis":
        report_records = build_patient_level_report_records()
    elif INPUT_MODE == "txt":
        report_records = _load_txt_reports()
    else:
        raise ValueError(f"Ungültiger INPUT_MODE: {INPUT_MODE}")

    if MAX_REPORTS is not None:
        if isinstance(MAX_REPORTS, int) and MAX_REPORTS > 0:
            report_records = report_records[:MAX_REPORTS]
            print(
                f"Hinweis: MAX_REPORTS aktiv ({MAX_REPORTS}) - "
                f"es werden nur die ersten Berichte verarbeitet."
            )
        else:
            raise ValueError("MAX_REPORTS muss None oder eine positive Ganzzahl sein.")

    return report_records


def _get_output_path() -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_DIR / f"agent1_agent2_agent3_results_{INTERPRETATION_MODE}.csv"


def _sanitize_provider_model_slug(provider: str, model_label: str) -> str:
    """Filename-safe slug for `<provider>_<model>` (no dots/colons)."""
    raw = f"{provider}_{model_label}"
    s = re.sub(r"[^0-9A-Za-z_-]+", "_", raw.strip())
    return (s[:200] or "model").strip("_") or "model"


def _get_model_named_output_path() -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _sanitize_provider_model_slug(LLM_PROVIDER, LLM_MODEL_LABEL)
    return PREDICTIONS_DIR / f"agent_results_{slug}.csv"


def _assert_binary_klassen(rows: list) -> None:
    for row in rows:
        k = row.get("klasse")
        try:
            ki = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid klasse value (expected 0/1): {k!r}") from exc
        if ki not in (0, 1):
            raise ValueError(f"Non-binary klasse (expected 0 or 1): {ki}")


def _run_single_report(report: dict) -> Tuple[dict, bool]:
    """Returns (row_dict, skipped_by_prefilter bool)."""
    full_report_text = str(report.get("report_text", "") or "")
    reduction = reduce_report_text_for_llm(full_report_text)
    text = reduction.reduced_text
    patient_id = str(report.get("PatientenID", "") or "").strip()
    default_bericht = (
        f"berichte_{patient_id}.txt" if INPUT_MODE == "berichte" else f"diagnosis_{patient_id}.txt"
    )
    report_name = str(report.get("bericht", default_bericht) or "").strip()

    print(
        f"[LLM report text] original_len={reduction.original_report_text_length} "
        f"reduced_len={reduction.llm_report_text_length} "
        f"method={reduction.llm_text_reduction_method}"
    )
    LOGGER.info(
        "LLM report text reduction patient=%s original_len=%d reduced_len=%d method=%s",
        patient_id,
        reduction.original_report_text_length,
        reduction.llm_report_text_length,
        reduction.llm_text_reduction_method,
    )

    if not _report_contains_delirium_hint(full_report_text, text):
        print(
            f"[Delirium prefilter] Patient={patient_id} — keine Hinweisbegriffe, "
            f"Agent 1 + LLM Interpretation werden übersprungen."
        )
        LOGGER.info("Delirium prefilter skipped LLM for patient=%s", patient_id)
        return (_prediction_row_prefilter_skip(patient_id, report_name, reduction), True)

    print("\n=== DEBUG REPORT ===")
    print("Patient:", patient_id)
    print("Full report length:", len(full_report_text))
    print("LLM input length:", len(text))
    print("Text preview:", text[:500])
    print("====================\n")

    result = extract_passages(text, patient_id=patient_id, report_name=report_name)

    if INTERPRETATION_MODE == "rule":
        interpretation = interpret_signals(text, result)
    elif INTERPRETATION_MODE == "prompt":
        interpretation = interpret_signals_llm(text, result, patient_id=patient_id, report_name=report_name)
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

    if interpretation.get("alternative_erklaerung_keywords"):
        print(
            "    alternative_erklaerung_keywords: "
            + ", ".join(interpretation["alternative_erklaerung_keywords"])
        )

    if interpretation.get("begruendung"):
        print("    begruendung:")
        for reason in interpretation["begruendung"]:
            print(f"      - {reason}")

    print("  [classification]")
    print(f"    klasse: {classification['klasse']}")
    print(f"    klassifikation: {classification['klassifikation']}")

    if classification.get("begruendung"):
        print("    begruendung:")
        for reason in classification["begruendung"]:
            print(f"      - {reason}")

    print()

    return ({
        "PatientenID": patient_id,
        "bericht": report_name,
        "original_report_text_length": reduction.original_report_text_length,
        "llm_report_text_length": reduction.llm_report_text_length,
        "llm_text_reduction_method": reduction.llm_text_reduction_method,
        "delir_keyword_hits_count": reduction.delir_keyword_hits_count,
        "llm_skipped_by_prefilter": False,
        "anzahl_treffer": len(hits),
        "delir_signale": " | ".join(hits),
        "signalstaerke": interpretation["signalstaerke"],
        "kontext": interpretation["kontext"],
        "alternative_erklaerung": interpretation["alternative_erklaerung"],
        "alternative_erklaerung_keywords": " | ".join(
            interpretation.get("alternative_erklaerung_keywords", [])
        ),
        "begruendung": " | ".join(interpretation.get("begruendung", [])),
        "klasse": classification["klasse"],
        "klassifikation": classification["klassifikation"],
        "klassifikation_begruendung": " | ".join(classification.get("begruendung", [])),
    }, False)


def main():
    output_csv = _get_output_path()
    report_records = _get_report_records()

    print(f"\n=== Agent 1 + Agent 2 + Agent 3: Delir-Pipeline ({INTERPRETATION_MODE}) ===")
    print(f"Anzahl Berichte: {len(report_records)}\n")

    rows = []
    n_prefilter_skip = 0
    n_llm = 0
    for report in report_records:
        row_dict, skipped = _run_single_report(report)
        rows.append(row_dict)
        if skipped:
            n_prefilter_skip += 1
        else:
            n_llm += 1

    LOGGER.info(
        "Delirium prefilter summary: skipped_llm_reports=%d sent_to_llm=%d total=%d",
        n_prefilter_skip,
        n_llm,
        len(report_records),
    )
    print(
        "\n=== Delirium prefilter summary ===\n"
        f"Übersprungen (keine Hinweisbegriffe): {n_prefilter_skip}\n"
        f"An LLM geschickt: {n_llm}\n"
        f"Berichte gesamt: {len(report_records)}\n"
    )

    _assert_binary_klassen(rows)

    fieldnames = [
        "PatientenID",
        "bericht",
        "original_report_text_length",
        "llm_report_text_length",
        "llm_text_reduction_method",
        "delir_keyword_hits_count",
        "llm_skipped_by_prefilter",
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
    ]

   

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    model_copy_path = _get_model_named_output_path()
    shutil.copy2(output_csv, model_copy_path)

    print(f"Ergebnisse gespeichert in: {output_csv}")
    print(f"Ergebnisse (Modellkopie) gespeichert in: {model_copy_path}")


if __name__ == "__main__":
    main()