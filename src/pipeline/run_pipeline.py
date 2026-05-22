import csv
import logging
import os
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.classification import classify_delirium
from src.agents.clinical_guardrails import apply_clinical_decision_guardrails
from src.agents.extraction import extract_passages
from src.agents.interpretation import interpret_signals
from src.agents.interpretation_llm import interpret_signals_llm
from src.models.model_config import LLM_MODEL_LABEL, LLM_PROVIDER
from src.pipeline.paths import (
    ANONYMIZED_DIR,
    BERICHTE_INPUT_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    PREDICTIONS_DIR,
    MAX_REPORTS,
    SQLITE_PREDICTIONS_DB_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.validation_cohort_filter import (
    filter_report_records_for_validation_cohort,
    validation_cohort_only_enabled,
)
from src.agents.delirium_probability import delirium_probability_estimate
from src.preprocessing.berichte_filters import normalize_bertyp
from src.preprocessing.berichte_mapper import build_report_level_berichte_records
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL
from src.preprocessing.diagnosis_mapper import build_patient_level_report_records
from src.preprocessing.evidence_extraction import (
    METHOD_SHORT_REPORT_FULLTEXT,
    apply_short_report_fulltext_to_evidence,
    evidence_snippets_json_for_csv,
    extract_delirium_evidence,
    llm_should_receive_evidence,
    should_send_short_report_without_evidence,
)


SIGNAL_KEYS = [
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
]

INTERPRETATION_MODE = "prompt"  # "rule" oder "prompt"
INPUT_MODE = "berichte"  # "berichte" (production) | "diagnosis" (legacy) | "txt"

LOGGER = logging.getLogger(__name__)

DEBUG_VERBOSE = os.environ.get("DEBUG_LLM_OUTPUT", "").strip().lower() in ("1", "true", "yes")

NO_EVIDENCE_KONTEXT = "LLM übersprungen: keine regelbasierten Delir-Hinweise im Bericht gefunden."
NO_EVIDENCE_BE = "Kein Delir-Hinweis in regelbasierter Volltextsuche."

_KLASSE_NULL_BE = "Keine ausreichenden Hinweise für ein dokumentiertes Delir."

UNKNOWN_BERTYP = "unknown"


def resolve_bertyp(report: dict) -> str:
    """Normalized bertyp label for logging/CSV; empty/missing → ``unknown``."""
    label = normalize_bertyp(report.get("bertyp", ""))
    return label if label else UNKNOWN_BERTYP


def _normalize_report_record_bertyp(record: dict) -> dict:
    """Ensure each report dict carries a string ``bertyp`` field."""
    out = dict(record)
    out["bertyp"] = resolve_bertyp(out)
    return out


def _new_bertyp_stats() -> Dict[str, Dict[str, int]]:
    return {"total": 0, "skipped": 0, "sent_to_llm": 0, "positives": 0}


def accumulate_bertyp_stat(
    stats: Dict[str, Dict[str, int]],
    bertyp: str,
    *,
    skipped: bool,
    failed: bool,
    klasse: int,
) -> None:
    """Update per-bertyp counters for one processed report."""
    key = bertyp or UNKNOWN_BERTYP
    bucket = stats[key]
    bucket["total"] += 1
    if skipped:
        bucket["skipped"] += 1
    elif not failed:
        bucket["sent_to_llm"] += 1
    if int(klasse) == 1:
        bucket["positives"] += 1


def format_bertyp_summary_lines(stats: Dict[str, Dict[str, int]]) -> List[str]:
    """Human-readable report-type summary lines for stdout."""
    if not stats:
        return ["Report type summary:", "  (no reports processed)"]
    lines = ["Report type summary:"]
    for bertyp in sorted(stats.keys()):
        c = stats[bertyp]
        lines.append(
            f"  {bertyp}: total={c['total']}, sent_to_llm={c['sent_to_llm']}, "
            f"skipped={c['skipped']}, positives={c['positives']}"
        )
    return lines


def _bool_csv(b: bool) -> str:
    return "True" if b else "False"


def _base_evidence_metadata(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "original_report_text_length": ev["original_report_text_length"],
        "llm_report_text_length": ev["llm_report_text_length"],
        "llm_text_reduction_method": ev["llm_text_reduction_method"],
        "delir_keyword_hits_count": ev["delir_keyword_hits_count"],
        "has_direct_delir_evidence": _bool_csv(bool(ev["has_direct_delir_evidence"])),
        "has_indirect_delir_evidence": _bool_csv(bool(ev["has_indirect_delir_evidence"])),
        "has_negated_delir_evidence": _bool_csv(bool(ev["has_negated_delir_evidence"])),
        "has_prophylaxis_or_risk_only": _bool_csv(bool(ev["has_prophylaxis_or_risk_only"])),
        "evidence_snippets": evidence_snippets_json_for_csv(ev.get("evidence_snippets") or []),
    }


def _guardrail_fields(guard: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "has_alternative_explanation": _bool_csv(bool(guard.get("has_alternative_explanation", False))),
        "manual_review_candidate": _bool_csv(bool(guard.get("manual_review_candidate", False))),
        "decision_rule_applied": str(guard.get("decision_rule_applied", "")),
    }


def _processing_status_fields(
    *,
    status: str,
    llm_called: int,
    skipped_reason: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "llm_called": int(llm_called),
        "skipped_reason": str(skipped_reason or ""),
    }


def _report_identity_fields(report: dict) -> Dict[str, Any]:
    """Stable keys for cohort export merge (from Berichte.csv row identity)."""
    return {
        SOURCE_REPORT_ROW_ID_COL: str(report.get(SOURCE_REPORT_ROW_ID_COL, "") or "").strip(),
        "berdat": str(report.get("berdat", "") or "").strip(),
    }


def _prediction_row_no_evidence(
    ev: Dict[str, Any],
    patient_id: str,
    report_name: str,
    bertyp: str = "",
) -> Dict[str, Any]:
    guard = apply_clinical_decision_guardrails(
        {"signalstaerke": "niedrig", "kontext": NO_EVIDENCE_KONTEXT, "alternative_erklaerung": False, "begruendung": []},
        {},
        ev,
        llm_skipped=True,
    )
    prob = delirium_probability_estimate(
        "niedrig",
        0,
        decision_rule_applied=str(guard.get("decision_rule_applied", "")),
    )
    return {
        "PatientenID": patient_id,
        "bericht": report_name,
        "bertyp": bertyp,
        **_base_evidence_metadata(ev),
        "delir_probability_estimate": prob,
        **_guardrail_fields(guard),
        "llm_skipped_by_prefilter": True,
        "anzahl_treffer": 0,
        "delir_signale": "",
        "signalstaerke": guard["signalstaerke"],
        "kontext": NO_EVIDENCE_KONTEXT,
        "alternative_erklaerung": False,
        "alternative_erklaerung_keywords": "",
        "begruendung": NO_EVIDENCE_BE,
        "klasse": 0,
        "klassifikation": "kein_delir",
        "klassifikation_begruendung": _KLASSE_NULL_BE + " | " + NO_EVIDENCE_BE,
        **_processing_status_fields(
            status="skipped",
            llm_called=0,
            skipped_reason=str(guard.get("decision_rule_applied", "")) or "no_evidence_prefilter_skip",
        ),
    }


def _prediction_row_pipeline_error(
    ev: Dict[str, Any],
    patient_id: str,
    report_name: str,
    err: str,
    bertyp: str = "",
) -> Dict[str, Any]:
    return {
        "PatientenID": patient_id,
        "bericht": report_name,
        "bertyp": bertyp,
        **_base_evidence_metadata(ev),
        "delir_probability_estimate": 0,
        "llm_skipped_by_prefilter": False,
        "anzahl_treffer": 0,
        "delir_signale": "",
        "signalstaerke": "niedrig",
        "kontext": f"Pipeline-Fehler: {err[:500]}",
        "alternative_erklaerung": False,
        "alternative_erklaerung_keywords": "",
        "begruendung": "Ausführung fehlgeschlagen",
        "klasse": 0,
        "klassifikation": "kein_delir",
        "klassifikation_begruendung": _KLASSE_NULL_BE,
        **_processing_status_fields(
            status="failed",
            llm_called=1,
            skipped_reason="pipeline_error",
        ),
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
                "bertyp": UNKNOWN_BERTYP,
                "report_text": load_report(str(report_path)),
            }
        )

    return rows


def _get_report_records():
    if INPUT_MODE == "berichte":
        if not BERICHTE_INPUT_PATH.exists():
            raise FileNotFoundError(
                f"Primary report input missing: {BERICHTE_INPUT_PATH}. "
                "Expected data/raw/Berichte.csv (semicolon-separated). "
                "Legacy INPUT_MODE='diagnosis' requires synthetic DATA_MODE and synthetic_diagnoses.csv."
            )
        report_records, excluded_db = build_report_level_berichte_records()
        print(f"excluded_dokumentationsblatt_count={excluded_db}")
        LOGGER.info("excluded_dokumentationsblatt_count=%d", excluded_db)
    elif INPUT_MODE == "diagnosis":
        # Legacy: Diagnosenliste-style CSV (synthetic mode only).
        report_records = [
            _normalize_report_record_bertyp(r) for r in build_patient_level_report_records()
        ]
    elif INPUT_MODE == "txt":
        report_records = _load_txt_reports()
    else:
        raise ValueError(f"Ungültiger INPUT_MODE: {INPUT_MODE}")

    if validation_cohort_only_enabled():
        report_records, spec = filter_report_records_for_validation_cohort(report_records)
        print(
            f"VALIDATION_COHORT_ONLY=true: processing {len(report_records)} reports "
            f"from frozen cohort ({FROZEN_PATIENT_VALIDATION_COHORT_PATH.name}, "
            f"filter_mode={spec.filter_mode})"
        )
        if MAX_REPORTS is not None:
            LOGGER.warning(
                "MAX_REPORTS is ignored when VALIDATION_COHORT_ONLY=true "
                "(cohort row set defines the run)."
            )
    elif MAX_REPORTS is not None:
        if isinstance(MAX_REPORTS, int) and MAX_REPORTS > 0:
            report_records = report_records[:MAX_REPORTS]
            print(
                f"Hinweis: MAX_REPORTS aktiv ({MAX_REPORTS}) - "
                f"es werden nur die ersten Berichte verarbeitet."
            )
        else:
            raise ValueError("MAX_REPORTS muss None oder eine positive Ganzzahl sein.")

    return [_normalize_report_record_bertyp(r) for r in report_records]


def _get_output_path() -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if validation_cohort_only_enabled():
        return VALIDATION_COHORT_PREDICTIONS_PATH
    return PREDICTIONS_DIR / f"agent1_agent2_agent3_results_{INTERPRETATION_MODE}.csv"


def _sanitize_provider_model_slug(provider: str, model_label: str) -> str:
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


def _compact_line(
    idx: int,
    total: int,
    patient_id: str,
    ev: Dict[str, Any],
    *,
    bertyp: str,
    status: str,
    klasse: Any,
    signal: str,
    manual_review: bool = False,
    decision_rule: str = "",
) -> str:
    n_snip = len(ev.get("evidence_snippets") or [])
    bertyp_disp = bertyp or UNKNOWN_BERTYP
    line = (
        f"Patient {idx}/{total} | ID={patient_id} | bertyp={bertyp_disp} | evidence={n_snip} | "
        f"original={ev['original_report_text_length']} | llm={ev['llm_report_text_length']} | "
        f"method={ev['llm_text_reduction_method']} | klasse={klasse} | signal={signal} | status={status}"
    )
    if manual_review:
        line += f" | manual_review_candidate=True | rule={decision_rule}"
    return line


def _print_evidence_preview(ev: Dict[str, Any], limit: int = 3) -> None:
    snips: List[Dict[str, Any]] = list(ev.get("evidence_snippets") or [])
    if not snips:
        return
    pos = [s for s in snips if s.get("evidence_type") in ("direct_delir", "indirect_symptom")]
    show = pos[:limit] if pos else snips[:limit]
    print("Evidence:")
    for s in show:
        sec = s.get("section", "")
        et = s.get("evidence_type", "")
        tx = str(s.get("text") or "")[:400]
        print(f"- [{sec} | {et}] {tx}")


def _run_single_report(report: dict, idx: int, total: int) -> Tuple[dict, bool, bool]:
    """
    Returns (row_dict, skipped_no_evidence, failed).

    skipped_no_evidence: rule layer found nothing to send to LLM.
    failed: exception during LLM path (row still written).
    """
    full_report_text = str(report.get("report_text", "") or "")
    patient_id = str(report.get("PatientenID", "") or "").strip()
    default_bericht = (
        f"berichte_{patient_id}.txt" if INPUT_MODE == "berichte" else f"diagnosis_{patient_id}.txt"
    )
    report_name = str(report.get("bericht", default_bericht) or "").strip()
    bertyp = resolve_bertyp(report)

    ev = extract_delirium_evidence(full_report_text)
    snippets = ev.get("evidence_snippets") or []

    if not llm_should_receive_evidence(snippets):
        if should_send_short_report_without_evidence(
            full_report_text,
            bertyp,
            snippets,
            original_length=int(ev.get("original_report_text_length") or 0),
        ):
            ev = apply_short_report_fulltext_to_evidence(ev, full_report_text)
        else:
            row = _prediction_row_no_evidence(ev, patient_id, report_name, bertyp=bertyp)
            msg = _compact_line(
                idx, total, patient_id, ev, bertyp=bertyp, status="skipped", klasse=0, signal="niedrig"
            )
            print(msg)
            LOGGER.info(msg)
            return row, True, False

    llm_text = ev["llm_report_text"]

    if DEBUG_VERBOSE:
        print(
            _compact_line(idx, total, patient_id, ev, bertyp=bertyp, status="llm", klasse="n/a", signal="…")
        )
        print("\n=== DEBUG LLM input (evidence bundle) ===")
        print(f"Patient: {patient_id} | len={len(llm_text)}")
        print(llm_text[:4000])
        print("====================\n")
    else:
        LOGGER.info(
            "Evidence patient=%s snippets=%d llm_len=%d method=%s",
            patient_id,
            len(snippets),
            len(llm_text),
            ev["llm_text_reduction_method"],
        )

    try:
        result = extract_passages(llm_text, patient_id=patient_id, report_name=report_name)

        if INTERPRETATION_MODE == "rule":
            interpretation = interpret_signals(llm_text, result)
        elif INTERPRETATION_MODE == "prompt":
            interpretation = interpret_signals_llm(
                llm_text, result, patient_id=patient_id, report_name=report_name
            )
        else:
            raise ValueError(f"Ungültiger INTERPRETATION_MODE: {INTERPRETATION_MODE}")

        classification = classify_delirium(interpretation)
        guard = apply_clinical_decision_guardrails(
            interpretation,
            result,
            ev,
            llm_skipped=False,
        )
        final_klasse = int(guard["klasse"])
        final_signal = str(guard["signalstaerke"])
        final_kontext = str(guard.get("kontext") or interpretation.get("kontext", ""))
        final_begr = list(guard.get("begruendung") or [])
        klassifikation_begr_str = (
            " | ".join(str(x) for x in final_begr)
            if final_begr
            else " | ".join(classification.get("begruendung", []))
        )

        hits: List[str] = []
        for key in SIGNAL_KEYS:
            values = result.get(key, [])
            if isinstance(values, list):
                hits.extend(values)

        if DEBUG_VERBOSE:
            print(f"[{report_name}] PatientenID={patient_id} | Treffer gesamt: {len(hits)}")
            if hits:
                for key in SIGNAL_KEYS:
                    values = result.get(key, [])
                    if isinstance(values, list) and values:
                        print(f"  [{key}]")
                        for hit_idx, hit in enumerate(values, start=1):
                            print(f"    {hit_idx}. {hit}")
            print("  [interpretation]")
            print(f"    signalstaerke: {interpretation['signalstaerke']}")
            print(f"    kontext: {interpretation['kontext']}")
            print("  [classification]")
            print(f"    klasse: {final_klasse}")
            print(f"    klassifikation: {guard['klassifikation']}")
            print(f"    decision_rule: {guard.get('decision_rule_applied')}")
            print()
        else:
            msg = _compact_line(
                idx,
                total,
                patient_id,
                ev,
                bertyp=bertyp,
                status="success",
                klasse=final_klasse,
                signal=final_signal,
                manual_review=bool(guard.get("manual_review_candidate")),
                decision_rule=str(guard.get("decision_rule_applied", "")),
            )
            print(msg)
            if final_klasse == 1:
                _print_evidence_preview(ev)
            LOGGER.info(msg)

        prob = delirium_probability_estimate(
            final_signal,
            final_klasse,
            manual_review_candidate=bool(guard.get("manual_review_candidate")),
            decision_rule_applied=str(guard.get("decision_rule_applied", "")),
            has_direct_delir_evidence=bool(ev.get("has_direct_delir_evidence")),
        )
        row: Dict[str, Any] = {
            "PatientenID": patient_id,
            "bericht": report_name,
            "bertyp": bertyp,
            **_base_evidence_metadata(ev),
            **_guardrail_fields(guard),
            "llm_skipped_by_prefilter": False,
            "anzahl_treffer": len(hits),
            "delir_signale": " | ".join(hits),
            "signalstaerke": final_signal,
            "delir_probability_estimate": prob,
            "kontext": final_kontext,
            "alternative_erklaerung": guard.get("alternative_erklaerung", interpretation["alternative_erklaerung"]),
            "alternative_erklaerung_keywords": " | ".join(
                interpretation.get("alternative_erklaerung_keywords", [])
            ),
            "begruendung": " | ".join(interpretation.get("begruendung", [])),
            "klasse": final_klasse,
            "klassifikation": guard["klassifikation"],
            "klassifikation_begruendung": klassifikation_begr_str,
            **_processing_status_fields(
                status="processed",
                llm_called=1,
                skipped_reason=str(guard.get("decision_rule_applied", "")),
            ),
        }
        return row, False, False

    except Exception as exc:
        LOGGER.exception("Pipeline failure patient=%s", patient_id)
        err = f"{type(exc).__name__}: {exc}"
        row = _prediction_row_pipeline_error(ev, patient_id, report_name, err, bertyp=bertyp)
        msg = _compact_line(
            idx, total, patient_id, ev, bertyp=bertyp, status="failed", klasse=0, signal="niedrig"
        )
        print(msg)
        if DEBUG_VERBOSE:
            print(traceback.format_exc())
        LOGGER.error("%s | %s", msg, err)
        return row, False, True


def main():
    output_csv = _get_output_path()
    report_records = _get_report_records()
    total = len(report_records)

    print(f"\n=== Agent 1 + Agent 2 + Agent 3: Delir-Pipeline ({INTERPRETATION_MODE}) ===")
    print(f"Anzahl Berichte: {total}\n")

    rows: List[Dict[str, Any]] = []
    n_prefilter_skip = 0
    n_sent_short_no_evidence = 0
    n_llm = 0
    n_failed = 0
    n_k0 = n_k1 = 0
    sig_counts: Dict[str, int] = {}
    sum_orig = sum_llm = 0
    bertyp_stats: Dict[str, Dict[str, int]] = defaultdict(_new_bertyp_stats)

    for i, report in enumerate(report_records, start=1):
        row_dict, skipped, failed = _run_single_report(report, i, total)
        row_dict.update(_report_identity_fields(report))
        rows.append(row_dict)
        accumulate_bertyp_stat(
            bertyp_stats,
            str(row_dict.get("bertyp") or UNKNOWN_BERTYP),
            skipped=skipped,
            failed=failed,
            klasse=int(row_dict.get("klasse") or 0),
        )
        sum_orig += int(row_dict.get("original_report_text_length") or 0)
        sum_llm += int(row_dict.get("llm_report_text_length") or 0)
        k = int(row_dict.get("klasse") or 0)
        if k == 1:
            n_k1 += 1
        else:
            n_k0 += 1
        sig = str(row_dict.get("signalstaerke") or "")
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        if failed:
            n_failed += 1
        elif skipped:
            n_prefilter_skip += 1
        else:
            n_llm += 1
            if str(row_dict.get("llm_text_reduction_method") or "") == METHOD_SHORT_REPORT_FULLTEXT:
                n_sent_short_no_evidence += 1

    LOGGER.info(
        "Run summary: total=%d skipped=%d llm=%d failed=%d klasse0=%d klasse1=%d",
        total,
        n_prefilter_skip,
        n_llm,
        n_failed,
        n_k0,
        n_k1,
    )

    avg_orig = sum_orig / total if total else 0.0
    avg_llm = sum_llm / total if total else 0.0

    print("\n=== Run summary ===")
    print(f"total_reports={total}")
    print(f"sent_to_llm={n_llm}")
    print(f"skipped_no_evidence={n_prefilter_skip}")
    print(f"sent_short_no_evidence={n_sent_short_no_evidence}")
    print(f"failed={n_failed}")
    print(f"klasse: 0={n_k0}, 1={n_k1}")
    print(f"signalstaerke: {sig_counts}")
    print(f"avg_original_length={avg_orig:.1f}")
    print(f"avg_llm_input_length={avg_llm:.1f}")
    for line in format_bertyp_summary_lines(dict(bertyp_stats)):
        print(line)

    _assert_binary_klassen(rows)

    fieldnames = [
        "PatientenID",
        "bericht",
        "bertyp",
        "berdat",
        SOURCE_REPORT_ROW_ID_COL,
        "original_report_text_length",
        "llm_report_text_length",
        "llm_text_reduction_method",
        "delir_keyword_hits_count",
        "has_direct_delir_evidence",
        "has_indirect_delir_evidence",
        "has_negated_delir_evidence",
        "has_prophylaxis_or_risk_only",
        "has_alternative_explanation",
        "manual_review_candidate",
        "decision_rule_applied",
        "status",
        "llm_called",
        "skipped_reason",
        "llm_skipped_by_prefilter",
        "anzahl_treffer",
        "delir_signale",
        "evidence_snippets",
        "signalstaerke",
        "delir_probability_estimate",
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

    if os.environ.get("ENABLE_SQLITE_LOGGING", "").strip().lower() in ("1", "true", "yes"):
        from src.pipeline.sqlite_logging import init_prediction_db, log_prediction_row

        init_prediction_db(SQLITE_PREDICTIONS_DB_PATH)
        for row_dict in rows:
            log_prediction_row(SQLITE_PREDICTIONS_DB_PATH, row_dict)
        print(f"SQLite prediction log: {SQLITE_PREDICTIONS_DB_PATH}")

    print(f"Ergebnisse gespeichert in: {output_csv}")
    if validation_cohort_only_enabled():
        print(
            "VALIDATION_COHORT_ONLY: full predictions file was NOT updated "
            f"({PREDICTIONS_DIR / f'agent1_agent2_agent3_results_{INTERPRETATION_MODE}.csv'} unchanged)."
        )
    else:
        model_copy_path = _get_model_named_output_path()
        shutil.copy2(output_csv, model_copy_path)
        print(f"Ergebnisse (Modellkopie) gespeichert in: {model_copy_path}")


if __name__ == "__main__":
    main()
