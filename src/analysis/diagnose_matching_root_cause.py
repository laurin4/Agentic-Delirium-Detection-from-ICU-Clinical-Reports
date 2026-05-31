"""
Diagnose prediction ↔ cohort matching failures (read-only).

Analyzes MATCH_FAIL cases from the positive prediction audit and writes
outputs/analysis/manual_validation/matching_root_cause_report.txt
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.audit_all_positive_predictions_matching import (
    MATCHING_AUDIT_POSITIVE_DIR,
    _is_positive_report,
)
from src.analysis.validation_report_trace import (
    build_report_trace,
    load_trace_inputs,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.preprocessing.report_identity import (
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
)

LOGGER = logging.getLogger(__name__)

MATCHING_ROOT_CAUSE_REPORT_PATH = MANUAL_VALIDATION_DIR / "matching_root_cause_report.txt"

FAILURE_CLASS_STALE_SOURCE_ROW_ID = "stale_positional_source_report_row_id"
FAILURE_CLASS_LEGACY_PIPELINE_BERICHT_COLLISION = "legacy_pipeline_bericht_collision"
FAILURE_CLASS_PREDICTION_IDENTITY_DRIFT = "prediction_identity_field_drift"
FAILURE_CLASS_NO_PREDICTION_ROW = "no_prediction_row"
FAILURE_CLASS_EVIDENCE_NOT_IN_RAW = "evidence_not_in_raw_report"
FAILURE_CLASS_OTHER = "other"


def _classify_failure(trace) -> str:
    issues = " | ".join(trace.issues).lower()
    cohort = trace.cohort_row or {}
    pred = trace.prediction_row or {}

    if "no_prediction_row" in issues:
        return FAILURE_CLASS_NO_PREDICTION_ROW

    cohort_sid = str(cohort.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
    pred_sid = str(pred.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
    cohort_bertyp = str(cohort.get("bertyp", "")).strip()
    pred_bertyp = str(pred.get("bertyp", "")).strip()
    cohort_berdat = str(cohort.get("berdat", "")).strip()
    pred_berdat = str(pred.get("berdat", "")).strip()
    cohort_pid = str(cohort.get("PatientenID", "")).strip()
    pred_pid = str(pred.get("PatientenID", "")).strip()

    if cohort_sid and pred_sid and cohort_sid == pred_sid:
        if cohort_bertyp and pred_bertyp and cohort_bertyp != pred_bertyp:
            return FAILURE_CLASS_STALE_SOURCE_ROW_ID
        if cohort_berdat and pred_berdat and cohort_berdat != pred_berdat:
            return FAILURE_CLASS_STALE_SOURCE_ROW_ID
        if cohort_pid and pred_pid and cohort_pid != pred_pid:
            return FAILURE_CLASS_STALE_SOURCE_ROW_ID

    if "patientenid_mismatch" in issues or "bertyp" in issues or "berdat" in issues:
        if trace.merge_strategy in ("patientenid_pipeline_bericht", "fallback_patientenid_bertyp_berdat_bericht"):
            return FAILURE_CLASS_LEGACY_PIPELINE_BERICHT_COLLISION
        return FAILURE_CLASS_PREDICTION_IDENTITY_DRIFT

    if "evidence_not_in_raw_report" in issues:
        return FAILURE_CLASS_EVIDENCE_NOT_IN_RAW

    return FAILURE_CLASS_OTHER


def _failure_detail_row(trace) -> Dict[str, Any]:
    cohort = trace.cohort_row or {}
    pred = trace.prediction_row or {}
    return {
        "validation_report_id": trace.validation_report_id,
        "validation_patient_id": cohort.get("validation_patient_id", ""),
        "failure_class": _classify_failure(trace),
        "verdict": trace.verdict,
        "merge_strategy": trace.merge_strategy,
        "merge_key": trace.merge_key,
        "source_report_row_id_cohort": cohort.get(SOURCE_REPORT_ROW_ID_COL, ""),
        "source_report_row_id_prediction": pred.get(SOURCE_REPORT_ROW_ID_COL, ""),
        "PatientenID_cohort": cohort.get("PatientenID", ""),
        "PatientenID_prediction": pred.get("PatientenID", ""),
        "bertyp_cohort": cohort.get("bertyp", ""),
        "bertyp_prediction": pred.get("bertyp", ""),
        "berdat_cohort": cohort.get("berdat", ""),
        "berdat_prediction": pred.get("berdat", ""),
        "pipeline_bericht_cohort": cohort.get(PIPELINE_BERICHT_COL, cohort.get("bericht", "")),
        "pipeline_bericht_prediction": pred.get(PIPELINE_BERICHT_COL, pred.get("bericht", "")),
        "issues": " | ".join(trace.issues),
    }


def analyze_matching_failures(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cohort, labels, preds, spine, raw_full = load_trace_inputs(
        cohort_path, labels_path, predictions_path, berichte_path
    )

    positive = cohort[cohort.apply(_is_positive_report, axis=1)]
    fail_traces = []
    all_traces = []

    for rid in cohort["validation_report_id"].astype(str):
        trace = build_report_trace(rid, cohort, labels, preds, spine, raw_full)
        all_traces.append(trace)
        if trace.verdict == "MATCH_FAIL":
            fail_traces.append(trace)

    fail_rows = [_failure_detail_row(t) for t in fail_traces]
    failure_classes = Counter(r["failure_class"] for r in fail_rows)

    preds_have_source = (
        SOURCE_REPORT_ROW_ID_COL in preds.columns
        and preds[SOURCE_REPORT_ROW_ID_COL].astype(str).str.strip().ne("").any()
    )
    spine_dup_pber = 0
    if PIPELINE_BERICHT_COL in spine.columns:
        dup = spine.duplicated(subset=["PatientenID", PIPELINE_BERICHT_COL], keep=False)
        spine_dup_pber = int(dup.sum())

    affected_patients = set()
    for r in fail_rows:
        pid = str(r.get("PatientenID_cohort") or r.get("PatientenID_prediction") or "").strip()
        if pid:
            affected_patients.add(pid)

    positive_fail = [
        r for r in fail_rows if r["validation_report_id"] in set(positive["validation_report_id"].astype(str))
    ]

    summary = {
        "total_cohort_rows": len(cohort),
        "total_positive_reports": len(positive),
        "total_match_fail_rows": len(fail_rows),
        "positive_match_fail_rows": len(positive_fail),
        "affected_patients": len(affected_patients),
        "failure_classes": dict(failure_classes),
        "predictions_have_source_report_row_id": bool(preds_have_source),
        "spine_duplicate_patient_pipeline_bericht_rows": spine_dup_pber,
        "primary_merge_strategy": fail_traces[0].merge_strategy if fail_traces else "",
    }
    return fail_rows, summary


def determine_root_cause(summary: Dict[str, Any], fail_rows: List[Dict[str, Any]]) -> str:
    classes = summary.get("failure_classes") or {}
    stale = classes.get(FAILURE_CLASS_STALE_SOURCE_ROW_ID, 0)
    legacy = classes.get(FAILURE_CLASS_LEGACY_PIPELINE_BERICHT_COLLISION, 0)
    drift = classes.get(FAILURE_CLASS_PREDICTION_IDENTITY_DRIFT, 0)
    evidence = classes.get(FAILURE_CLASS_EVIDENCE_NOT_IN_RAW, 0)

    if stale > 0:
        return (
            "PRIMARY: Positional source_report_row_id drift — "
            "source_report_row_id is assigned as berichte_row_<pandas_index> on the full Berichte.csv "
            "load order. When Berichte.csv changes (rows added/removed/skipped) OR when predictions "
            "were produced in a different run than the frozen cohort spine, the same "
            "source_report_row_id can point to a different clinical report. Predictions keep stale "
            "bertyp/berdat/PatientenID/evidence from the original row while the cohort spine shows "
            "the current row at that index → MATCH_FAIL with same source_report_row_id but mismatched "
            "identity fields and evidence not in raw text."
        )
    if legacy > 0 or drift > 0:
        return (
            "PRIMARY: Prediction merge key mismatch — predictions were linked via "
            "PatientenID+pipeline_bericht (legacy) or fallback keys instead of stable report identity. "
            "Legacy enrichment uses drop_duplicates(..., keep='first') on (PatientenID, pipeline_bericht) "
            "which can attach the wrong source_report_row_id when keys collide. "
            "Wrong prediction rows inherit mismatched bertyp/berdat/evidence."
        )
    if evidence > 0:
        return (
            "PRIMARY: Evidence text not found in reconstructed raw report — likely consequence of "
            "wrong report linkage (see stale source_report_row_id or merge key collision)."
        )
    return "Unable to classify from available MATCH_FAIL rows; inspect sample failures manually."


def format_root_cause_report(
    fail_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    root_cause = determine_root_cause(summary, fail_rows)
    lines = [
        "Matching root cause diagnosis",
        "=" * 44,
        "",
        "ROOT CAUSE",
        "-" * 44,
        root_cause,
        "",
        "SUMMARY COUNTS",
        "-" * 44,
        f"total_cohort_rows={summary.get('total_cohort_rows', 0)}",
        f"total_positive_reports={summary.get('total_positive_reports', 0)}",
        f"match_fail_rows={summary.get('total_match_fail_rows', 0)}",
        f"positive_match_fail_rows={summary.get('positive_match_fail_rows', 0)}",
        f"affected_patients={summary.get('affected_patients', 0)}",
        f"predictions_have_source_report_row_id={summary.get('predictions_have_source_report_row_id')}",
        f"spine_duplicate_patient_pipeline_bericht_rows={summary.get('spine_duplicate_patient_pipeline_bericht_rows', 0)}",
        "",
        "FAILURE CLASS COUNTS",
        "-" * 44,
    ]
    for cls, cnt in sorted((summary.get("failure_classes") or {}).items()):
        lines.append(f"  {cls}={cnt}")

    lines.extend(
        [
            "",
            "MANUAL GROUND TRUTH AFFECTED?",
            "-" * 44,
            "Manual report labels (manual_report_ground_truth) are keyed by validation_report_id on the "
            "frozen Berichte spine row ordering. Labels remain attached to the correct validation_report_id "
            "for the CURRENT spine, but model_report_prediction / evidence_snippets on that same row may "
            "come from a different clinical report if prediction merge failed.",
            "=> Manual GT per validation_report_id is still valid for the report shown in cohort, BUT "
            "model predictions and evidence on that row are NOT trustworthy until merge is fixed.",
            "",
            "EVALUATION METRICS AFFECTED?",
            "-" * 44,
            "YES — model_patient_positive, TP/FP/TN/FN vs manual GT, and evidence-based FP review are "
            "affected wherever MATCH_FAIL applies to model-positive rows.",
            "Final evaluation MUST be re-run after predictions are re-linked with stable keys and "
            "validation_cohort_predictions.csv is regenerated from a cohort-aligned pipeline run.",
            "",
            "EXACT FIX REQUIRED",
            "-" * 44,
            "1. Stop using positional berichte_row_<index> as the sole merge key across runs.",
            "2. Re-run pipeline (VALIDATION_COHORT_ONLY=true) on the SAME Berichte.csv snapshot used for frozen cohort.",
            "3. Merge predictions to spine on stable composite key: (PatientenID, bertyp, berdat, bername/pipeline_bericht) "
            "with source_report_row_id as secondary check — reject merge when bertyp/berdat/PatientenID disagree.",
            "4. Remove or fix legacy drop_duplicates(..., keep='first') enrichment in _prepare_predictions_for_merge.",
            "5. Regenerate validation_cohort_predictions.csv; re-export or re-merge cohort predictions WITHOUT "
            "touching manual_report_labels_frozen.csv; re-run final_manual_validation_evaluation.",
            "",
            "MATCH_FAIL CASE DETAILS (all failing validation_report_id)",
            "-" * 44,
        ]
    )

    if not fail_rows:
        lines.append("(no MATCH_FAIL rows detected in current inputs)")
    else:
        for row in fail_rows:
            lines.append("")
            for key in (
                "validation_report_id",
                "validation_patient_id",
                "failure_class",
                "merge_strategy",
                "merge_key",
                "source_report_row_id_cohort",
                "source_report_row_id_prediction",
                "PatientenID_cohort",
                "PatientenID_prediction",
                "bertyp_cohort",
                "bertyp_prediction",
                "berdat_cohort",
                "berdat_prediction",
                "pipeline_bericht_cohort",
                "pipeline_bericht_prediction",
                "issues",
            ):
                lines.append(f"  {key}={row.get(key, '')}")

    lines.append("")
    return "\n".join(lines)


def run_matching_root_cause_diagnosis(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    output_path: Path = MATCHING_ROOT_CAUSE_REPORT_PATH,
) -> str:
    fail_rows, summary = analyze_matching_failures(
        cohort_path, labels_path, predictions_path, berichte_path
    )
    report = format_root_cause_report(fail_rows, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    audit_dir = MATCHING_AUDIT_POSITIVE_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fail_rows).to_csv(audit_dir / "all_match_fail_details.csv", index=False)

    LOGGER.info("Wrote matching root cause report: %s", output_path)
    return report


def main() -> None:
    report = run_matching_root_cause_diagnosis()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
