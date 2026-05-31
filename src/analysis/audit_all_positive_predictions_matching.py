"""
Audit matching for all model-positive validation reports (read-only).

Outputs mismatch list for FP/debug review when evidence may not belong to raw report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.validation_report_trace import (
    build_report_trace,
    load_trace_inputs,
    trace_to_mismatch_record,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)

LOGGER = logging.getLogger(__name__)

MATCHING_AUDIT_POSITIVE_DIR = MANUAL_VALIDATION_DIR / "matching_audit_positive"


def _is_positive_report(row: pd.Series) -> bool:
    val = row.get("model_report_prediction", row.get("klasse", 0))
    try:
        return int(pd.to_numeric(val, errors="coerce") or 0) == 1
    except (TypeError, ValueError):
        return False


def run_positive_prediction_matching_audit(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    output_dir: Path = MATCHING_AUDIT_POSITIVE_DIR,
) -> tuple[pd.DataFrame, str]:
    cohort, labels, preds, spine, raw_full = load_trace_inputs(
        cohort_path, labels_path, predictions_path, berichte_path
    )

    positive = cohort[cohort.apply(_is_positive_report, axis=1)].copy()
    records = []
    verdict_counts = {"MATCH_OK": 0, "MATCH_SUSPICIOUS": 0, "MATCH_FAIL": 0}

    for rid in positive["validation_report_id"].astype(str):
        trace = build_report_trace(rid, cohort, labels, preds, spine, raw_full)
        verdict_counts[trace.verdict] = verdict_counts.get(trace.verdict, 0) + 1
        rec = trace_to_mismatch_record(trace)
        records.append(rec)

    df = pd.DataFrame(records)
    mismatches = df[df["verdict"] != "MATCH_OK"].copy() if not df.empty else df

    lines = [
        "Positive prediction matching audit",
        "=" * 44,
        f"positive_report_rows={len(positive)}",
        f"MATCH_OK={verdict_counts.get('MATCH_OK', 0)}",
        f"MATCH_SUSPICIOUS={verdict_counts.get('MATCH_SUSPICIOUS', 0)}",
        f"MATCH_FAIL={verdict_counts.get('MATCH_FAIL', 0)}",
        f"mismatch_rows={len(mismatches)}",
        "",
        "Review MATCH_FAIL rows first — evidence may be attached to wrong raw report.",
        "Use trace_validation_report for single-report deep dive.",
    ]
    report = "\n".join(lines) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "positive_prediction_matching_report.txt").write_text(
        report, encoding="utf-8"
    )
    mismatches.to_csv(output_dir / "positive_prediction_mismatches.csv", index=False)

    LOGGER.info(
        "Positive audit: %d positive, %d mismatches -> %s",
        len(positive),
        len(mismatches),
        output_dir,
    )
    return mismatches, report


def main() -> None:
    mismatches, report = run_positive_prediction_matching_audit()
    print(report)
    print(f"Wrote outputs to {MATCHING_AUDIT_POSITIVE_DIR}")
    if not mismatches.empty:
        print(f"Mismatches: {len(mismatches)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
