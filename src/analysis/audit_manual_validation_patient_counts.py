"""
Reconcile manual validation patient counts (frozen cohort).

Use when numbers disagree between:
  - final_evaluation/report.txt (complete_patients)
  - baseline_manual_comparison_summary.txt (Patients evaluated)
  - manual_validation_progress_report.txt

Run:
  python -m src.analysis.audit_manual_validation_patient_counts
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.analysis.export_baseline_manual_comparison_summary import (
    _binary_col,
    build_patient_table_from_frozen_cohort,
    resolve_comparison_columns,
)
from src.analysis.final_manual_validation_evaluation import (
    MANUAL_GT_COL,
    primary_evaluation_cohort,
)
from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_COHORT_METADATA_PATH,
    MANUAL_VALIDATION_PROGRESS_REPORT_PATH,
)

SIGNAL_COLUMNS: Tuple[str, ...] = (
    "derived_manual_patient_ground_truth",
    "model_patient_positive",
    "baseline_icdsc_ge_4",
    "baseline_icd10",
    "baseline_composite_or",
    "baseline_composite_and",
)


def _missing_signal_reasons(row: pd.Series, cols) -> List[str]:
    reasons: List[str] = []
    checks = (
        ("manual", cols.manual),
        ("v2_model", cols.v2),
        ("icdsc", cols.icdsc),
        ("icd10", cols.icd10),
        ("composite_or", cols.composite_or),
        ("composite_and", cols.composite_and),
    )
    for label, col in checks:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val) or int(val) not in (0, 1):
            reasons.append(f"missing_{label}")
    return reasons


def audit_patient_counts(patient_gt: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    cols, col_errors = resolve_comparison_columns(patient_gt)
    if cols is None:
        return "\n".join(["Column mapping failed:", *col_errors]) + "\n", pd.DataFrame()

    complete = primary_evaluation_cohort(patient_gt)
    manual = _binary_col(complete, cols.manual)
    valid_mask = pd.Series(True, index=complete.index)
    for col_name in (
        cols.manual,
        cols.v2,
        cols.icdsc,
        cols.icd10,
        cols.composite_or,
        cols.composite_and,
    ):
        series = _binary_col(complete, col_name)
        valid_mask &= series.isin([0, 1])

    with_all_signals = complete.loc[valid_mask].copy()
    dropped = complete.loc[~valid_mask].copy()

    drop_rows = []
    for idx, row in dropped.iterrows():
        drop_rows.append(
            {
                "validation_patient_id": row.get("validation_patient_id", ""),
                "PatientenID": row.get("PatientenID", ""),
                MANUAL_GT_COL: row.get(MANUAL_GT_COL, ""),
                "model_patient_positive": row.get("model_patient_positive", ""),
                "drop_reasons": "; ".join(_missing_signal_reasons(row, cols)),
            }
        )
    dropped_df = pd.DataFrame(drop_rows)

    n_total = len(patient_gt)
    n_complete = len(complete)
    n_incomplete = n_total - n_complete
    n_signals = len(with_all_signals)
    n_dropped = len(dropped)

    manual_complete = _binary_col(complete, cols.manual)
    manual_signals = _binary_col(with_all_signals, cols.manual)

    lines = [
        "MANUAL VALIDATION PATIENT COUNT AUDIT",
        "=" * 40,
        "",
        "Tier 1 — Frozen cohort",
        f"  total_frozen_patients:           {n_total}",
        f"  incomplete_manual_labels:        {n_incomplete}",
        "",
        "Tier 2 — Primary manual evaluation (all reports labeled 0/1)",
        f"  complete_patients:               {n_complete}",
        f"  manual_positive:                 {int((manual_complete == 1).sum())}",
        f"  manual_negative:                 {int((manual_complete == 0).sum())}",
        "",
        "Tier 3 — Baseline comparison summary (complete + model + baselines)",
        f"  patients_with_all_signals:       {n_signals}",
        f"  dropped_from_baseline_summary:   {n_dropped}",
    ]
    if n_signals:
        lines.extend(
            [
                f"  manual_positive (tier 3):        {int((manual_signals == 1).sum())}",
                f"  manual_negative (tier 3):        {int((manual_signals == 0).sum())}",
            ]
        )

    if n_dropped:
        lines.extend(
            [
                "",
                "Dropped patients (complete manual labels, missing model or baseline signal)",
                "-" * 72,
            ]
        )
        for _, row in dropped_df.iterrows():
            lines.append(
                f"  {row['validation_patient_id']} | PatientenID={row['PatientenID']} | "
                f"manual={row[MANUAL_GT_COL]} | reasons: {row['drop_reasons']}"
            )
        lines.extend(
            [
                "",
                "Interpretation",
                "-" * 72,
                "If complete_patients=99 but patients_with_all_signals=97, two patients are",
                "fully labeled manually but lack a valid V2 patient prediction and/or structured",
                "baseline values. Re-run validation cohort predictions or check PatientenID in",
                "structured_baseline.csv for those IDs.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "OK: every complete patient has manual labels, model prediction, and baselines.",
            ]
        )

    return "\n".join(lines) + "\n", dropped_df


def main(
    output_path: Optional[Path] = None,
    dropped_csv_path: Optional[Path] = None,
) -> None:
    patient_gt, pred_src, base_src = build_patient_table_from_frozen_cohort()
    report, dropped_df = audit_patient_counts(patient_gt)

    header = [
        f"predictions_source: {pred_src}",
        f"baseline_source: {base_src}",
    ]
    if FROZEN_COHORT_METADATA_PATH.exists():
        header.append(f"frozen_metadata: {FROZEN_COHORT_METADATA_PATH}")
    full_report = "\n".join(header) + "\n\n" + report

    out = output_path or (FINAL_MANUAL_VALIDATION_EVAL_DIR / "patient_count_audit.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_report, encoding="utf-8")

    if not dropped_df.empty:
        csv_out = dropped_csv_path or (FINAL_MANUAL_VALIDATION_EVAL_DIR / "baseline_summary_dropped_patients.csv")
        dropped_df.to_csv(csv_out, index=False)
        print(f"Dropped patients CSV: {csv_out}")

    print(full_report)
    print(f"Saved: {out}")
    if MANUAL_VALIDATION_PROGRESS_REPORT_PATH.exists():
        print(f"Also compare: {MANUAL_VALIDATION_PROGRESS_REPORT_PATH}")
    print(f"Also compare: {FINAL_MANUAL_VALIDATION_EVAL_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
