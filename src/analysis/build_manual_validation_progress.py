"""
Patient-level manual validation annotation progress and aggregation.

Reads frozen cohort + frozen manual report labels (does not overwrite frozen files).
Writes progress table and summary report under manual_validation/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.manual_report_labels import (
    _normalize_manual_report_gt,
    merge_manual_report_labels,
)
from src.pipeline.paths import (
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_PROGRESS_PATH,
    MANUAL_VALIDATION_PROGRESS_REPORT_PATH,
)

LOGGER = logging.getLogger(__name__)

PROGRESS_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "PatientenID",
    "n_reports_total",
    "n_reports_labeled",
    "n_reports_missing_label",
    "is_patient_complete",
    "n_positive_reports_manual",
    "derived_manual_patient_ground_truth",
    "model_patient_positive",
    "confusion_group",
)


def _parse_report_gt(series: pd.Series) -> pd.Series:
    return series.map(_normalize_manual_report_gt)


def assign_confusion_group(model_pos: Any, derived: Any) -> str:
    """TP/FP/TN/FN when both model and derived manual GT are valid 0/1; else empty."""
    if derived is None or (isinstance(derived, float) and pd.isna(derived)):
        return ""
    if model_pos is None or (isinstance(model_pos, float) and pd.isna(model_pos)):
        return ""
    try:
        mp = int(model_pos)
        d = int(derived)
    except (TypeError, ValueError):
        return ""
    if mp not in (0, 1) or d not in (0, 1):
        return ""
    if mp == 1 and d == 1:
        return "TP"
    if mp == 1 and d == 0:
        return "FP"
    if mp == 0 and d == 0:
        return "TN"
    if mp == 0 and d == 1:
        return "FN"
    return ""


def _patient_model_positive(grp: pd.DataFrame) -> Optional[int]:
    """Patient positive = max report prediction (current merged klasse), not frozen cohort snapshot."""
    if "model_report_prediction" in grp.columns:
        pred = pd.to_numeric(grp["model_report_prediction"], errors="coerce")
        valid = pred[pred.isin([0, 1])]
        if not valid.empty:
            return int(valid.max())
    if "model_patient_positive" in grp.columns:
        vals = pd.to_numeric(grp["model_patient_positive"], errors="coerce").dropna()
        if len(vals):
            v = int(vals.max())
            if v in (0, 1):
                return v
    return None


def build_manual_validation_progress(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate report-level manual labels to one row per ``validation_patient_id``.

    Empty manual labels are not treated as 0. Incomplete patients get empty derived GT.
    """
    if "validation_patient_id" not in cohort.columns:
        raise ValueError("cohort must contain validation_patient_id")

    df = cohort.copy()
    gt_col = df["manual_report_ground_truth"] if "manual_report_ground_truth" in df.columns else pd.Series(
        index=df.index, dtype=object
    )
    df["_gt"] = _parse_report_gt(gt_col)

    rows: List[Dict[str, Any]] = []
    for vpid, grp in df.groupby("validation_patient_id", sort=True):
        n_total = int(len(grp))
        labeled = grp["_gt"].notna()
        n_labeled = int(labeled.sum())
        n_missing = n_total - n_labeled
        is_complete = n_total > 0 and n_missing == 0

        n_pos = int((grp.loc[labeled, "_gt"] == "1").sum())
        if is_complete:
            derived: Any = 1 if n_pos > 0 else 0
        else:
            derived = pd.NA

        model_pos = _patient_model_positive(grp)
        confusion = assign_confusion_group(model_pos, derived)

        patienten_id = ""
        if "PatientenID" in grp.columns:
            pid_vals = grp["PatientenID"].dropna()
            if len(pid_vals):
                patienten_id = str(pid_vals.iloc[0])

        rows.append(
            {
                "validation_patient_id": vpid,
                "PatientenID": patienten_id,
                "n_reports_total": n_total,
                "n_reports_labeled": n_labeled,
                "n_reports_missing_label": n_missing,
                "is_patient_complete": int(is_complete),
                "n_positive_reports_manual": n_pos,
                "derived_manual_patient_ground_truth": derived,
                "model_patient_positive": model_pos if model_pos is not None else pd.NA,
                "confusion_group": confusion,
            }
        )

    out = pd.DataFrame(rows)
    for col in PROGRESS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(PROGRESS_COLUMNS)]


def format_progress_report(progress: pd.DataFrame) -> str:
    """Human-readable summary of annotation progress and confusion counts."""
    lines = [
        "Manual validation progress",
        "=" * 44,
        "",
    ]
    n_patients = len(progress)
    complete = int(progress["is_patient_complete"].sum()) if n_patients else 0
    incomplete = n_patients - complete
    lines.append(f"total_patients={n_patients}")
    lines.append(f"complete_patients={complete}")
    lines.append(f"incomplete_patients={incomplete}")

    complete_df = progress[progress["is_patient_complete"] == 1]
    if not complete_df.empty:
        derived = pd.to_numeric(complete_df["derived_manual_patient_ground_truth"], errors="coerce")
        lines.append(f"manual_positive_patients={int((derived == 1).sum())}")
        lines.append(f"manual_negative_patients={int((derived == 0).sum())}")
    else:
        lines.append("manual_positive_patients=0")
        lines.append("manual_negative_patients=0")

    if "confusion_group" in progress.columns:
        cg = progress["confusion_group"].astype(str).str.strip()
        for label in ("TP", "FP", "TN", "FN"):
            lines.append(f"{label}={int((cg == label).sum())}")

    if n_patients:
        lines.extend(
            [
                "",
                "Report-level totals (across cohort reports in progress build)",
                f"  sum_n_reports_total={int(progress['n_reports_total'].sum())}",
                f"  sum_n_reports_labeled={int(progress['n_reports_labeled'].sum())}",
                f"  sum_n_reports_missing_label={int(progress['n_reports_missing_label'].sum())}",
            ]
        )

    lines.extend(
        [
            "",
            "Rules",
            "-" * 44,
            "- Patient complete: every report has manual_report_ground_truth in {0,1}.",
            "- derived_manual_patient_ground_truth: 1 if any report is 1; 0 if all labeled and all 0;",
            "  empty if patient incomplete.",
            "- Empty manual labels are NOT treated as 0.",
            "- confusion_group: TP/FP/TN/FN vs model_patient_positive when patient complete;",
            "  empty if incomplete or model unavailable.",
        ]
    )
    return "\n".join(lines) + "\n"


def _empty_for_csv(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value


def build_progress_from_files(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
) -> pd.DataFrame:
    """Load frozen cohort and labels, merge, and compute patient-level progress."""
    if not cohort_path.exists():
        raise FileNotFoundError(f"Frozen patient validation cohort missing: {cohort_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Frozen manual report labels missing: {labels_path}")

    cohort = pd.read_csv(cohort_path)
    labels = pd.read_csv(labels_path)
    merged = merge_manual_report_labels(
        cohort, labels, log_context="manual validation progress"
    )
    return build_manual_validation_progress(merged)


def main(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    output_csv: Path = MANUAL_VALIDATION_PROGRESS_PATH,
    output_report: Path = MANUAL_VALIDATION_PROGRESS_REPORT_PATH,
) -> None:
    progress = build_progress_from_files(cohort_path, labels_path)
    report = format_progress_report(progress)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = progress.copy()
    for col in ("derived_manual_patient_ground_truth", "model_patient_positive"):
        if col in out.columns:
            out[col] = out[col].map(_empty_for_csv)
    out.to_csv(output_csv, index=False)

    output_report.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote %s (%d patients)", output_csv, len(progress))
    LOGGER.info("Wrote %s", output_report)
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
