"""
Final patient-level manual validation evaluation (frozen cohort).

Primary metrics use ONLY patients with all reports manually labeled (0/1).
Empty manual labels are never treated as 0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.build_manual_validation_progress import (
    _parse_report_gt,
    _patient_model_positive,
    assign_confusion_group,
)
from src.analysis.manual_report_labels import merge_manual_report_labels
from src.analysis.manual_validation_eval import plot_confusion_matrix
from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
)

LOGGER = logging.getLogger(__name__)

MANUAL_GT_COL = "derived_manual_patient_ground_truth"

PATIENT_GT_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "PatientenID",
    "n_reports_total",
    "n_reports_labeled",
    "n_reports_missing_label",
    "is_patient_complete",
    "n_positive_reports_manual",
    MANUAL_GT_COL,
    "model_patient_positive",
    "baseline_icdsc_ge_4",
    "baseline_icd10",
    "baseline_composite_or",
    "baseline_composite_and",
    "manual_comments_summary",
    "representative_evidence",
)

EVALUATION_METHODS: tuple[tuple[str, str, str], ...] = (
    ("model_patient_positive", "model", "confusion_matrix_model_vs_manual.png"),
    ("baseline_icdsc_ge_4", "icdsc", "confusion_matrix_icdsc_vs_manual.png"),
    ("baseline_icd10", "icd10", "confusion_matrix_icd10_vs_manual.png"),
    (
        "baseline_composite_or",
        "composite_or",
        "confusion_matrix_composite_or_vs_manual.png",
    ),
    (
        "baseline_composite_and",
        "composite_and",
        "confusion_matrix_composite_and_vs_manual.png",
    ),
)

ERROR_EXPORT_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "PatientenID",
    MANUAL_GT_COL,
    "model_patient_positive",
    "baseline_icdsc_ge_4",
    "baseline_icd10",
    "baseline_composite_or",
    "baseline_composite_and",
    "n_reports_total",
    "n_positive_reports_manual",
    "manual_comments_summary",
    "representative_evidence",
)


def _binary_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def derive_composite_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``baseline_composite_or`` / ``baseline_composite_and`` when missing."""
    out = df.copy()
    icdsc = pd.to_numeric(out.get("baseline_icdsc_ge_4"), errors="coerce")
    icd10 = pd.to_numeric(out.get("baseline_icd10"), errors="coerce")

    if "baseline_composite_or" not in out.columns:
        out["baseline_composite_or"] = pd.NA
    if "baseline_composite_and" not in out.columns:
        out["baseline_composite_and"] = pd.NA

    or_vals = pd.to_numeric(out["baseline_composite_or"], errors="coerce")
    and_vals = pd.to_numeric(out["baseline_composite_and"], errors="coerce")
    derived_or = pd.concat([icdsc, icd10], axis=1).max(axis=1, skipna=True)
    derived_and = pd.concat([icdsc, icd10], axis=1).min(axis=1, skipna=True)

    out["baseline_composite_or"] = or_vals.where(or_vals.notna(), derived_or)
    out["baseline_composite_and"] = and_vals.where(and_vals.notna(), derived_and)
    return out


def _aggregate_manual_comments(grp: pd.DataFrame) -> str:
    if "manual_comment" not in grp.columns:
        return ""
    comments = grp["manual_comment"].dropna().astype(str).str.strip()
    comments = comments[comments != ""]
    if comments.empty:
        return ""
    return " | ".join(dict.fromkeys(comments.tolist()))


def _representative_evidence(grp: pd.DataFrame, gt_parsed: pd.Series) -> str:
    if "evidence_snippets" not in grp.columns:
        return ""
    pos_mask = gt_parsed == "1"
    if pos_mask.any():
        ev = grp.loc[pos_mask, "evidence_snippets"].dropna().astype(str).str.strip()
        ev = ev[ev != ""]
        if not ev.empty:
            return ev.iloc[0]
    ev = grp["evidence_snippets"].dropna().astype(str).str.strip()
    ev = ev[ev != ""]
    return ev.iloc[0] if not ev.empty else ""


def _patient_baseline_row(grp: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in (
        "baseline_icdsc_ge_4",
        "baseline_icd10",
        "baseline_composite_or",
        "baseline_composite_and",
    ):
        if col in grp.columns:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[col] = int(vals.iloc[0]) if len(vals) else pd.NA
        else:
            row[col] = pd.NA
    return row


def build_patient_level_ground_truth(merged_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``validation_patient_id`` with manual GT and baseline signals.

    Incomplete patients retain empty ``derived_manual_patient_ground_truth``.
    """
    if "validation_patient_id" not in merged_cohort.columns:
        raise ValueError("cohort must contain validation_patient_id")

    df = merged_cohort.copy()
    gt_col = (
        df["manual_report_ground_truth"]
        if "manual_report_ground_truth" in df.columns
        else pd.Series(index=df.index, dtype=object)
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
        patienten_id = ""
        if "PatientenID" in grp.columns:
            pid_vals = grp["PatientenID"].dropna()
            if len(pid_vals):
                patienten_id = str(pid_vals.iloc[0])

        row = {
            "validation_patient_id": vpid,
            "PatientenID": patienten_id,
            "n_reports_total": n_total,
            "n_reports_labeled": n_labeled,
            "n_reports_missing_label": n_missing,
            "is_patient_complete": int(is_complete),
            "n_positive_reports_manual": n_pos,
            MANUAL_GT_COL: derived,
            "model_patient_positive": model_pos if model_pos is not None else pd.NA,
            "manual_comments_summary": _aggregate_manual_comments(grp),
            "representative_evidence": _representative_evidence(grp, grp["_gt"]),
        }
        row.update(_patient_baseline_row(grp))
        rows.append(row)

    out = pd.DataFrame(rows)
    out = derive_composite_baselines(out)
    for col in PATIENT_GT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(PATIENT_GT_COLUMNS)]


def primary_evaluation_cohort(patient_gt: pd.DataFrame) -> pd.DataFrame:
    """Complete patients only — primary thesis evaluation set."""
    return patient_gt[patient_gt["is_patient_complete"] == 1].copy()


def compute_method_metrics(
    manual: pd.Series,
    predicted: pd.Series,
    *,
    method_name: str,
) -> Dict[str, Any]:
    """
    Confusion vs manual GT (positive class = manual delirium present).

    TP: pred=1 & manual=1; FP: pred=1 & manual=0; TN: pred=0 & manual=0; FN: pred=0 & manual=1
    """
    y_true = _binary_series(manual)
    y_pred = _binary_series(predicted)
    valid = y_true.notna() & y_pred.notna() & y_true.isin([0, 1]) & y_pred.isin([0, 1])
    yt = y_true.loc[valid].astype(int)
    yp = y_pred.loc[valid].astype(int)

    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    total = tp + tn + fp + fn

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    recall = sensitivity
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "method": method_name,
        "n_patients": int(total),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": round(sensitivity, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "precision": round(precision, 6),
        "ppv": round(precision, 6),
        "npv": round(npv, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def evaluate_all_methods(complete: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics and confusion counts for each comparison method."""
    if complete.empty:
        empty_metrics = pd.DataFrame(
            columns=[
                "method",
                "n_patients",
                "tp",
                "fp",
                "tn",
                "fn",
                "sensitivity",
                "recall",
                "specificity",
                "precision",
                "ppv",
                "npv",
                "f1",
                "accuracy",
            ]
        )
        empty_conf = pd.DataFrame(columns=["method", "tp", "fp", "tn", "fn"])
        return empty_metrics, empty_conf

    manual = complete[MANUAL_GT_COL]
    metric_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    for pred_col, method_key, _plot in EVALUATION_METHODS:
        if pred_col not in complete.columns:
            LOGGER.warning("Skipping method %s: column %s missing", method_key, pred_col)
            continue
        m = compute_method_metrics(manual, complete[pred_col], method_name=method_key)
        metric_rows.append(m)
        confusion_rows.append(
            {
                "method": method_key,
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def write_confusion_plots(
    complete: pd.DataFrame,
    confusion_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    manual = complete[MANUAL_GT_COL]

    for pred_col, method_key, plot_name in EVALUATION_METHODS:
        if pred_col not in complete.columns:
            continue
        row = confusion_df[confusion_df["method"] == method_key]
        if row.empty:
            m = compute_method_metrics(manual, complete[pred_col], method_name=method_key)
            counts = {"tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]}
        else:
            r = row.iloc[0]
            counts = {"tp": int(r["tp"]), "fp": int(r["fp"]), "tn": int(r["tn"]), "fn": int(r["fn"])}

        titles = {
            "model": "Model vs manual patient GT",
            "icdsc": "ICDSC>=4 vs manual patient GT",
            "icd10": "ICD10 vs manual patient GT",
            "composite_or": "Composite OR vs manual patient GT",
            "composite_and": "Composite AND vs manual patient GT",
        }
        plot_confusion_matrix(
            counts,
            titles.get(method_key, method_key),
            plots_dir / plot_name,
            ylabel="Manual GT (reference)",
            xlabel=pred_col,
        )


def _error_export_row(row: pd.Series) -> Dict[str, Any]:
    return {col: row.get(col, "") for col in ERROR_EXPORT_COLUMNS}


def export_model_error_slices(complete: pd.DataFrame, output_dir: Path) -> None:
    """Write model_TP/FP/TN/FN.csv for complete patients."""
    work = complete.copy()
    work["model_confusion_group"] = work.apply(
        lambda r: assign_confusion_group(
            r.get("model_patient_positive"),
            r.get(MANUAL_GT_COL),
        ),
        axis=1,
    )
    for label in ("TP", "FP", "TN", "FN"):
        subset = work[work["model_confusion_group"] == label]
        rows = [_error_export_row(subset.loc[idx]) for idx in subset.index]
        out = pd.DataFrame(rows, columns=list(ERROR_EXPORT_COLUMNS))
        out.to_csv(output_dir / f"model_{label}.csv", index=False)


def format_final_report(
    patient_gt: pd.DataFrame,
    complete: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    incomplete_patient_ids: Sequence[str],
) -> str:
    n_total = len(patient_gt)
    n_complete = len(complete)
    n_incomplete = n_total - n_complete

    manual = _binary_series(complete[MANUAL_GT_COL]) if n_complete else pd.Series(dtype="Int64")
    n_manual_pos = int((manual == 1).sum()) if n_complete else 0
    n_manual_neg = int((manual == 0).sum()) if n_complete else 0

    lines = [
        "Final manual validation evaluation",
        "=" * 44,
        "",
        "Cohort counts",
        "-" * 44,
        f"total_frozen_patients={n_total}",
        f"complete_patients={n_complete}",
        f"incomplete_patients={n_incomplete}",
        f"manual_positive_patients={n_manual_pos}",
        f"manual_negative_patients={n_manual_neg}",
        "",
        "WARNING: Incomplete patients are EXCLUDED from primary evaluation.",
        "Empty manual labels are NOT treated as 0.",
        "",
    ]
    if incomplete_patient_ids:
        preview = list(incomplete_patient_ids)[:20]
        lines.append(f"incomplete_patient_ids (first {len(preview)}): {preview}")
        lines.append("")

    lines.extend(["Primary evaluation metrics (complete patients only)", "-" * 44, ""])
    if metrics.empty:
        lines.append("No metrics computed (no complete patients or missing columns).")
    else:
        for _, row in metrics.iterrows():
            lines.append(
                f"{row['method']}: n={row['n_patients']} "
                f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']} "
                f"sens={row['sensitivity']} spec={row['specificity']} "
                f"PPV={row['ppv']} NPV={row['npv']} F1={row['f1']} acc={row['accuracy']}"
            )

    lines.extend(
        [
            "",
            "Interpretation",
            "-" * 44,
            "- Manual report labels aggregated to patient level: any report positive => patient positive.",
            "- Model and baseline signals (ICDSC, ICD10, composite OR/AND) are compared against",
            "  derived manual patient ground truth as the primary reference for this validation.",
            "- ICDSC and ICD10 are structured reference signals; composite OR/AND are exploratory.",
            "- Review model_FP.csv and model_FN.csv for qualitative error analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_final_evaluation(
    merged_cohort: pd.DataFrame,
    output_dir: Path = FINAL_MANUAL_VALIDATION_EVAL_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Build patient GT, evaluate complete patients, write all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    patient_gt = build_patient_level_ground_truth(merged_cohort)
    patient_gt.to_csv(output_dir / "patient_level_ground_truth.csv", index=False)

    complete = primary_evaluation_cohort(patient_gt)
    incomplete_ids = patient_gt.loc[
        patient_gt["is_patient_complete"] == 0, "validation_patient_id"
    ].astype(str).tolist()

    metrics, confusion = evaluate_all_methods(complete)
    metrics.to_csv(output_dir / "final_metrics_summary.csv", index=False)
    confusion.to_csv(output_dir / "confusion_counts.csv", index=False)

    write_confusion_plots(complete, confusion, plots_dir)
    export_model_error_slices(complete, output_dir)

    report = format_final_report(
        patient_gt,
        complete,
        metrics,
        incomplete_patient_ids=incomplete_ids,
    )
    (output_dir / "report.txt").write_text(report, encoding="utf-8")

    LOGGER.info(
        "Final evaluation: %d total patients, %d complete, %d incomplete",
        len(patient_gt),
        len(complete),
        len(incomplete_ids),
    )
    return patient_gt, metrics, confusion, report


def load_merged_frozen_cohort(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
) -> pd.DataFrame:
    if not cohort_path.exists():
        raise FileNotFoundError(f"Frozen patient validation cohort missing: {cohort_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Frozen manual report labels missing: {labels_path}")
    cohort = pd.read_csv(cohort_path)
    labels = pd.read_csv(labels_path)
    return merge_manual_report_labels(cohort, labels, log_context="final manual validation")


def main(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    output_dir: Path = FINAL_MANUAL_VALIDATION_EVAL_DIR,
) -> None:
    merged = load_merged_frozen_cohort(cohort_path, labels_path)
    _, metrics, _, report = run_final_evaluation(merged, output_dir=output_dir)
    print(report)
    if not metrics.empty:
        print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
