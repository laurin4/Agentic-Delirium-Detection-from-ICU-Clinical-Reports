"""Metrics and derived labels for manual validation cohort evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MANUAL_REPORT_GT_COL = "manual_report_ground_truth"
DERIVED_PATIENT_GT_COL = "derived_manual_patient_ground_truth"
N_POSITIVE_REPORTS_COL = "n_positive_reports_manual"


def _series_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def derive_patient_manual_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient: ``derived_manual_patient_ground_truth`` = max(manual_report_ground_truth),
    ``n_positive_reports_manual`` = count of report positives.
    """
    out = df.copy()
    if MANUAL_REPORT_GT_COL not in out.columns:
        out[MANUAL_REPORT_GT_COL] = pd.NA
    manual = _series_binary(out[MANUAL_REPORT_GT_COL].fillna(0))
    out[N_POSITIVE_REPORTS_COL] = 0
    out[DERIVED_PATIENT_GT_COL] = 0

    if "validation_patient_id" not in out.columns:
        grouped = manual.groupby(out["PatientenID"])
        out[N_POSITIVE_REPORTS_COL] = grouped.transform("sum")
        out[DERIVED_PATIENT_GT_COL] = grouped.transform("max")
        return out

    for vpid, grp in manual.groupby(out["validation_patient_id"]):
        mask = out["validation_patient_id"] == vpid
        n_pos = int((manual.loc[mask] == 1).sum())
        derived = int(manual.loc[mask].max()) if n_pos else 0
        out.loc[mask, N_POSITIVE_REPORTS_COL] = n_pos
        out.loc[mask, DERIVED_PATIENT_GT_COL] = derived
    return out


def compute_model_patient_positive(df: pd.DataFrame) -> pd.Series:
    """Max ``model_report_prediction`` per ``validation_patient_id``."""
    key = "validation_patient_id" if "validation_patient_id" in df.columns else "PatientenID"
    pred = _series_binary(df["model_report_prediction"])
    return pred.groupby(df[key]).transform("max")


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    yt = _series_binary(y_true)
    yp = _series_binary(y_pred)
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = recall
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "n": int(total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "sensitivity": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "accuracy": round(accuracy, 6),
    }


def plot_confusion_matrix(
    counts: Dict[str, int],
    title: str,
    out_path: Path,
    *,
    ylabel: str = "Reference / manual GT",
    xlabel: str = "Model prediction",
) -> None:
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_annotated_cohort(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, str]:
    """
    Report-level and patient-level metrics; reference comparisons (ICDSC, ICD10).

    Requires non-empty ``manual_report_ground_truth`` for primary metrics (rows with valid 0/1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    work = derive_patient_manual_labels(df)
    work["model_patient_positive"] = compute_model_patient_positive(work)

    annotated = work[work[MANUAL_REPORT_GT_COL].astype(str).str.strip().isin(("0", "1"))]
    if annotated.empty:
        report = "No rows with manual_report_ground_truth in {0,1}; metrics skipped."
        return pd.DataFrame(), report

    pat = annotated.drop_duplicates(subset=["validation_patient_id"], keep="first")
    rows: List[Dict[str, Any]] = []

    # A) Report-level
    m_rep = binary_metrics(
        annotated[MANUAL_REPORT_GT_COL],
        annotated["model_report_prediction"],
    )
    rows.append({"level": "report", "comparison": "model_vs_manual_report_gt", **m_rep})
    plot_confusion_matrix(
        m_rep,
        "Report-level: model vs manual_report_ground_truth",
        plots_dir / "confusion_report_model_vs_manual.png",
        ylabel="manual_report_ground_truth",
        xlabel="model_report_prediction",
    )

    # B) Patient-level
    m_pat = binary_metrics(
        pat[DERIVED_PATIENT_GT_COL],
        pat["model_patient_positive"],
    )
    rows.append({"level": "patient", "comparison": "model_patient_vs_derived_manual_gt", **m_pat})
    plot_confusion_matrix(
        m_pat,
        "Patient-level: model_patient_positive vs derived_manual_patient_ground_truth",
        plots_dir / "confusion_patient_model_vs_derived.png",
        ylabel="derived_manual_patient_ground_truth",
        xlabel="model_patient_positive",
    )

    # Reference signals (exploratory — not absolute truth)
    for ref_col, label in (
        ("baseline_icdsc_ge_4", "ICDSC>=4 reference vs derived patient GT"),
        ("baseline_icd10", "ICD10 reference vs derived patient GT"),
    ):
        if ref_col in pat.columns:
            m_ref = binary_metrics(pat[DERIVED_PATIENT_GT_COL], pat[ref_col])
            rows.append({"level": "patient", "comparison": label, **m_ref})
            safe = ref_col.replace("baseline_", "")
            plot_confusion_matrix(
                m_ref,
                label,
                plots_dir / f"confusion_patient_{safe}_vs_derived.png",
                ylabel="derived_manual_patient_ground_truth",
                xlabel=ref_col,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "metrics_summary.csv", index=False)

    report_lines = [
        "Manual validation evaluation",
        "=" * 40,
        f"annotated_report_rows={len(annotated)}",
        f"unique_patients={annotated['validation_patient_id'].nunique()}",
        "",
        "PRIMARY thesis evaluation: patient-level (derived_manual_patient_ground_truth)",
        f"  model_patient_positive vs derived: F1={m_pat['f1']} sensitivity={m_pat['sensitivity']} specificity={m_pat['specificity']}",
        "",
        "Report-level (per-report annotation):",
        f"  model_report_prediction vs manual_report_ground_truth: F1={m_rep['f1']}",
        "",
        "ICDSC and ICD10 are reference signals only — not treated as absolute ground truth.",
        "",
        f"tables: {tables_dir}",
        f"plots: {plots_dir}",
    ]
    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary, report_path.read_text(encoding="utf-8")
