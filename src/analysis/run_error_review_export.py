"""
Export false-positive and false-negative case lists per structured baseline binary column.

Reads: outputs/comparisons/report_vs_baseline_comparison.csv

Does not change prediction or baseline logic — analysis / review tooling only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    ERROR_REVIEW_DIR,
    ERROR_REVIEW_PLOTS_DIR,
    ERROR_REVIEW_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)

LOGGER = logging.getLogger(__name__)

BASELINE_COLUMNS: Sequence[str] = (
    "baseline_icdsc_ge_1",
    "baseline_icdsc_ge_2",
    "baseline_icdsc_ge_3",
    "baseline_icdsc_ge_4",
    "baseline_icdsc_ge_5",
    "baseline_icdsc_0",
    "baseline_icdsc_1_to_3",
    "baseline_icdsc_ge_4_grouped",
    "baseline_icd10",
)

# Prefer these columns when present on the merged comparison file
REVIEW_EXPORT_COLUMNS = [
    "PatientenID",
    "bericht",
    "klasse",
    "klassifikation",
    "signalstaerke",
    "anzahl_treffer",
    "delir_signale",
    "kontext",
    "begruendung",
    "has_delir_icd10",
    "max_icdsc",
]

BASELINE_COLUMNS_TO_INCLUDE = list(BASELINE_COLUMNS) + ["prediction_binary"]

OPTIONAL_COLUMNS = [
    "llm_text_reduction_method",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_skipped_by_prefilter",
    "alternative_erklaerung",
    "alternative_erklaerung_keywords",
    "klassifikation_begruendung",
]


def _mpl_cfg() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def safe_baseline_filename(baseline_col: str) -> str:
    return baseline_col.replace("/", "_")


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, int]:
    """Binary 0/1 truth and predictions."""
    t = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).clip(0, 1)
    p = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int).clip(0, 1)
    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    if precision != precision or recall != recall:
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "n_evaluable": float(n),
    }


def select_export_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in REVIEW_EXPORT_COLUMNS + BASELINE_COLUMNS_TO_INCLUDE:
        if c in df.columns and c not in cols:
            cols.append(c)
    for baseline in BASELINE_COLUMNS:
        if baseline in df.columns and baseline not in cols:
            cols.append(baseline)
    for c in OPTIONAL_COLUMNS:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def run_error_review(
    cmp_path: Path = REPORT_VS_BASELINE_PATH,
    out_dir: Path = ERROR_REVIEW_DIR,
    tables_dir: Path = ERROR_REVIEW_TABLES_DIR,
    plots_dir: Path = ERROR_REVIEW_PLOTS_DIR,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg()))
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cmp_path.exists():
        raise FileNotFoundError(f"Comparison file missing: {cmp_path}. Run compare_reports_vs_baseline first.")

    df = pd.read_csv(cmp_path)
    if "PatientenID" not in df.columns or "klasse" not in df.columns:
        raise ValueError(f"Expected PatientenID and klasse columns. Found: {list(df.columns)}")

    df = df.copy()
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    export_cols = select_export_columns(df)
    summary_rows: List[Dict[str, object]] = []

    fp_totals = []
    fn_totals = []
    labels = []

    for baseline_col in BASELINE_COLUMNS:
        if baseline_col not in df.columns:
            LOGGER.warning("Skipping missing baseline column: %s", baseline_col)
            continue

        y_true = pd.to_numeric(df[baseline_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        y_pred = pd.to_numeric(df["klasse"], errors="coerce").fillna(0).astype(int).clip(0, 1)

        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        cm = confusion_counts(y_true, y_pred)
        mets = binary_metrics(cm)
        summary_rows.append(
            {
                "baseline": baseline_col,
                **cm,
                **mets,
            }
        )

        tag = safe_baseline_filename(baseline_col)
        df.loc[fp_mask, export_cols].to_csv(tables_dir / f"false_positives_{tag}.csv", index=False)
        df.loc[fn_mask, export_cols].to_csv(tables_dir / f"false_negatives_{tag}.csv", index=False)

        fp_totals.append(cm["fp"])
        fn_totals.append(cm["fn"])
        labels.append(baseline_col.replace("baseline_", ""))

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(tables_dir / "error_review_summary.csv", index=False)

    if labels:
        fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.35 * len(labels))))
        y = np.arange(len(labels))
        w = 0.35
        ax.barh(y - w / 2, fp_totals, height=w, label="False positives (pred=1, baseline=0)", color="#ea580c")
        ax.barh(y + w / 2, fn_totals, height=w, label="False negatives (pred=0, baseline=1)", color="#64748b")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title("Error counts by baseline definition\nClinical review advised for FP/FN exports")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(plots_dir / "error_counts_fp_fn_by_baseline.png", dpi=120)
        plt.close(fig)

    report_path = out_dir / "report.txt"
    lines = [
        "Error review export",
        "",
        f"Source: {cmp_path}",
        f"Tables: {tables_dir}",
        f"Plots: {plots_dir}",
        "",
        "Definitions:",
        "  FP: model klasse == 1 and baseline == 0.",
        "  FN: model klasse == 0 and baseline == 1.",
        "",
        "Interpretation:",
        "  FP may indicate model overcalling or undercoding / timing mismatch in structured baselines.",
        "  FN cases deserve manual chart review for documentation of delirium the model missed.",
        "  Structured ICD/ICDSC-derived labels are imperfect clinical references, not oracle diagnoses.",
        "",
        "Per-baseline CSVs:",
        "  false_positives_<baseline>.csv | false_negatives_<baseline>.csv",
        "",
        f"n_rows in comparison file: {len(df)}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Error review summary: {tables_dir / 'error_review_summary.csv'}")
    print(f"Report: {report_path}")


def main() -> None:
    run_error_review()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
