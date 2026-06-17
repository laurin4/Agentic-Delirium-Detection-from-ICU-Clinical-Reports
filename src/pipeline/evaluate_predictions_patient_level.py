"""
Patient-level binary baseline evaluation (full corpus).

Aggregated rule: model_patient_positive = max(report klasse) per PatientenID.
Compares against ICDSC>=4, ICD10, composite OR, and composite AND baselines.

Mirrors ``evaluate_predictions`` output layout under outputs/evaluation/patient_level/.
"""

from __future__ import annotations

import os
from pathlib import Path

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.baseline_composite import (
    baseline_composite_fp_interpretation_note,
    format_baseline_composite_mode_banner,
)
from src.pipeline.evaluate_predictions import (
    BASELINE_COLUMNS,
    _binary_confusion,
    _metrics_from_counts,
)
from src.pipeline.paths import (
    EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH,
    EVALUATION_PATIENT_LEVEL_DIR,
    EVALUATION_PATIENT_LEVEL_PLOTS_DIR,
    EVALUATION_PATIENT_LEVEL_REPORT_PATH,
    EVALUATION_PATIENT_LEVEL_SUMMARY_PATH,
    EVALUATION_PATIENT_LEVEL_TABLES_DIR,
    EVALUATION_SUMMARY_PATH,
    PATIENT_VS_BASELINE_PATH,
)
from src.pipeline.predictions_source import (
    get_predictions_source,
    log_predictions_source,
    resolve_predictions_path,
)
from src.pipeline.prepare_structured_data import add_binary_baselines

_BASELINE_PLOT_LABELS = {
    "baseline_icdsc_ge_4": "ICDSC >= 4",
    "baseline_icd10": "ICD-10 delirium",
    "baseline_composite_or": "ICDSC>=4 OR ICD10",
    "baseline_composite_and": "ICDSC>=4 AND ICD10",
}


def _plot_confusion_matrix_binary(counts: dict, baseline_name: str, out_path: Path) -> None:
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    title = f"Patient-level: {_BASELINE_PLOT_LABELS.get(baseline_name, baseline_name)}"
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["pred_0", "pred_1"],
        yticklabels=["true_0", "true_1"],
        ylabel="Baseline",
        xlabel="Report text model (patient)",
        title=title,
    )
    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_distribution_comparison(df: pd.DataFrame, out_path: Path) -> None:
    labels = ["model_patient_positive"] + [
        _BASELINE_PLOT_LABELS.get(c, c) for c in BASELINE_COLUMNS
    ]
    positive_counts = [int(df["prediction_binary"].sum())]
    for col in BASELINE_COLUMNS:
        positive_counts.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).sum()))
    fig_w = max(10.0, 0.9 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(labels, positive_counts, color="#3b82f6")
    ax.set_ylabel("Positive count (class=1)")
    ax.set_title("Patient-level positive distribution: model vs primary baselines")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    pred_path = resolve_predictions_path()
    log_predictions_source(pred_path)
    print(format_baseline_composite_mode_banner())
    if not PATIENT_VS_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Patient comparison file not found: {PATIENT_VS_BASELINE_PATH}. "
            f"Run 'python -m src.pipeline.compare_patients_vs_baseline' first "
            f"(with PREDICTIONS_SOURCE={get_predictions_source()} if using cohort predictions)."
        )

    df = pd.read_csv(PATIENT_VS_BASELINE_PATH)
    if "model_patient_positive" not in df.columns:
        raise ValueError("Spalte 'model_patient_positive' fehlt.")
    if "PatientenID" not in df.columns:
        raise ValueError("Spalte 'PatientenID' fehlt.")

    df = add_binary_baselines(df.copy())
    missing_baseline_columns = [col for col in BASELINE_COLUMNS if col not in df.columns]
    if missing_baseline_columns:
        raise ValueError(
            "Missing required binary baseline columns for patient evaluation: "
            + ", ".join(missing_baseline_columns)
        )

    df["model_patient_positive"] = pd.to_numeric(df["model_patient_positive"], errors="coerce")
    df = df[df["model_patient_positive"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("Keine gueltigen binaeren Patienten-Vorhersagen gefunden (erwartet 0/1).")
    df["prediction_binary"] = df["model_patient_positive"].astype(int)

    EVALUATION_PATIENT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATIENT_LEVEL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATIENT_LEVEL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    confusion_rows = []

    for baseline_name in BASELINE_COLUMNS:
        y_true = pd.to_numeric(df[baseline_name], errors="coerce").fillna(0).astype(int)
        y_pred = df["prediction_binary"].astype(int)
        counts = _binary_confusion(y_true=y_true, y_pred=y_pred)
        metrics = _metrics_from_counts(counts)

        summary_rows.append({"baseline_name": baseline_name, **metrics})
        confusion_rows.append({"baseline_name": baseline_name, **counts})

        _plot_confusion_matrix_binary(
            counts=counts,
            baseline_name=baseline_name,
            out_path=EVALUATION_PATIENT_LEVEL_PLOTS_DIR / f"confusion_matrix_{baseline_name}.png",
        )

    summary_df = pd.DataFrame(summary_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    summary_df.to_csv(EVALUATION_PATIENT_LEVEL_SUMMARY_PATH, index=False)
    confusion_df.to_csv(EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH, index=False)
    _plot_distribution_comparison(
        df=df,
        out_path=EVALUATION_PATIENT_LEVEL_PLOTS_DIR / "class_distribution_comparison.png",
    )

    best_row = summary_df.sort_values("f1", ascending=False).iloc[0]
    report_lines = [
        "Binary baseline evaluation (patient level)",
        "",
        format_baseline_composite_mode_banner(),
        "",
        f"PREDICTIONS_SOURCE: {get_predictions_source()}",
        f"predictions_path: {pred_path}",
        f"comparison_input: {PATIENT_VS_BASELINE_PATH}",
        "",
        "aggregation_rule: model_patient_positive = max(report klasse) per PatientenID",
        "primary_baselines: ICDSC>=4, ICD10, composite OR, composite AND",
        f"n_patients: {len(df)}",
        f"best_baseline_by_f1: {best_row['baseline_name']}",
        f"best_baseline_f1: {best_row['f1']}",
        "",
        "Note: Report-level evaluation lives under outputs/evaluation/binary_baselines/.",
        "Patient-level metrics here use one row per PatientenID (no duplicate baseline rows).",
        "",
        baseline_composite_fp_interpretation_note(),
        "",
        f"summary_table: {EVALUATION_PATIENT_LEVEL_SUMMARY_PATH}",
        f"confusion_counts: {EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH}",
        f"plots_dir: {EVALUATION_PATIENT_LEVEL_PLOTS_DIR}",
    ]
    EVALUATION_PATIENT_LEVEL_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    combined_rows = [
        {"metric": "evaluation_mode", "value": "binary_baselines_patient_level"},
        {"metric": "n_patients", "value": str(len(df))},
        {"metric": "best_baseline_by_f1", "value": str(best_row["baseline_name"])},
        {"metric": "best_baseline_f1", "value": str(best_row["f1"])},
        {
            "metric": "patient_level_summary_csv",
            "value": str(EVALUATION_PATIENT_LEVEL_SUMMARY_PATH),
        },
        {
            "metric": "patient_level_confusion_csv",
            "value": str(EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH),
        },
    ]
    if EVALUATION_SUMMARY_PATH.exists():
        existing = pd.read_csv(EVALUATION_SUMMARY_PATH)
        combined = pd.concat([existing, pd.DataFrame(combined_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(combined_rows)
    combined.to_csv(EVALUATION_SUMMARY_PATH, index=False)

    print(f"Gespeichert: {EVALUATION_PATIENT_LEVEL_SUMMARY_PATH}")
    print(f"Gespeichert: {EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH}")
    print(f"Plots: {EVALUATION_PATIENT_LEVEL_PLOTS_DIR}")
    print(f"Report: {EVALUATION_PATIENT_LEVEL_REPORT_PATH}")


if __name__ == "__main__":
    main()
