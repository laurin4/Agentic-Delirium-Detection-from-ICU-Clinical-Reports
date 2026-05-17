"""
Binary analysis summary from report_vs_baseline_comparison.csv.

LEGACY: Earlier versions of this script produced 3-class confusion matrices against
`baseline_reference_class`. That multiclass path was removed; primary evaluation
is binary (see `src.pipeline.evaluate_predictions` and baseline columns below).

This module writes supplementary tables/plots under outputs/analysis/evaluation/.

Pre-model cohort / data-coverage plots (Berichte vs structured_baseline) live in
`src.analysis.run_data_coverage_analysis` — not here.
"""

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

from src.pipeline.evaluate_predictions import BASELINE_COLUMNS, _binary_confusion, _metrics_from_counts
from src.pipeline.paths import (
    ANALYSIS_DIR,
    ANALYSIS_EVALUATION_PLOTS_DIR,
    ANALYSIS_EVALUATION_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)
from src.pipeline.prepare_structured_data import add_binary_baselines

REPORT_PATH = ANALYSIS_DIR / "evaluation" / "report.txt"

# Primary baseline for a single highlighted confusion plot (change if needed).
PRIMARY_BASELINE_FOR_PLOT = "baseline_icdsc_ge_4"


def _load_main_df() -> pd.DataFrame:
    if not REPORT_VS_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {REPORT_VS_BASELINE_PATH}. "
            "Run compare_reports_vs_baseline after run_pipeline."
        )
    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    if "klasse" in df.columns:
        df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    if "anzahl_treffer" in df.columns:
        df["anzahl_treffer"] = pd.to_numeric(df["anzahl_treffer"], errors="coerce")
    return df


def _plot_binary_confusion(counts: dict, baseline_name: str, out_path: Path) -> None:
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["pred_0", "pred_1"],
        yticklabels=["true_0", "true_1"],
        ylabel=f"Baseline ({baseline_name})",
        xlabel="Report text model",
        title=f"Binary confusion: {baseline_name}",
    )
    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_text_summary(n_patients: int, summary_df: pd.DataFrame) -> None:
    lines = [
        "Binary analysis summary (supplementary)",
        "",
        f"Source: {REPORT_VS_BASELINE_PATH}",
        f"Patients (binary klasse 0/1): {n_patients}",
        "",
        "Primary evaluation outputs live under outputs/evaluation/binary_baselines/",
        "(run: python -m src.pipeline.evaluate_predictions)",
        "",
        "Per-baseline metrics (y_true=baseline, y_pred=model klasse):",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"  {row['baseline_name']}: acc={row['accuracy']} prec={row['precision']} "
            f"rec={row['recall']} f1={row['f1']}"
        )
    lines.extend(["", f"Tables: {ANALYSIS_EVALUATION_TABLES_DIR}", f"Plots: {ANALYSIS_EVALUATION_PLOTS_DIR}"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_EVALUATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_main_df()
    if "PatientenID" not in df.columns:
        raise ValueError("Expected column 'PatientenID' in comparison CSV.")
    df = add_binary_baselines(df.copy())
    missing = [c for c in BASELINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing baseline columns after add_binary_baselines: " + ", ".join(missing))

    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    df = df[df["klasse"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No rows with binary klasse in {0, 1}.")

    df["prediction_binary"] = df["klasse"].astype(int)

    summary_rows = []
    confusion_rows = []
    for baseline_name in BASELINE_COLUMNS:
        y_true = pd.to_numeric(df[baseline_name], errors="coerce").fillna(0).astype(int)
        y_pred = df["prediction_binary"].astype(int)
        counts = _binary_confusion(y_true=y_true, y_pred=y_pred)
        metrics = _metrics_from_counts(counts)
        summary_rows.append({"baseline_name": baseline_name, **metrics})
        confusion_rows.append({"baseline_name": baseline_name, **counts})

    summary_df = pd.DataFrame(summary_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    summary_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "binary_metrics_by_baseline.csv", index=False)
    confusion_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "binary_confusion_counts.csv", index=False)

    if PRIMARY_BASELINE_FOR_PLOT in BASELINE_COLUMNS:
        row = confusion_df.loc[confusion_df["baseline_name"] == PRIMARY_BASELINE_FOR_PLOT].iloc[0]
        cdict = {k: int(row[k]) for k in ("tp", "tn", "fp", "fn")}
        _plot_binary_confusion(
            cdict,
            PRIMARY_BASELINE_FOR_PLOT,
            ANALYSIS_EVALUATION_PLOTS_DIR / f"binary_confusion_{PRIMARY_BASELINE_FOR_PLOT}.png",
        )

    _write_text_summary(len(df), summary_df)

    print(f"Binary analysis tables: {ANALYSIS_EVALUATION_TABLES_DIR}")
    print(f"Binary analysis plots:  {ANALYSIS_EVALUATION_PLOTS_DIR}")
    print(f"Binary analysis report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
