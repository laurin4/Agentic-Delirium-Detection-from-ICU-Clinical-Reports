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

from src.pipeline.paths import (
    EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH,
    EVALUATION_BINARY_BASELINE_REPORT_PATH,
    EVALUATION_BINARY_BASELINE_SUMMARY_PATH,
    EVALUATION_BINARY_BASELINES_DIR,
    EVALUATION_BINARY_BASELINES_PLOTS_DIR,
    EVALUATION_BINARY_BASELINES_TABLES_DIR,
    EVALUATION_SUMMARY_PATH,
    REPORT_VS_BASELINE_PATH,
)
from src.pipeline.prepare_structured_data import add_binary_baselines


BASELINE_COLUMNS = [
    "baseline_icdsc_ge_1",
    "baseline_icdsc_ge_2",
    "baseline_icdsc_ge_3",
    "baseline_icdsc_ge_4",
    "baseline_icdsc_ge_5",
    "baseline_icd10",
    "baseline_icdsc_0",
    "baseline_icdsc_1_to_3",
    "baseline_icdsc_ge_4_grouped",
]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_confusion(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_counts(counts: dict) -> dict:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, total)
    return {
        "n_patients": int(total),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_tn": tn,
        "confusion_fp": fp,
        "confusion_fn": fn,
        "confusion_tp": tp,
        "prediction_positive_count": int(tp + fp),
        "baseline_positive_count": int(tp + fn),
    }


def _plot_confusion_matrix_binary(counts: dict, baseline_name: str, out_path: Path) -> None:
    cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["pred_0", "pred_1"],
        yticklabels=["true_0", "true_1"],
        ylabel="Baseline",
        xlabel="Report text model",
        title=f"Confusion: {baseline_name}",
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
    labels = ["report_text_model"] + BASELINE_COLUMNS
    positive_counts = [int(df["prediction_binary"].sum())]
    for col in BASELINE_COLUMNS:
        positive_counts.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).sum()))
    fig_w = max(14.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(labels, positive_counts, color="#3b82f6")
    ax.set_ylabel("Positive count (class=1)")
    ax.set_title("Positive class distribution: report text model vs baselines")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not REPORT_VS_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {REPORT_VS_BASELINE_PATH}. "
            "Run 'python -m src.pipeline.compare_reports_vs_baseline' first."
        )

    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    if "klasse" not in df.columns:
        raise ValueError("Spalte 'klasse' fehlt.")
    if "PatientenID" not in df.columns:
        raise ValueError("Spalte 'PatientenID' fehlt.")

    df = add_binary_baselines(df.copy())
    missing_baseline_columns = [col for col in BASELINE_COLUMNS if col not in df.columns]
    if missing_baseline_columns:
        raise ValueError(
            "Missing required binary baseline columns for evaluation: "
            + ", ".join(missing_baseline_columns)
        )
    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    df = df[df["klasse"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("Keine gueltigen binaeren Vorhersagen in 'klasse' gefunden (erwartet 0/1).")
    df["prediction_binary"] = df["klasse"].astype(int)

    EVALUATION_BINARY_BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_BINARY_BASELINES_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_BINARY_BASELINES_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
            out_path=EVALUATION_BINARY_BASELINES_PLOTS_DIR / f"confusion_matrix_{baseline_name}.png",
        )

    summary_df = pd.DataFrame(summary_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    summary_df.to_csv(EVALUATION_BINARY_BASELINE_SUMMARY_PATH, index=False)
    confusion_df.to_csv(EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH, index=False)
    _plot_distribution_comparison(
        df=df,
        out_path=EVALUATION_BINARY_BASELINES_PLOTS_DIR / "class_distribution_comparison.png",
    )

    icd10_vs_icdsc_rows = []
    y_icd10 = pd.to_numeric(df["baseline_icd10"], errors="coerce").fillna(0).astype(int)
    for threshold in [1, 2, 3, 4, 5]:
        base_name = f"baseline_icdsc_ge_{threshold}"
        y_icdsc = pd.to_numeric(df[base_name], errors="coerce").fillna(0).astype(int)
        counts = _binary_confusion(y_true=y_icdsc, y_pred=y_icd10)
        icd10_vs_icdsc_rows.append({"baseline_name": f"icd10_vs_{base_name}", **_metrics_from_counts(counts)})
    pd.DataFrame(icd10_vs_icdsc_rows).to_csv(
        EVALUATION_BINARY_BASELINES_TABLES_DIR / "icd10_vs_icdsc_thresholds.csv",
        index=False,
    )

    best_row = summary_df.sort_values("f1", ascending=False).iloc[0]
    report_lines = [
        "Binary baseline evaluation",
        "",
        f"n_patients: {len(df)}",
        f"best_baseline_by_f1: {best_row['baseline_name']}",
        f"best_baseline_f1: {best_row['f1']}",
        "",
        f"summary_table: {EVALUATION_BINARY_BASELINE_SUMMARY_PATH}",
        f"confusion_counts: {EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH}",
        f"plots_dir: {EVALUATION_BINARY_BASELINES_PLOTS_DIR}",
    ]
    EVALUATION_BINARY_BASELINE_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    combined_rows = [
        {"metric": "evaluation_mode", "value": "binary_baselines_primary"},
        {"metric": "n_patients", "value": str(len(df))},
        {"metric": "best_baseline_by_f1", "value": str(best_row["baseline_name"])},
        {"metric": "best_baseline_f1", "value": str(best_row["f1"])},
        {"metric": "binary_baseline_summary_csv", "value": str(EVALUATION_BINARY_BASELINE_SUMMARY_PATH)},
        {"metric": "binary_baseline_confusion_csv", "value": str(EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH)},
    ]
    pd.DataFrame(combined_rows).to_csv(EVALUATION_SUMMARY_PATH, index=False)

    print(f"Gespeichert: {EVALUATION_BINARY_BASELINE_SUMMARY_PATH}")
    print(f"Gespeichert: {EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH}")
    print(f"Plots: {EVALUATION_BINARY_BASELINES_PLOTS_DIR}")
    print(f"Report: {EVALUATION_BINARY_BASELINE_REPORT_PATH}")


if __name__ == "__main__":
    main()
