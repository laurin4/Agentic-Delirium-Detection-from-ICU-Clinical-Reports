import os
from pathlib import Path
from typing import Dict, List

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    ANALYSIS_DIR,
    ANALYSIS_EVALUATION_PLOTS_DIR,
    ANALYSIS_EVALUATION_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)

CLASS_LABELS = [0, 1, 2]
REPORT_PATH = ANALYSIS_DIR / "evaluation" / "report.txt"


def _load_main_df() -> pd.DataFrame:
    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    for col in ["klasse", "baseline_reference_class", "anzahl_treffer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cm = pd.crosstab(df["baseline_reference_class"], df["klasse"], dropna=False)
    return cm.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0).astype(int)


def _classification_report(cm: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for cls in CLASS_LABELS:
        tp = int(cm.loc[cls, cls])
        fp = int(cm[cls].sum() - tp)
        fn = int(cm.loc[cls].sum() - tp)
        support = int(cm.loc[cls].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "class": cls,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def _class_distribution_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pred = df["klasse"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    ref = df["baseline_reference_class"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    rows = []
    for cls in CLASS_LABELS:
        rows.append(
            {
                "class": cls,
                "prediction_count": int(pred.loc[cls]),
                "baseline_count": int(ref.loc[cls]),
                "prediction_share": round(_safe_div(pred.loc[cls], len(df)), 6),
                "baseline_share": round(_safe_div(ref.loc[cls], len(df)), 6),
            }
        )
    return pd.DataFrame(rows)


def _ordinal_error_statistics(df: pd.DataFrame) -> pd.DataFrame:
    diff = pd.to_numeric(df["klasse"], errors="coerce") - pd.to_numeric(df["baseline_reference_class"], errors="coerce")
    abs_diff = diff.abs()
    return pd.DataFrame(
        [
            {"metric": "mean_absolute_error", "value": round(float(abs_diff.mean()), 6)},
            {"metric": "overprediction_count", "value": int((diff > 0).sum())},
            {"metric": "underprediction_count", "value": int((diff < 0).sum())},
            {"metric": "exact_match_count", "value": int((diff == 0).sum())},
            {"metric": "total_rows", "value": int(diff.notna().sum())},
        ]
    )


def _plot_confusion_heatmap(cm: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Baseline class")
    ax.set_title("Confusion Matrix (3-Class)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(ANALYSIS_EVALUATION_PLOTS_DIR / "01_confusion_matrix_heatmap.png", dpi=300)
    plt.close(fig)


def _plot_distribution_comparison(dist_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(CLASS_LABELS))
    width = 0.35
    ax.bar(x - width / 2, dist_df["prediction_count"], width=width, label="Prediction", color="#2E86AB")
    ax.bar(x + width / 2, dist_df["baseline_count"], width=width, label="Baseline", color="#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLASS_LABELS])
    ax.set_title("Predicted vs Baseline Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of patients")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_EVALUATION_PLOTS_DIR / "02_predicted_vs_baseline_distribution.png", dpi=300)
    plt.close(fig)


def _plot_error_distribution(df: pd.DataFrame) -> pd.DataFrame:
    diff = pd.to_numeric(df["klasse"], errors="coerce") - pd.to_numeric(df["baseline_reference_class"], errors="coerce")
    counts = pd.DataFrame(
        [
            {"error_type": "false_positive_overprediction", "count": int((diff > 0).sum())},
            {"error_type": "false_negative_underprediction", "count": int((diff < 0).sum())},
            {"error_type": "exact_match", "count": int((diff == 0).sum())},
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts["error_type"], counts["count"], color=["#E67E22", "#5DADE2", "#58D68D"])
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_EVALUATION_PLOTS_DIR / "03_error_distribution.png", dpi=300)
    plt.close(fig)
    return counts


def _write_text_summary(
    main_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    ordinal_df: pd.DataFrame,
    dist_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Thesis-Level Model Evaluation Report")
    lines.append("")
    lines.append(f"Evaluated rows: {len(main_df)}")
    if "PatientenID" in main_df.columns:
        lines.append(f"Unique patients: {main_df['PatientenID'].nunique()}")

    lines.append("")
    lines.append("Key statistics")
    for _, row in ordinal_df.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.append("")
    lines.append("Main findings")
    best = classification_df.sort_values("f1_score", ascending=False).head(1)
    worst = classification_df.sort_values("f1_score", ascending=True).head(1)
    if not best.empty and not worst.empty:
        lines.append(f"- Best-performing class by F1: {int(best.iloc[0]['class'])} ({best.iloc[0]['f1_score']})")
        lines.append(f"- Weakest class by F1: {int(worst.iloc[0]['class'])} ({worst.iloc[0]['f1_score']})")

    drift = dist_df.assign(abs_gap=(dist_df["prediction_share"] - dist_df["baseline_share"]).abs())
    if not drift.empty:
        top_drift = drift.sort_values("abs_gap", ascending=False).iloc[0]
        lines.append(
            f"- Largest class share gap: class {int(top_drift['class'])} "
            f"(prediction_share={top_drift['prediction_share']}, baseline_share={top_drift['baseline_share']})"
        )

    lines.append("")
    lines.append("Potential issues in model behavior")
    over = ordinal_df.loc[ordinal_df["metric"] == "overprediction_count", "value"]
    under = ordinal_df.loc[ordinal_df["metric"] == "underprediction_count", "value"]
    if len(over) and len(under):
        if float(over.iloc[0]) > float(under.iloc[0]):
            lines.append("- The model tends to overpredict severity classes.")
        elif float(under.iloc[0]) > float(over.iloc[0]):
            lines.append("- The model tends to underpredict severity classes.")
        else:
            lines.append("- Overprediction and underprediction occur at similar rates.")
    lines.append("- Inspect confusion hotspots between neighboring ordinal classes.")

    lines.append("")
    lines.append("Artifacts")
    lines.append(f"- tables: {ANALYSIS_EVALUATION_TABLES_DIR}")
    lines.append(f"- plots: {ANALYSIS_EVALUATION_PLOTS_DIR}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_EVALUATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    main_df = _load_main_df()
    if "klasse" not in main_df.columns or "baseline_reference_class" not in main_df.columns:
        raise ValueError("Evaluation benötigt die Spalten 'klasse' und 'baseline_reference_class'.")

    cm = _confusion_matrix(main_df)
    cm.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "confusion_matrix_3x3.csv")

    class_report_df = _classification_report(cm)
    class_report_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "classification_report.csv", index=False)

    dist_df = _class_distribution_comparison(main_df)
    dist_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "class_distribution_comparison.csv", index=False)

    ordinal_df = _ordinal_error_statistics(main_df)
    ordinal_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "ordinal_error_statistics.csv", index=False)

    error_dist_df = _plot_error_distribution(main_df)
    error_dist_df.to_csv(ANALYSIS_EVALUATION_TABLES_DIR / "error_distribution_counts.csv", index=False)

    _plot_confusion_heatmap(cm)
    _plot_distribution_comparison(dist_df)
    _write_text_summary(main_df, class_report_df, ordinal_df, dist_df)

    print(f"Evaluation tables: {ANALYSIS_EVALUATION_TABLES_DIR}")
    print(f"Evaluation plots:  {ANALYSIS_EVALUATION_PLOTS_DIR}")
    print(f"Evaluation report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
