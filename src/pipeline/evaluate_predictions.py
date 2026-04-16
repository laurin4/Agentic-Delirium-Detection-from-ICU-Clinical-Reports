import logging
import os
from pathlib import Path
from typing import Dict

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    EVALUATION_CONFUSION_3CLASS_PATH,
    EVALUATION_DIR,
    EVALUATION_MULTICLASS_SUMMARY_PATH,
    EVALUATION_SUMMARY_PATH,
    REPORT_VS_BASELINE_PATH,
)
from src.pipeline.prepare_structured_data import add_reference_class

LOGGER = logging.getLogger(__name__)

CLASS_LABELS = [0, 1, 2]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def _ensure_reference_class(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline_reference_class" in df.columns and df["baseline_reference_class"].notna().all():
        return df
    needed = {"has_delir_icd10", "any_delir_flag", "max_icdsc"}
    if not needed.issubset(df.columns):
        raise ValueError(
            "Merged evaluation data must contain 'baseline_reference_class' or columns: "
            + ", ".join(sorted(needed))
        )
    LOGGER.info("Computing baseline_reference_class via add_reference_class.")
    return add_reference_class(df.copy())


def _multiclass_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = reference (baseline), columns = prediction (klasse)."""
    y_true = pd.to_numeric(df["baseline_reference_class"], errors="coerce").fillna(-1).astype(int)
    y_pred = pd.to_numeric(df["klasse"], errors="coerce").fillna(-1).astype(int)
    valid = (y_true.isin(CLASS_LABELS)) & (y_pred.isin(CLASS_LABELS))
    ct = pd.crosstab(y_true[valid], y_pred[valid], rownames=["reference"], colnames=["prediction"])
    ct = ct.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)
    return ct.astype(int)


def _per_class_metrics(cm: pd.DataFrame) -> pd.DataFrame:
    """cm: rows=reference, cols=prediction."""
    rows = []
    for c in CLASS_LABELS:
        tp = int(cm.loc[c, c]) if c in cm.index and c in cm.columns else 0
        support = int(cm.loc[c].sum()) if c in cm.index else 0
        pred_pos = int(cm[c].sum()) if c in cm.columns else 0
        precision = safe_div(tp, pred_pos)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "class": c,
                "support": support,
                "predicted_positive": pred_pos,
                "tp_on_diagonal": tp,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return pd.DataFrame(rows)


def _ordinal_summaries(df: pd.DataFrame) -> Dict[str, float]:
    y_true = pd.to_numeric(df["baseline_reference_class"], errors="coerce")
    y_pred = pd.to_numeric(df["klasse"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    diff = (y_pred[valid] - y_true[valid]).astype(float)
    mae = float(diff.abs().mean()) if len(diff) else 0.0
    return {
        "mean_absolute_error_ordinal": round(mae, 6),
        "count_pred_gt_reference": int((diff > 0).sum()),
        "count_pred_lt_reference": int((diff < 0).sum()),
        "count_pred_eq_reference": int((diff == 0).sum()),
    }


def _plot_class_distributions(df: pd.DataFrame, out_path: Path) -> None:
    pred = df["klasse"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    ref = df["baseline_reference_class"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    x = np.arange(len(CLASS_LABELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, pred.values, width=w, label="prediction (klasse)")
    ax.bar(x + w / 2, ref.values, width=w, label="reference (baseline_reference_class)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLASS_LABELS])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Class distribution: prediction vs reference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion_matrix(cm: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[str(c) for c in cm.columns],
        yticklabels=[str(c) for c in cm.index],
        ylabel="Reference (baseline)",
        xlabel="Prediction (klasse)",
        title="3-class confusion matrix",
    )
    thresh = cm.values.max() / 2.0 if cm.values.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm.values[i, j], "d"), ha="center", va="center", color="w" if cm.values[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_ref_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Jittered scatter for ordinal comparison."""
    y_true = pd.to_numeric(df["baseline_reference_class"], errors="coerce")
    y_pred = pd.to_numeric(df["klasse"], errors="coerce")
    mask = y_true.notna() & y_pred.notna() & y_true.isin(CLASS_LABELS) & y_pred.isin(CLASS_LABELS)
    y_true = y_true[mask].astype(float)
    y_pred = y_pred[mask].astype(float)
    rng = np.random.default_rng(42)
    jitter = lambda s: s + rng.normal(0, 0.06, size=len(s))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(jitter(y_true), jitter(y_pred), alpha=0.5, s=25)
    ax.plot([-0.5, 2.5], [-0.5, 2.5], "k--", linewidth=0.8, label="perfect agreement")
    ax.set_xticks(CLASS_LABELS)
    ax.set_yticks(CLASS_LABELS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Reference (baseline_reference_class)")
    ax.set_ylabel("Prediction (klasse)")
    ax.set_title("Prediction vs reference (jittered)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = EVALUATION_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "klasse" not in df.columns:
        raise ValueError("Spalte 'klasse' fehlt.")
    if "PatientenID" not in df.columns:
        raise ValueError("Spalte 'PatientenID' fehlt.")

    df = _ensure_reference_class(df)
    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    df["baseline_reference_class"] = pd.to_numeric(df["baseline_reference_class"], errors="coerce")

    cm3 = _multiclass_confusion(df)
    cm3.to_csv(EVALUATION_CONFUSION_3CLASS_PATH)
    exact_acc = safe_div((df["klasse"] == df["baseline_reference_class"]).sum(), len(df))
    per_class = _per_class_metrics(cm3)
    ordinal = _ordinal_summaries(df)

    pred_dist = df["klasse"].value_counts(dropna=False).reindex(CLASS_LABELS, fill_value=0)
    ref_dist = df["baseline_reference_class"].value_counts(dropna=False).reindex(CLASS_LABELS, fill_value=0)

    summary_rows = [
        {"metric": "n_rows", "value": str(len(df))},
        {"metric": "multiclass_exact_accuracy", "value": str(round(exact_acc, 6))},
        {"metric": "mean_absolute_error_ordinal", "value": str(ordinal["mean_absolute_error_ordinal"])},
        {"metric": "count_pred_gt_reference", "value": str(ordinal["count_pred_gt_reference"])},
        {"metric": "count_pred_lt_reference", "value": str(ordinal["count_pred_lt_reference"])},
        {"metric": "count_pred_eq_reference", "value": str(ordinal["count_pred_eq_reference"])},
        {"metric": "prediction_class_distribution", "value": str(pred_dist.to_dict())},
        {"metric": "reference_class_distribution", "value": str(ref_dist.to_dict())},
    ]
    pd.DataFrame(summary_rows).to_csv(EVALUATION_MULTICLASS_SUMMARY_PATH, index=False)
    per_class.to_csv(EVALUATION_DIR / "multiclass_per_class_metrics.csv", index=False)

    mism = df[df["klasse"] != df["baseline_reference_class"]].copy()
    over = df[df["klasse"] > df["baseline_reference_class"]].copy()
    under = df[df["klasse"] < df["baseline_reference_class"]].copy()
    mism.to_csv(EVALUATION_DIR / "errors_exact_mismatch.csv", index=False)
    over.to_csv(EVALUATION_DIR / "errors_overcall_pred_gt_reference.csv", index=False)
    under.to_csv(EVALUATION_DIR / "errors_undercall_pred_lt_reference.csv", index=False)

    _plot_class_distributions(df, plots_dir / "class_distribution_pred_vs_ref.png")
    _plot_confusion_matrix(cm3, plots_dir / "confusion_matrix_3class.png")
    _plot_pred_vs_ref_scatter(df, plots_dir / "prediction_vs_reference_scatter.png")

    binary_frames = []
    if "any_delir_flag" not in df.columns or "has_delir_icd10" not in df.columns:
        LOGGER.warning("Binary secondary evaluation skipped: missing ICD helper columns.")
    else:
        df_bin = df if "baseline_delir_reference" in df.columns else add_reference_class(df.copy())

        eval_icdsc = (
            df_bin["agreement_report_vs_icdsc"].value_counts(dropna=False)
            if "agreement_report_vs_icdsc" in df_bin.columns
            else pd.Series(dtype=int)
        )
        eval_icd10 = (
            df_bin["agreement_report_vs_icd10"].value_counts(dropna=False)
            if "agreement_report_vs_icd10" in df_bin.columns
            else pd.Series(dtype=int)
        )
        eval_combined = (
            df_bin["agreement_report_vs_combined_baseline"].value_counts(dropna=False)
            if "agreement_report_vs_combined_baseline" in df_bin.columns
            else pd.Series(dtype=int)
        )

        crosstab_icdsc = pd.crosstab(df_bin["klasse"], df_bin["any_delir_flag"], dropna=False)
        crosstab_icd10 = pd.crosstab(df_bin["klasse"], df_bin["has_delir_icd10"], dropna=False)
        crosstab_combined = pd.crosstab(df_bin["klasse"], df_bin["baseline_delir_reference"], dropna=False)

        df_binary = df_bin.copy()
        df_binary["pred_delir"] = (df_binary["klasse"] == 2).astype(int)
        df_binary["true_delir"] = pd.to_numeric(df_binary["baseline_delir_reference"], errors="coerce").fillna(0).astype(int)

        tp = int(((df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 1)).sum())
        tn = int(((df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 0)).sum())
        fp = int(((df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 0)).sum())
        fn = int(((df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 1)).sum())

        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        confusion_matrix_binary = pd.DataFrame(
            {"pred_0": [tn, fn], "pred_1": [fp, tp]},
            index=["true_0", "true_1"],
        )

        false_positives_bin = df_binary[(df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 0)].copy()
        false_negatives_bin = df_binary[(df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 1)].copy()

        legacy_rows = [
            {"metric": "agreement_report_vs_icdsc", "value": str(eval_icdsc.to_dict())},
            {"metric": "agreement_report_vs_icd10", "value": str(eval_icd10.to_dict())},
            {"metric": "agreement_report_vs_combined_baseline", "value": str(eval_combined.to_dict())},
            {"metric": "tp_binary_documented_delir", "value": str(tp)},
            {"metric": "tn_binary_documented_delir", "value": str(tn)},
            {"metric": "fp_binary_documented_delir", "value": str(fp)},
            {"metric": "fn_binary_documented_delir", "value": str(fn)},
            {"metric": "accuracy_binary", "value": str(round(accuracy, 6))},
            {"metric": "precision_binary", "value": str(round(precision, 6))},
            {"metric": "recall_binary", "value": str(round(recall, 6))},
            {"metric": "f1_binary", "value": str(round(f1, 6))},
        ]
        pd.DataFrame(legacy_rows).to_csv(EVALUATION_DIR / "evaluation_binary_secondary_summary.csv", index=False)

        crosstab_icdsc.to_csv(EVALUATION_DIR / "crosstab_report_vs_icdsc.csv")
        crosstab_icd10.to_csv(EVALUATION_DIR / "crosstab_report_vs_icd10.csv")
        crosstab_combined.to_csv(EVALUATION_DIR / "crosstab_report_vs_combined_baseline.csv")
        confusion_matrix_binary.to_csv(EVALUATION_DIR / "confusion_matrix_binary_baseline_secondary.csv")
        false_positives_bin.to_csv(EVALUATION_DIR / "false_positives_binary_secondary.csv", index=False)
        false_negatives_bin.to_csv(EVALUATION_DIR / "false_negatives_binary_secondary.csv", index=False)
        binary_frames.append(pd.DataFrame(legacy_rows))

    multiclass_df = pd.read_csv(EVALUATION_MULTICLASS_SUMMARY_PATH)
    if binary_frames:
        combined_summary = pd.concat([multiclass_df, binary_frames[0]], ignore_index=True)
    else:
        combined_summary = multiclass_df
    combined_summary.to_csv(EVALUATION_SUMMARY_PATH, index=False)

    print("\n=== Multiclass (primary) ===")
    print(f"Exact accuracy: {exact_acc:.4f}")
    print("Confusion (rows=reference, cols=prediction):\n", cm3)
    print("Ordinal summaries:", ordinal)
    print(f"\nArtifacts under: {EVALUATION_DIR}")
    print(f"Plots under:     {plots_dir}")


if __name__ == "__main__":
    main()
