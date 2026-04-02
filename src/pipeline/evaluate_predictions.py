import pandas as pd
from src.pipeline.paths import REPORT_VS_BASELINE_PATH, EVALUATION_DIR, EVALUATION_SUMMARY_PATH


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def main():
    df = pd.read_csv(REPORT_VS_BASELINE_PATH)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    if "klasse" not in df.columns:
        raise ValueError("Spalte 'klasse' fehlt.")
    if "any_delir_flag" not in df.columns:
        raise ValueError("Spalte 'any_delir_flag' fehlt.")
    if "has_delir_icd10" not in df.columns:
        raise ValueError("Spalte 'has_delir_icd10' fehlt.")
    if "baseline_delir_reference" not in df.columns:
        raise ValueError("Spalte 'baseline_delir_reference' fehlt.")

    eval_icdsc = df["agreement_report_vs_icdsc"].value_counts(dropna=False)
    eval_icd10 = df["agreement_report_vs_icd10"].value_counts(dropna=False)
    eval_combined = df["agreement_report_vs_combined_baseline"].value_counts(dropna=False)

    class_distribution = df["klasse"].value_counts(dropna=False).sort_index()
    icdsc_distribution = df["any_delir_flag"].value_counts(dropna=False).sort_index()
    icd10_distribution = df["has_delir_icd10"].value_counts(dropna=False).sort_index()
    combined_distribution = df["baseline_delir_reference"].value_counts(dropna=False).sort_index()

    crosstab_icdsc = pd.crosstab(df["klasse"], df["any_delir_flag"], dropna=False)
    crosstab_icd10 = pd.crosstab(df["klasse"], df["has_delir_icd10"], dropna=False)
    crosstab_combined = pd.crosstab(df["klasse"], df["baseline_delir_reference"], dropna=False)

    # Binäre Evaluation: Modell-Delir = klasse 2, Baseline-Delir = baseline_delir_reference 1
    df_binary = df.copy()
    df_binary["pred_delir"] = (df_binary["klasse"] == 2).astype(int)
    df_binary["true_delir"] = df_binary["baseline_delir_reference"].astype(int)

    tp = int(((df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 1)).sum())
    tn = int(((df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 0)).sum())
    fp = int(((df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 0)).sum())
    fn = int(((df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 1)).sum())

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    confusion_matrix = pd.DataFrame(
        {
            "pred_0": [tn, fn],
            "pred_1": [fp, tp],
        },
        index=["true_0", "true_1"],
    )

    false_positives = df_binary[(df_binary["pred_delir"] == 1) & (df_binary["true_delir"] == 0)].copy()
    false_negatives = df_binary[(df_binary["pred_delir"] == 0) & (df_binary["true_delir"] == 1)].copy()

    print("\n=== Evaluation gegen ICDSC ===")
    print(eval_icdsc)

    print("\n=== Evaluation gegen ICD10 ===")
    print(eval_icd10)

    print("\n=== Evaluation gegen kombinierte Baseline ===")
    print(eval_combined)

    print("\n=== Verteilung Report-Klassen ===")
    print(class_distribution)

    print("\n=== Verteilung ICDSC DelirFlag ===")
    print(icdsc_distribution)

    print("\n=== Verteilung ICD10 Delir ===")
    print(icd10_distribution)

    print("\n=== Verteilung kombinierte Baseline ===")
    print(combined_distribution)

    print("\n=== Cross-tab Report vs ICDSC ===")
    print(crosstab_icdsc)

    print("\n=== Cross-tab Report vs ICD10 ===")
    print(crosstab_icd10)

    print("\n=== Cross-tab Report vs kombinierte Baseline ===")
    print(crosstab_combined)

    print("\n=== Binäre Metriken gegen kombinierte Baseline ===")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    print("\n=== Confusion Matrix gegen kombinierte Baseline ===")
    print(confusion_matrix)

    summary_rows = [
        {"metric": "agreement_report_vs_icdsc", "value": str(eval_icdsc.to_dict())},
        {"metric": "agreement_report_vs_icd10", "value": str(eval_icd10.to_dict())},
        {"metric": "agreement_report_vs_combined_baseline", "value": str(eval_combined.to_dict())},
        {"metric": "report_class_distribution", "value": str(class_distribution.to_dict())},
        {"metric": "icdsc_distribution", "value": str(icdsc_distribution.to_dict())},
        {"metric": "icd10_distribution", "value": str(icd10_distribution.to_dict())},
        {"metric": "combined_baseline_distribution", "value": str(combined_distribution.to_dict())},
        {"metric": "tp", "value": str(tp)},
        {"metric": "tn", "value": str(tn)},
        {"metric": "fp", "value": str(fp)},
        {"metric": "fn", "value": str(fn)},
        {"metric": "accuracy", "value": str(round(accuracy, 6))},
        {"metric": "precision", "value": str(round(precision, 6))},
        {"metric": "recall", "value": str(round(recall, 6))},
        {"metric": "f1", "value": str(round(f1, 6))},
    ]
    pd.DataFrame(summary_rows).to_csv(EVALUATION_SUMMARY_PATH, index=False)

    crosstab_icdsc.to_csv(EVALUATION_DIR / "crosstab_report_vs_icdsc.csv")
    crosstab_icd10.to_csv(EVALUATION_DIR / "crosstab_report_vs_icd10.csv")
    crosstab_combined.to_csv(EVALUATION_DIR / "crosstab_report_vs_combined_baseline.csv")
    confusion_matrix.to_csv(EVALUATION_DIR / "confusion_matrix_combined_baseline.csv")
    false_positives.to_csv(EVALUATION_DIR / "false_positives.csv", index=False)
    false_negatives.to_csv(EVALUATION_DIR / "false_negatives.csv", index=False)

    print(f"\nEvaluation gespeichert in: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()