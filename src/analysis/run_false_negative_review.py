import pandas as pd
from pathlib import Path

from src.pipeline.paths import COMPARISONS_DIR, OUTPUTS_DIR

REVIEW_DIR = OUTPUTS_DIR / "analysis" / "false_negative_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    cmp_path = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
    df = pd.read_csv(cmp_path).copy()

    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    false_negatives = df[
        (df["baseline_reference_class"].isin([1, 2])) &
        (df["klasse"] == 0)
    ].copy()

    cols = [
        "PatientenID",
        "bericht",
        "baseline_reference_class",
        "klasse",
        "anzahl_treffer",
        "delir_signale",
        "signalstaerke",
        "kontext",
        "klassifikation",
        "klassifikation_begruendung",
    ]

    cols = [c for c in cols if c in false_negatives.columns]
    false_negatives = false_negatives[cols]

    false_negatives.to_csv(REVIEW_DIR / "false_negatives_review.csv", index=False)

    summary = pd.DataFrame([
        ("n_false_negatives_total", len(false_negatives)),
        ("n_baseline_1_pred_0", int(((df["baseline_reference_class"] == 1) & (df["klasse"] == 0)).sum())),
        ("n_baseline_2_pred_0", int(((df["baseline_reference_class"] == 2) & (df["klasse"] == 0)).sum())),
    ], columns=["metric", "count"])

    summary.to_csv(REVIEW_DIR / "false_negative_summary.csv", index=False)

    print(f"Saved review file: {REVIEW_DIR / 'false_negatives_review.csv'}")
    print(f"Saved summary file: {REVIEW_DIR / 'false_negative_summary.csv'}")

if __name__ == "__main__":
    main()