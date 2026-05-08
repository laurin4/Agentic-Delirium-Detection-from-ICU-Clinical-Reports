"""
Binary error review: model predictions vs a chosen structured baseline column.

LEGACY: Older versions filtered on multiclass `baseline_reference_class`.
The primary review path is now binary.

Adjust REVIEW_BASELINE_COLUMN to compare against another baseline (e.g.
`baseline_icdsc_ge_4_grouped`, `baseline_icd10`).
"""

import pandas as pd

from src.pipeline.paths import COMPARISONS_DIR, OUTPUTS_DIR

REVIEW_DIR = OUTPUTS_DIR / "analysis" / "false_negative_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Default primary review baseline (ICDSC threshold >= 4 at patient level).
REVIEW_BASELINE_COLUMN = "baseline_icdsc_ge_4"


def main():
    cmp_path = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
    df = pd.read_csv(cmp_path).copy()

    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()
    if REVIEW_BASELINE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{REVIEW_BASELINE_COLUMN}' missing in {cmp_path}. "
            "Re-run compare_reports_vs_baseline after prepare_structured_data, "
            "or set REVIEW_BASELINE_COLUMN to an existing baseline column."
        )

    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    base = pd.to_numeric(df[REVIEW_BASELINE_COLUMN], errors="coerce").fillna(0).astype(int)

    false_negatives = df[(base == 1) & (df["klasse"] == 0)].copy()
    false_positives = df[(base == 0) & (df["klasse"] == 1)].copy()

    cols = [
        "PatientenID",
        "bericht",
        REVIEW_BASELINE_COLUMN,
        "klasse",
        "anzahl_treffer",
        "delir_signale",
        "signalstaerke",
        "kontext",
        "klassifikation",
        "klassifikation_begruendung",
    ]
    cols = [c for c in cols if c in df.columns]

    false_negatives = false_negatives[[c for c in cols if c in false_negatives.columns]]
    false_positives = false_positives[[c for c in cols if c in false_positives.columns]]

    false_negatives.to_csv(REVIEW_DIR / "false_negatives_review.csv", index=False)
    false_positives.to_csv(REVIEW_DIR / "false_positives_review.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "review_baseline_column", "value": REVIEW_BASELINE_COLUMN},
            {"metric": "n_false_negatives_baseline1_pred0", "value": str(len(false_negatives))},
            {"metric": "n_false_positives_baseline0_pred1", "value": str(len(false_positives))},
        ]
    )
    summary.to_csv(REVIEW_DIR / "binary_error_summary.csv", index=False)

    print(f"Baseline column: {REVIEW_BASELINE_COLUMN}")
    print(f"Saved: {REVIEW_DIR / 'false_negatives_review.csv'}")
    print(f"Saved: {REVIEW_DIR / 'false_positives_review.csv'}")
    print(f"Saved: {REVIEW_DIR / 'binary_error_summary.csv'}")


if __name__ == "__main__":
    main()
