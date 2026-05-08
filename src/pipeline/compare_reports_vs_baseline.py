import pandas as pd
from src.pipeline.paths import (
    STRUCTURED_BASELINE_PATH,
    REPORT_VS_BASELINE_PATH,
    PREDICTIONS_DIR,
)
from src.pipeline.prepare_structured_data import add_reference_class

REPORT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"

# Columns that must be present on every merged row (from structured baseline).
# Missing values indicate no baseline row or incomplete baseline data for that PatientenID.
REQUIRED_BASELINE_COLUMNS = [
    "has_delir_icd10",
    "max_icdsc",
    "baseline_icd10",
    "baseline_icdsc_ge_1",
    "baseline_icdsc_ge_2",
    "baseline_icdsc_ge_3",
    "baseline_icdsc_ge_4",
    "baseline_icdsc_ge_5",
    "baseline_icdsc_0",
    "baseline_icdsc_1_to_3",
    "baseline_icdsc_ge_4_grouped",
]


def _raise_if_incomplete_baseline_merge(merged: pd.DataFrame) -> None:
    """Ensure every prediction row has full baseline data (no silent NaN -> 0 for unmatched)."""
    missing_cols = [c for c in REQUIRED_BASELINE_COLUMNS if c not in merged.columns]
    if missing_cols:
        raise ValueError(
            "structured_baseline.csv or merge result is missing required baseline columns: "
            + ", ".join(missing_cols)
            + ". Re-run prepare_structured_data with an up-to-date pipeline."
        )

    subset = merged[REQUIRED_BASELINE_COLUMNS]
    incomplete_mask = subset.isna().any(axis=1)
    if not incomplete_mask.any():
        return

    bad_ids = (
        merged.loc[incomplete_mask, "PatientenID"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    n = len(bad_ids)
    preview = bad_ids[:20]
    raise ValueError(
        f"Prediction merge has {n} PatientenID(s) without complete baseline data "
        f"(missing baseline row and/or NaN in required baseline columns). "
        f"First up to 20 IDs: {preview!r}. "
        "Fix structured_baseline.csv coverage or prediction PatientenIDs before compare_reports_vs_baseline."
    )


def load_data():
    if not STRUCTURED_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Structured baseline not found: {STRUCTURED_BASELINE_PATH}. "
            "Run 'python -m src.pipeline.prepare_structured_data' first."
        )
    if not REPORT_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {REPORT_PREDICTIONS_PATH}. "
            "Run 'python -m src.pipeline.run_pipeline' first."
        )
    baseline = pd.read_csv(STRUCTURED_BASELINE_PATH)
    reports = pd.read_csv(REPORT_PREDICTIONS_PATH)
    return baseline, reports


def main():
    REPORT_VS_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline, reports = load_data()

    if "PatientenID" not in baseline.columns:
        raise ValueError("In structured_baseline.csv fehlt die Spalte 'PatientenID'.")

    if "PatientenID" not in reports.columns:
        raise ValueError(
            "In der Report-Prediction-CSV fehlt die Spalte 'PatientenID'. "
            "Diese muss später aus den Berichten mitgeführt werden."
        )

    reports["PatientenID"] = reports["PatientenID"].astype(str).str.strip()
    baseline["PatientenID"] = baseline["PatientenID"].astype(str).str.strip()
    merged = reports.merge(baseline, on="PatientenID", how="left")

    _raise_if_incomplete_baseline_merge(merged)

    merged = add_reference_class(merged)
    merged["klasse"] = pd.to_numeric(merged["klasse"], errors="coerce")
    merged["prediction_binary"] = (merged["klasse"] == 1).astype(int)

    for threshold in [1, 2, 3, 4, 5]:
        baseline_col = f"baseline_icdsc_ge_{threshold}"
        agreement_col = f"agreement_report_vs_{baseline_col}"
        merged[baseline_col] = pd.to_numeric(merged[baseline_col], errors="coerce").fillna(0).astype(int)
        merged[agreement_col] = merged["prediction_binary"] == merged[baseline_col]

    merged["baseline_icd10"] = pd.to_numeric(merged["baseline_icd10"], errors="coerce").fillna(0).astype(int)
    merged["agreement_report_vs_baseline_icd10"] = merged["prediction_binary"] == merged["baseline_icd10"]

    # Legacy columns kept for backwards compatibility with older analyses.
    merged["agreement_report_vs_icdsc"] = merged["agreement_report_vs_baseline_icdsc_ge_4"]
    merged["agreement_report_vs_icd10"] = merged["agreement_report_vs_baseline_icd10"]
    # Deprecated/disabled: project now uses binary baselines; comparing binary klasse
    # against legacy multiclass baseline_reference_class would be semantically wrong.
    merged["agreement_report_vs_combined_baseline"] = pd.NA

    merged.to_csv(REPORT_VS_BASELINE_PATH, index=False)

    print(f"Gespeichert: {REPORT_VS_BASELINE_PATH}")
    print(f"Anzahl gemergte Zeilen: {len(merged)}")


if __name__ == "__main__":
    main()