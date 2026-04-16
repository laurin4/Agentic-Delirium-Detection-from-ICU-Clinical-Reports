import pandas as pd
from src.pipeline.paths import (
    STRUCTURED_BASELINE_PATH,
    REPORT_VS_BASELINE_PATH,
    PREDICTIONS_DIR,
)
from src.pipeline.prepare_structured_data import add_reference_class

REPORT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"


def load_data():
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

    merged = reports.merge(baseline, on="PatientenID", how="left")

    merged = add_reference_class(merged)

    merged["agreement_report_vs_icdsc"] = merged.apply(
        lambda row: True if (
            (row["klasse"] == 2 and row["any_delir_flag"] == 1) or
            (row["klasse"] == 0 and row["any_delir_flag"] == 0)
        ) else (None if row["klasse"] == 1 else False),
        axis=1
    )

    merged["agreement_report_vs_icd10"] = merged.apply(
        lambda row: True if (
            (row["klasse"] == 2 and row["has_delir_icd10"] == 1) or
            (row["klasse"] == 0 and row["has_delir_icd10"] == 0)
        ) else (None if row["klasse"] == 1 else False),
        axis=1
    )

    merged["agreement_report_vs_combined_baseline"] = merged.apply(
        lambda row: True if (
            (row["klasse"] == row["baseline_reference_class"])
        ) else (None if row["klasse"] == 1 else False),
        axis=1
    )

    merged.to_csv(REPORT_VS_BASELINE_PATH, index=False)

    print(f"Gespeichert: {REPORT_VS_BASELINE_PATH}")
    print(f"Anzahl gemergte Zeilen: {len(merged)}")


if __name__ == "__main__":
    main()