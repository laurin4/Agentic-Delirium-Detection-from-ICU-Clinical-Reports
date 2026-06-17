"""
Merge patient-level model predictions with structured baseline (one row per PatientenID).

Mirrors ``compare_reports_vs_baseline`` but aggregates report predictions first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

from src.pipeline.baseline_composite import PRIMARY_EVALUATION_BASELINES
from src.pipeline.compare_reports_vs_baseline import (
    _baseline_patient_ids,
    _build_excluded_export,
    _first_n_unique_patient_ids,
    load_data,
)
from src.pipeline.patient_prediction_aggregate import aggregate_predictions_to_patient_level
from src.pipeline.paths import (
    PATIENT_VS_BASELINE_EXCLUDED_PATH,
    PATIENT_VS_BASELINE_PATH,
    STRUCTURED_BASELINE_PATH,
)
from src.pipeline.predictions_source import log_predictions_source, resolve_predictions_path
from src.pipeline.prepare_structured_data import add_reference_class
from src.pipeline.schema_normalize import SchemaValidationError, require_columns

# Baseline columns required for patient-level evaluation (primary four + identifiers).
REQUIRED_PATIENT_BASELINE_COLUMNS = list(PRIMARY_EVALUATION_BASELINES) + [
    "has_delir_icd10",
    "max_icdsc",
]


def _ensure_required_patient_baseline_columns_exist(merged: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_PATIENT_BASELINE_COLUMNS if c not in merged.columns]
    if missing_cols:
        raise ValueError(
            "structured_baseline.csv or merge result is missing required baseline columns: "
            + ", ".join(missing_cols)
            + ". Re-run prepare_structured_data with an up-to-date pipeline."
        )


def _split_evaluable_vs_excluded(
    merged: pd.DataFrame,
    baseline_ids: Set[str],
) -> Tuple[pd.Series, pd.Series]:
    subset = merged[REQUIRED_PATIENT_BASELINE_COLUMNS]
    in_baseline = merged["PatientenID"].astype(str).str.strip().isin(baseline_ids)
    has_complete_baseline = ~subset.isna().any(axis=1)
    evaluable_mask = in_baseline & has_complete_baseline

    reason = pd.Series("", index=merged.index, dtype=object)
    reason.loc[~in_baseline] = "no_structured_baseline_row"
    reason.loc[in_baseline & ~has_complete_baseline] = "incomplete_baseline_columns"
    return evaluable_mask, reason


def run_compare(
    baseline_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    excluded_path: Optional[Path] = None,
    aggregate_output_path: Optional[Path] = None,
) -> None:
    baseline_path = baseline_path or STRUCTURED_BASELINE_PATH
    explicit_pred = predictions_path is not None
    predictions_path = resolve_predictions_path(predictions_path)
    log_predictions_source(predictions_path, explicit_path=explicit_pred)
    output_path = output_path or PATIENT_VS_BASELINE_PATH
    excluded_path = excluded_path or PATIENT_VS_BASELINE_EXCLUDED_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline, reports = load_data(baseline_path, predictions_path)

    try:
        require_columns(
            baseline,
            ("PatientenID",),
            context=f"structured baseline ({baseline_path.name})",
        )
        require_columns(
            reports,
            ("PatientenID", "klasse"),
            context=f"report predictions ({predictions_path.name})",
        )
    except SchemaValidationError as exc:
        raise ValueError(str(exc)) from exc

    patients = aggregate_predictions_to_patient_level(reports)
    n_report_rows = len(reports)
    n_patients_agg = len(patients)

    if aggregate_output_path is not None:
        aggregate_output_path.parent.mkdir(parents=True, exist_ok=True)
        patients.to_csv(aggregate_output_path, index=False)

    baseline = baseline.copy()
    baseline_ids = _baseline_patient_ids(baseline)
    merged = patients.merge(baseline, on="PatientenID", how="left")

    _ensure_required_patient_baseline_columns_exist(merged)

    evaluable_mask, reason = _split_evaluable_vs_excluded(merged, baseline_ids)
    merged["reason"] = reason

    excluded_mask = ~evaluable_mask
    n_excluded = int(excluded_mask.sum())
    n_evaluable = int(evaluable_mask.sum())

    if n_excluded:
        excluded_export = _build_excluded_export(merged.loc[excluded_mask], patients.columns)
    else:
        pred_tail = [c for c in patients.columns if c != "PatientenID"]
        excluded_export = pd.DataFrame(columns=["PatientenID", "reason"] + pred_tail)

    excluded_export.to_csv(excluded_path, index=False)

    evaluable = merged.loc[evaluable_mask].drop(columns=["reason"]).copy()
    evaluable = add_reference_class(evaluable)
    evaluable["model_patient_positive"] = (
        pd.to_numeric(evaluable["model_patient_positive"], errors="coerce").fillna(0).astype(int)
    )
    evaluable["prediction_binary"] = evaluable["model_patient_positive"]

    for baseline_col in PRIMARY_EVALUATION_BASELINES:
        agreement_col = f"agreement_patient_vs_{baseline_col}"
        evaluable[baseline_col] = (
            pd.to_numeric(evaluable[baseline_col], errors="coerce").fillna(0).astype(int)
        )
        evaluable[agreement_col] = evaluable["prediction_binary"] == evaluable[baseline_col]

    evaluable["agreement_patient_vs_icdsc"] = evaluable["agreement_patient_vs_baseline_icdsc_ge_4"]
    evaluable["agreement_patient_vs_icd10"] = evaluable["agreement_patient_vs_baseline_icd10"]

    evaluable.to_csv(output_path, index=False)

    preview_ids = _first_n_unique_patient_ids(merged.loc[excluded_mask, "PatientenID"], 20)

    print(f"Gespeichert (evaluierbar, patient-level): {output_path}")
    print(f"Gespeichert (ausgeschlossen fehlende Baseline): {excluded_path}")
    print(f"Report-Zeilen in Predictions: {n_report_rows}")
    print(f"Patienten aggregiert (gueltige Report-Predictions): {n_patients_agg}")
    print(f"Evaluierbare Patienten: {n_evaluable}")
    print(f"Ausgeschlossene Patienten: {n_excluded}")
    if preview_ids:
        print(f"Erste ausgeschlossene PatientenIDs (bis 20): {preview_ids}")


def main() -> None:
    from src.pipeline.paths import EVALUATION_PATIENT_LEVEL_AGGREGATE_PATH

    run_compare(aggregate_output_path=EVALUATION_PATIENT_LEVEL_AGGREGATE_PATH)


if __name__ == "__main__":
    main()
