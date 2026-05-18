import pandas as pd
from pathlib import Path
from typing import Optional, Set, Tuple

from src.pipeline.paths import (
    STRUCTURED_BASELINE_PATH,
    REPORT_VS_BASELINE_PATH,
    REPORT_VS_BASELINE_EXCLUDED_PATH,
    PREDICTIONS_DIR,
)
from src.analysis.evidence_snippets import attach_evidence_snippets_to_dataframe
from src.pipeline.prepare_structured_data import add_reference_class
from src.pipeline.schema_normalize import (
    SchemaValidationError,
    normalize_patient_id_columns,
    require_columns,
)

REPORT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"

# Columns that must be present on every evaluable merged row (from structured baseline).
# Missing values indicate no baseline row or incomplete baseline data for that PatientenID.
REQUIRED_BASELINE_COLUMNS = [
    "baseline_composite",
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


def _ensure_required_baseline_columns_exist(merged: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_BASELINE_COLUMNS if c not in merged.columns]
    if missing_cols:
        raise ValueError(
            "structured_baseline.csv or merge result is missing required baseline columns: "
            + ", ".join(missing_cols)
            + ". Re-run prepare_structured_data with an up-to-date pipeline."
        )


def _baseline_patient_ids(baseline: pd.DataFrame) -> Set[str]:
    return set(baseline["PatientenID"].astype(str).str.strip().unique())


def _split_evaluable_vs_excluded(
    merged: pd.DataFrame,
    baseline_ids: Set[str],
) -> Tuple[pd.Series, pd.Series]:
    """Return (evaluable_mask, reason_series) without filling missing baseline values."""
    subset = merged[REQUIRED_BASELINE_COLUMNS]
    in_baseline = merged["PatientenID"].astype(str).str.strip().isin(baseline_ids)
    has_complete_baseline = ~subset.isna().any(axis=1)
    evaluable_mask = in_baseline & has_complete_baseline

    reason = pd.Series("", index=merged.index, dtype=object)
    reason.loc[~in_baseline] = "no_structured_baseline_row"
    reason.loc[in_baseline & ~has_complete_baseline] = "incomplete_baseline_columns"
    return evaluable_mask, reason


def _first_n_unique_patient_ids(series: pd.Series, n: int) -> list:
    seen = set()
    out = []
    for pid in series.astype(str).str.strip().tolist():
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
        if len(out) >= n:
            break
    return out


def _build_excluded_export(
    merged_excluded: pd.DataFrame,
    reports_columns: pd.Index,
) -> pd.DataFrame:
    pred_cols_ordered = [
        c for c in reports_columns if c in merged_excluded.columns and c != "PatientenID"
    ]
    data = {
        "PatientenID": merged_excluded["PatientenID"].values,
        "reason": merged_excluded["reason"].values,
    }
    for c in pred_cols_ordered:
        data[c] = merged_excluded[c].values
    return pd.DataFrame(data)


def load_data(
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    predictions_path: Path = REPORT_PREDICTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Structured baseline not found: {baseline_path}. "
            "Run 'python -m src.pipeline.prepare_structured_data' first."
        )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}. "
            "Run 'python -m src.pipeline.run_pipeline' first."
        )
    baseline = normalize_patient_id_columns(pd.read_csv(baseline_path))
    reports = normalize_patient_id_columns(pd.read_csv(predictions_path))
    return baseline, reports


def run_compare(
    baseline_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    excluded_path: Optional[Path] = None,
) -> None:
    baseline_path = baseline_path or STRUCTURED_BASELINE_PATH
    predictions_path = predictions_path or REPORT_PREDICTIONS_PATH
    output_path = output_path or REPORT_VS_BASELINE_PATH
    excluded_path = excluded_path or REPORT_VS_BASELINE_EXCLUDED_PATH

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
            ("PatientenID",),
            context=f"report predictions ({predictions_path.name})",
        )
    except SchemaValidationError as exc:
        raise ValueError(str(exc)) from exc

    reports = reports.copy()
    baseline = baseline.copy()

    baseline_ids = _baseline_patient_ids(baseline)
    merged = reports.merge(baseline, on="PatientenID", how="left")

    _ensure_required_baseline_columns_exist(merged)

    evaluable_mask, reason = _split_evaluable_vs_excluded(merged, baseline_ids)
    merged["reason"] = reason

    excluded_mask = ~evaluable_mask
    n_total = len(merged)
    n_evaluable = int(evaluable_mask.sum())
    n_excluded = int(excluded_mask.sum())

    if n_excluded:
        excluded_export = _build_excluded_export(merged.loc[excluded_mask], reports.columns)
    else:
        pred_tail = [c for c in reports.columns if c != "PatientenID"]
        excluded_export = pd.DataFrame(columns=["PatientenID", "reason"] + pred_tail)

    excluded_export.to_csv(excluded_path, index=False)

    evaluable = merged.loc[evaluable_mask].drop(columns=["reason"]).copy()
    evaluable = add_reference_class(evaluable)
    evaluable["klasse"] = pd.to_numeric(evaluable["klasse"], errors="coerce")
    evaluable["prediction_binary"] = (evaluable["klasse"] == 1).astype(int)

    for threshold in [1, 2, 3, 4, 5]:
        baseline_col = f"baseline_icdsc_ge_{threshold}"
        agreement_col = f"agreement_report_vs_{baseline_col}"
        evaluable[baseline_col] = pd.to_numeric(evaluable[baseline_col], errors="coerce").fillna(0).astype(int)
        evaluable[agreement_col] = evaluable["prediction_binary"] == evaluable[baseline_col]

    evaluable["baseline_icd10"] = pd.to_numeric(evaluable["baseline_icd10"], errors="coerce").fillna(0).astype(int)
    evaluable["agreement_report_vs_baseline_icd10"] = evaluable["prediction_binary"] == evaluable["baseline_icd10"]
    if "baseline_composite" in evaluable.columns:
        evaluable["baseline_composite"] = (
            pd.to_numeric(evaluable["baseline_composite"], errors="coerce").fillna(0).astype(int)
        )
        evaluable["agreement_report_vs_baseline_composite"] = (
            evaluable["prediction_binary"] == evaluable["baseline_composite"]
        )

    # Legacy columns kept for backwards compatibility with older analyses.
    evaluable["agreement_report_vs_icdsc"] = evaluable["agreement_report_vs_baseline_icdsc_ge_4"]
    evaluable["agreement_report_vs_icd10"] = evaluable["agreement_report_vs_baseline_icd10"]
    # Deprecated/disabled: project now uses binary baselines; comparing binary klasse
    # against legacy multiclass baseline_reference_class would be semantically wrong.
    evaluable["agreement_report_vs_combined_baseline"] = pd.NA

    evaluable = attach_evidence_snippets_to_dataframe(evaluable)

    evaluable.to_csv(output_path, index=False)

    preview_ids = _first_n_unique_patient_ids(merged.loc[excluded_mask, "PatientenID"], 20)

    print(f"Gespeichert (evaluierbar): {output_path}")
    print(f"Gespeichert (ausgeschlossen fehlende Baseline): {excluded_path}")
    print(f"Prediction-Zeilen gesamt: {n_total}")
    print(f"Evaluierbare Zeilen: {n_evaluable}")
    print(f"Ausgeschlossene Zeilen: {n_excluded}")
    if preview_ids:
        print(f"Erste ausgeschlossene PatientenIDs (bis 20): {preview_ids}")


def main() -> None:
    run_compare()


if __name__ == "__main__":
    main()
