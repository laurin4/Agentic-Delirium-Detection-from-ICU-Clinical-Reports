"""
Aggregate report-level pipeline predictions to patient level.

Rule (same as manual validation): patient positive if any report is positive (max klasse).
Dokumentationsblatt rows are excluded when ``bertyp`` is present.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.pipeline.schema_normalize import normalize_patient_id_columns
from src.preprocessing.berichte_filters import is_dokumentationsblatt, normalize_bertyp

PATIENT_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "PatientenID",
    "n_reports_in_aggregate",
    "n_reports_positive",
    "n_reports_negative",
    "model_patient_positive",
)


def aggregate_predictions_to_patient_level(
    predictions: pd.DataFrame,
    *,
    exclude_dokumentationsblatt: bool = True,
) -> pd.DataFrame:
    """
    One row per ``PatientenID`` with ``model_patient_positive = max(klasse)`` over reports.

    Only reports with binary ``klasse`` in {0, 1} contribute. Patients with no valid
    report predictions are omitted.
    """
    if "PatientenID" not in predictions.columns:
        raise ValueError("Predictions must contain column 'PatientenID'")
    if "klasse" not in predictions.columns:
        raise ValueError("Predictions must contain column 'klasse'")

    pred = normalize_patient_id_columns(predictions.copy())
    if "bertyp" in pred.columns:
        pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
        if exclude_dokumentationsblatt:
            pred = pred[~pred["bertyp"].map(is_dokumentationsblatt)].copy()

    pred["klasse"] = pd.to_numeric(pred["klasse"], errors="coerce")
    pred = pred[pred["klasse"].isin([0, 1])].copy()
    if pred.empty:
        return pd.DataFrame(columns=list(PATIENT_AGGREGATE_COLUMNS))

    grouped = (
        pred.groupby("PatientenID", as_index=False)
        .agg(
            n_reports_in_aggregate=("klasse", "count"),
            n_reports_positive=("klasse", lambda s: int((s == 1).sum())),
            n_reports_negative=("klasse", lambda s: int((s == 0).sum())),
            model_patient_positive=("klasse", lambda s: int(s.max())),
        )
        .astype(
            {
                "n_reports_in_aggregate": int,
                "n_reports_positive": int,
                "n_reports_negative": int,
                "model_patient_positive": int,
            }
        )
    )
    return grouped[list(PATIENT_AGGREGATE_COLUMNS)]
