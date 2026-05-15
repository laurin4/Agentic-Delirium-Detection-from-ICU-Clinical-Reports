"""
Normalize heterogeneous column names in structured baseline source CSVs.

Canonical internal names:
- PatientenID (from PatientID or PatientenID)
- max_icdsc (output of prepare_icdsc; sources may use ICDSC_Max / ICDSC_Value)
- Code, IsHauptDiagn (ICD sources may use icd_code / icd_hd)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

# Working names used inside prepare_icdsc before patient-level aggregation.
ICDSC_VALUE_ALIASES: tuple[str, ...] = ("ICDSC_Value", "ICDSC_Max", "max_icdsc")

ICD10_COLUMN_ALIASES: dict[str, str] = {
    "icd_code": "Code",
    "icd_hd": "IsHauptDiagn",
}


class SchemaValidationError(ValueError):
    """Raised when a required column is missing after alias normalization."""


def normalize_patient_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize patient identifier columns to canonical ``PatientenID``.

    - ``PatientenID`` is kept (stripped string values).
    - ``PatientID`` is renamed to ``PatientenID`` when ``PatientenID`` is absent.
    - If both exist, ``PatientenID`` is preferred and ``PatientID`` is dropped.
    """
    out = df.copy()
    has_patienten = "PatientenID" in out.columns
    has_patient = "PatientID" in out.columns

    if has_patienten and has_patient:
        out = out.drop(columns=["PatientID"])
    elif has_patient and not has_patienten:
        out = out.rename(columns={"PatientID": "PatientenID"})

    if "PatientenID" in out.columns:
        out["PatientenID"] = out["PatientenID"].astype(str).str.strip()
    return out


def require_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    context: str,
) -> None:
    """Raise ``SchemaValidationError`` listing missing and available columns."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        available = list(df.columns)
        raise SchemaValidationError(
            f"{context}: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {available}"
        )


def normalize_icd10_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map ICD.csv schema variants to canonical ``Code`` / ``IsHauptDiagn``."""
    out = df.copy()
    for src, dst in ICD10_COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    return out


def normalize_icdsc_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ICDSC.csv schema variants to working column ``ICDSC_Value``.

    ``ICDSC_Max`` and ``max_icdsc`` are treated as per-row score columns before aggregation.
    """
    out = df.copy()
    if "ICDSC_Value" in out.columns:
        return out
    for alias in ("ICDSC_Max", "max_icdsc"):
        if alias in out.columns:
            return out.rename(columns={alias: "ICDSC_Value"})
    return out


def require_icd10_source_columns(df: pd.DataFrame, context: str = "ICD input") -> None:
    """Validate ICD source after patient + column alias normalization."""
    require_columns(df, ("PatientenID", "Code"), context)


def require_icdsc_source_columns(df: pd.DataFrame, context: str = "ICDSC input") -> None:
    """Validate ICDSC source after patient + column alias normalization."""
    require_columns(df, ("PatientenID", "ICDSC_Value"), context)


def structured_baseline_output_columns() -> tuple[str, ...]:
    """Standard columns written to structured_baseline.csv (excluding reference-class extras)."""
    base = (
        "PatientenID",
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
    )
    return base


def assert_structured_baseline_columns(df: pd.DataFrame, context: str = "structured baseline") -> None:
    """Ensure downstream baseline artifact has expected standardized columns."""
    require_columns(df, structured_baseline_output_columns(), context)
