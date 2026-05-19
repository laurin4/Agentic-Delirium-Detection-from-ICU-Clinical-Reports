"""
Normalize heterogeneous column names in structured baseline source CSVs.

Final production schema (semicolon-separated):
- ICD.csv: PatientID; icd_hd; icd_code
- ICDSC.csv: PatientID; ICDSC_Max  (patient-level maximum, one row per patient typical)

Canonical internal names:
- PatientenID
- Code, IsHauptDiagn (from icd_code, icd_hd)
- ICDSC_Max (source) → max_icdsc (baseline output)
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

ICD10_COLUMN_ALIASES: dict[str, str] = {
    "icd_code": "Code",
    "icd_hd": "IsHauptDiagn",
}

# Legacy long-format ICDSC column names (synthetic / old exports).
ICDSC_LEGACY_VALUE_ALIASES: tuple[str, ...] = ("ICDSC_Value",)

ICDSC_MAX_ALIASES: tuple[str, ...] = ("ICDSC_Max", "max_icdsc")

# Thesis baseline: only these F05 subcodes (main diagnosis icd_hd==1 applied in prepare_icd10).
VALID_DELIR_ICD10_CODES: frozenset[str] = frozenset({"F05.0", "F05.8", "F05.9"})
# F05.1 = alcohol-related / withdrawal delirium — excluded from intended delirium cohort.
EXCLUDED_DELIR_ICD10_CODE = "F05.1"


class SchemaValidationError(ValueError):
    """Raised when a required column is missing after alias normalization."""


def _debug_patient_id_enabled() -> bool:
    raw = os.environ.get("DEBUG_PATIENT_ID", os.environ.get("DEBUG_LLM_OUTPUT", ""))
    return raw.strip().lower() in ("1", "true", "yes")


def clean_patient_id_value(value: object) -> str:
    """
    Normalize one patient identifier to a stripped string without float artifacts.

    Examples: 12345 -> "12345", 12345.0 -> "12345", NaN -> "".
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        try:
            if value == int(value):
                return str(int(value))
        except (TypeError, ValueError, OverflowError):
            pass
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit() or (head.startswith("-") and head[1:].isdigit()):
            return head
    return s


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
        out["PatientenID"] = out["PatientenID"].map(clean_patient_id_value)
    return out


def normalize_patient_id_column(df: pd.DataFrame, column: str = "PatientenID") -> pd.DataFrame:
    """
    Return a copy with canonical ``PatientenID`` as clean strings (merge-safe).

    Renames ``PatientID`` when needed, then applies :func:`clean_patient_id_value`.
    """
    out = normalize_patient_id_columns(df)
    if column in out.columns:
        out[column] = out[column].map(clean_patient_id_value)
    return out


def assert_patientenid_column(df: pd.DataFrame, context: str = "dataframe") -> None:
    """Raise if ``PatientenID`` is missing (after alias normalization)."""
    if df.empty:
        return
    if "PatientenID" not in df.columns and "PatientID" not in df.columns:
        raise SchemaValidationError(
            f"{context}: missing required column 'PatientenID'. "
            f"Available columns: {list(df.columns)}"
        )


def log_patientenid_dtype_if_debug(df: pd.DataFrame, context: str) -> None:
    """Log ``PatientenID`` dtype when DEBUG_PATIENT_ID or DEBUG_LLM_OUTPUT is set."""
    if not _debug_patient_id_enabled() or df.empty or "PatientenID" not in df.columns:
        return
    LOGGER.info(
        "%s PatientenID dtype=%s sample=%s",
        context,
        df["PatientenID"].dtype,
        df["PatientenID"].head(3).tolist(),
    )


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
    Ensure patient-level ICDSC score column ``ICDSC_Max`` exists.

    Legacy long-format ``ICDSC_Value`` is renamed to ``ICDSC_Max`` for downstream max().
    """
    out = df.copy()
    if "ICDSC_Max" in out.columns:
        return out
    for alias in ICDSC_LEGACY_VALUE_ALIASES:
        if alias in out.columns:
            return out.rename(columns={alias: "ICDSC_Max"})
    for alias in ICDSC_MAX_ALIASES:
        if alias in out.columns and alias != "ICDSC_Max":
            return out.rename(columns={alias: "ICDSC_Max"})
    return out


def is_main_diagnosis_flag(value: object) -> bool:
    """Treat icd_hd / IsHauptDiagn == 1 as main diagnosis."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip()
    if not s or s.lower() in ("nan", "null", "none"):
        return False
    try:
        return int(float(s)) == 1
    except (TypeError, ValueError):
        return s in ("1", "True", "true", "JA", "Ja", "ja")


def normalize_icd_code(code: object) -> str:
    return str(code or "").strip().upper()


def is_valid_delir_icd10_code(code: object) -> bool:
    """
    Return whether an ICD-10 code counts toward delirium baseline (main diagnosis applied separately).

    Included: F05.0, F05.8, F05.9 only.
    Excluded: F05.1 (alcohol-related delirium / Entzugsdelir — outside intended cohort)
    and all other F05 subcodes.
    """
    normalized = normalize_icd_code(code)
    if not normalized:
        return False
    if normalized == EXCLUDED_DELIR_ICD10_CODE:
        return False
    return normalized in VALID_DELIR_ICD10_CODES


def require_icd10_source_columns(df: pd.DataFrame, context: str = "ICD input") -> None:
    """Validate ICD source after patient + column alias normalization."""
    require_columns(df, ("PatientenID", "Code", "IsHauptDiagn"), context)


def require_icdsc_source_columns(df: pd.DataFrame, context: str = "ICDSC input") -> None:
    """Validate ICDSC source after patient + column alias normalization."""
    require_columns(df, ("PatientenID", "ICDSC_Max"), context)


def structured_baseline_output_columns() -> tuple[str, ...]:
    """Standard columns written to structured_baseline.csv (excluding reference-class extras)."""
    return (
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
        "baseline_composite",
    )


def assert_structured_baseline_columns(df: pd.DataFrame, context: str = "structured baseline") -> None:
    """Ensure downstream baseline artifact has expected standardized columns."""
    require_columns(df, structured_baseline_output_columns(), context)
