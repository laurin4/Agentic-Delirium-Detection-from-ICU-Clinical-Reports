"""
Build structured_baseline.csv from ICD.csv + ICDSC.csv only.

Final raw inputs (semicolon-separated, see paths.py):
  PatientID; icd_hd; icd_code
  PatientID; ICDSC_Max
"""

from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.baseline_composite import (
    compute_baseline_composite,
    format_baseline_composite_mode_banner,
)
from src.pipeline.paths import ICD10_PATH, ICDSC_PATH, STRUCTURED_BASELINE_PATH
from src.pipeline.schema_normalize import (
    EXCLUDED_DELIR_ICD10_CODE,
    SchemaValidationError,
    assert_structured_baseline_columns,
    is_excluded_delir_icd10_code,
    is_main_diagnosis_flag,
    is_valid_delir_icd10_code,
    normalize_icd10_source_columns,
    normalize_icd_code,
    normalize_icdsc_source_columns,
    normalize_patient_id_columns,
    require_icd10_source_columns,
    require_icdsc_source_columns,
)
from src.pipeline.tabular_io import read_tabular

LOGGER = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ICD10_PATH.exists():
        raise FileNotFoundError(f"ICD input not found: {ICD10_PATH}")
    if not ICDSC_PATH.exists():
        raise FileNotFoundError(f"ICDSC input not found: {ICDSC_PATH}")

    icd10 = read_tabular(ICD10_PATH)
    icdsc = read_tabular(ICDSC_PATH)

    icd10 = normalize_patient_id_columns(icd10)
    icdsc = normalize_patient_id_columns(icdsc)
    icd10 = normalize_icd10_source_columns(icd10)
    icdsc = normalize_icdsc_source_columns(icdsc)

    require_icd10_source_columns(icd10, context=f"ICD input ({ICD10_PATH.name})")
    require_icdsc_source_columns(icdsc, context=f"ICDSC input ({ICDSC_PATH.name})")
    return icd10, icdsc


def _delir_codes_for_patient(codes: pd.Series) -> str:
    valid = sorted({normalize_icd_code(c) for c in codes if is_valid_delir_icd10_code(c)})
    return " | ".join(valid)


def prepare_icd10(icd10: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level ICD-10 delirium flags.

    Primary ``has_delir_icd10`` / ``baseline_icd10``: ANY row with F05.0, F05.8, or F05.9.
    Main diagnosis (icd_hd == 1) is not required. F05.1 and other F05 subcodes are excluded.

    Optional columns:
    - ``has_delir_icd10_main``: allowed code with icd_hd == 1
    - ``icd10_f051_only``: F05.1 present but no allowed F05.0/F05.8/F05.9 code
    """
    icd10 = icd10.copy()
    icd10 = normalize_patient_id_columns(icd10)
    icd10 = normalize_icd10_source_columns(icd10)
    require_icd10_source_columns(icd10, context="prepare_icd10")

    icd10["PatientenID"] = icd10["PatientenID"].astype(str).str.strip()
    icd10["Code"] = icd10["Code"].map(normalize_icd_code)
    icd10["is_main"] = icd10["IsHauptDiagn"].map(is_main_diagnosis_flag)
    icd10["is_valid_delir_code"] = icd10["Code"].map(is_valid_delir_icd10_code)
    icd10["is_f051"] = icd10["Code"].map(is_excluded_delir_icd10_code)
    icd10["is_delir_icd10_main"] = icd10["is_main"] & icd10["is_valid_delir_code"]

    grouped = (
        icd10.groupby("PatientenID", as_index=False)
        .agg(
            has_delir_icd10=("is_valid_delir_code", "max"),
            has_delir_icd10_main=("is_delir_icd10_main", "max"),
            has_f051_code=("is_f051", "max"),
            delir_codes=("Code", _delir_codes_for_patient),
        )
    )
    grouped["has_delir_icd10"] = grouped["has_delir_icd10"].fillna(False).astype(int)
    grouped["has_delir_icd10_main"] = grouped["has_delir_icd10_main"].fillna(False).astype(int)
    grouped["icd10_f051_only"] = (
        grouped["has_f051_code"].fillna(False).astype(bool) & (grouped["has_delir_icd10"] == 0)
    ).astype(int)
    grouped["delir_codes"] = grouped["delir_codes"].fillna("")
    grouped = grouped.drop(columns=["has_f051_code"], errors="ignore")
    return grouped


def icd10_baseline_report_lines(merged: pd.DataFrame) -> list[str]:
    """Summary lines for ICD10 baseline construction."""
    n_icd10 = int((merged["baseline_icd10"] == 1).sum()) if "baseline_icd10" in merged.columns else 0
    n_f051_only = int(merged.get("icd10_f051_only", pd.Series(dtype=int)).fillna(0).sum())
    n_main = (
        int((merged["baseline_icd10_main_diagnosis"] == 1).sum())
        if "baseline_icd10_main_diagnosis" in merged.columns
        else 0
    )
    return [
        "ICD10 baseline (primary): ANY diagnosis row with F05.0, F05.8, or F05.9 "
        f"(excludes {EXCLUDED_DELIR_ICD10_CODE} and other F05 subcodes; icd_hd not required)",
        f"  baseline_icd10 positive patients: {n_icd10}",
        f"  baseline_icd10_main_diagnosis positive patients: {n_main}",
        f"  patients with excluded F05.1 only: {n_f051_only}",
    ]


def prepare_icdsc(icdsc: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level max ICDSC from ``ICDSC_Max`` (already the cohort maximum per patient).

    Duplicate PatientenID rows: take max(ICDSC_Max). Non-numeric values are coerced;
    if any row fails coercion, a warning is logged. All-missing scores raise an error.
    """
    icdsc = icdsc.copy()
    icdsc = normalize_patient_id_columns(icdsc)
    icdsc = normalize_icdsc_source_columns(icdsc)
    require_icdsc_source_columns(icdsc, context="prepare_icdsc")

    icdsc["PatientenID"] = icdsc["PatientenID"].astype(str).str.strip()
    raw_max = pd.to_numeric(icdsc["ICDSC_Max"], errors="coerce")
    n_bad = int(raw_max.isna().sum())
    if n_bad:
        LOGGER.warning(
            "ICDSC: %d row(s) with missing/non-numeric ICDSC_Max; treated as 0 for aggregation.",
            n_bad,
        )
    if len(icdsc) > 0 and raw_max.isna().all():
        raise SchemaValidationError(
            "ICDSC input: all ICDSC_Max values are missing or non-numeric. "
            f"Available columns: {list(icdsc.columns)}"
        )

    icdsc["_icdsc_numeric"] = raw_max.fillna(0)

    grouped = (
        icdsc.groupby("PatientenID", as_index=False)
        .agg(max_icdsc=("_icdsc_numeric", "max"))
    )
    grouped["max_icdsc"] = grouped["max_icdsc"].fillna(0)
    return grouped[["PatientenID", "max_icdsc"]]


def add_binary_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_input_columns = ["has_delir_icd10", "max_icdsc"]
    missing_input_columns = [col for col in required_input_columns if col not in df.columns]
    if missing_input_columns:
        raise ValueError(
            "Cannot generate binary baselines: missing required columns: "
            + ", ".join(missing_input_columns)
            + f". Available columns: {list(df.columns)}"
        )

    df["has_delir_icd10"] = (
        pd.to_numeric(df["has_delir_icd10"], errors="coerce").fillna(0).astype(int)
    )
    df["max_icdsc"] = pd.to_numeric(df["max_icdsc"], errors="coerce").fillna(0)

    for threshold in [1, 2, 3, 4, 5]:
        df[f"baseline_icdsc_ge_{threshold}"] = (df["max_icdsc"] >= threshold).astype(int)
    df["baseline_icd10"] = (df["has_delir_icd10"] == 1).astype(int)
    df["baseline_icd10_any"] = df["baseline_icd10"]
    if "has_delir_icd10_main" in df.columns:
        main = pd.to_numeric(df["has_delir_icd10_main"], errors="coerce").fillna(0).astype(int)
        df["baseline_icd10_main_diagnosis"] = (main == 1).astype(int)
    else:
        df["baseline_icd10_main_diagnosis"] = 0
    df["baseline_icdsc_0"] = (df["max_icdsc"] == 0).astype(int)
    df["baseline_icdsc_1_to_3"] = (
        (df["max_icdsc"] >= 1) & (df["max_icdsc"] <= 3)
    ).astype(int)
    df["baseline_icdsc_ge_4_grouped"] = (df["max_icdsc"] >= 4).astype(int)
    df["baseline_composite_or"] = compute_baseline_composite(
        df["baseline_icdsc_ge_4"],
        df["baseline_icd10"],
        mode="OR",
    )
    df["baseline_composite_and"] = compute_baseline_composite(
        df["baseline_icdsc_ge_4"],
        df["baseline_icd10"],
        mode="AND",
    )
    df["baseline_composite"] = compute_baseline_composite(
        df["baseline_icdsc_ge_4"],
        df["baseline_icd10"],
    )
    return df


def add_reference_class(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy multiclass reference (not primary evaluation); kept for backward compatibility."""
    df = df.copy()
    df = add_binary_baselines(df)

    def _assign_class(row):
        has_icd10_delir = row["has_delir_icd10"] == 1
        max_icdsc = row["max_icdsc"]

        if has_icd10_delir:
            return 2
        if max_icdsc >= 6:
            return 2
        if max_icdsc >= 4:
            return 1
        return 0

    df["baseline_reference_class"] = df.apply(_assign_class, axis=1)
    df["baseline_delir_reference"] = (df["baseline_reference_class"] == 2).astype(int)
    return df


def build_structured_baseline(icd10: pd.DataFrame, icdsc: pd.DataFrame) -> pd.DataFrame:
    """Merge ICD + ICDSC patient tables and attach binary baseline columns."""
    icd10_prepared = prepare_icd10(icd10)
    icdsc_prepared = prepare_icdsc(icdsc)
    merged = icd10_prepared.merge(icdsc_prepared, on="PatientenID", how="outer")
    merged["has_delir_icd10"] = merged["has_delir_icd10"].fillna(0).astype(int)
    merged["has_delir_icd10_main"] = merged.get("has_delir_icd10_main", 0)
    merged["has_delir_icd10_main"] = merged["has_delir_icd10_main"].fillna(0).astype(int)
    merged["icd10_f051_only"] = merged.get("icd10_f051_only", 0)
    merged["icd10_f051_only"] = merged["icd10_f051_only"].fillna(0).astype(int)
    merged["delir_codes"] = merged.get("delir_codes", "").fillna("")
    merged["max_icdsc"] = merged["max_icdsc"].fillna(0)
    merged = add_reference_class(merged)
    assert_structured_baseline_columns(merged)
    return merged


def main() -> None:
    print(format_baseline_composite_mode_banner())
    STRUCTURED_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    icd10, icdsc = load_data()
    merged = build_structured_baseline(icd10, icdsc)
    merged.to_csv(STRUCTURED_BASELINE_PATH, index=False)
    print(f"Gespeichert: {STRUCTURED_BASELINE_PATH}")
    print(f"Anzahl Patienten: {len(merged)}")
    for line in icd10_baseline_report_lines(merged):
        print(line)


if __name__ == "__main__":
    main()
