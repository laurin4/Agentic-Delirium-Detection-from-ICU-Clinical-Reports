"""
Deterministic validation of diagnosis, ICD10, ICDSC, and optional baseline artifacts.

Reads paths only from src.pipeline.paths (DATA_MODE selects real CSV files vs optional synthetic CSVs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.pipeline.tabular_io import read_tabular
from src.pipeline.paths import (
    DIAGNOSIS_INPUT_PATH,
    ICD10_PATH,
    ICDSC_PATH,
    STRUCTURED_BASELINE_PATH,
    VALIDATION_DIR,
    VALIDATION_RESULTS_CSV_PATH,
    VALIDATION_SUMMARY_TXT_PATH,
)
from src.pipeline.prepare_structured_data import add_reference_class
from src.preprocessing.diagnosis_mapper import build_patient_level_reports

LOGGER = logging.getLogger(__name__)


def _norm_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _load_icd_patients(path: Path, col_candidates: Tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_tabular(path)
    if "PatientenID" not in df.columns and "PatientID" in df.columns:
        df = df.rename(columns={"PatientID": "PatientenID"})
    for col in col_candidates:
        if col in df.columns:
            out = df[[col]].copy()
            out.rename(columns={col: "PatientenID"}, inplace=True)
            out["PatientenID"] = _norm_pid(out["PatientenID"])
            return out
    return pd.DataFrame(columns=["PatientenID"])


def _diagnosis_patient_reports() -> pd.DataFrame:
    """Uses same preprocessing as pipeline; input path from DATA_MODE."""
    return build_patient_level_reports()


def run_checks() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    reports_df = _diagnosis_patient_reports()
    n_reports = len(reports_df)
    dup_reports = int(reports_df["PatientenID"].duplicated().sum()) if n_reports else 0
    if n_reports:
        pid = reports_df["PatientenID"].astype(str).str.strip()
        missing_pid_reports = int((reports_df["PatientenID"].isna() | (pid == "") | (pid.str.lower() == "nan")).sum())
    else:
        missing_pid_reports = 0

    rows.append(
        {
            "check": "diagnosis_patient_reports_row_count",
            "status": "ok" if n_reports > 0 else "warn",
            "value": str(n_reports),
            "detail": "patient-level rows from diagnosis input",
        }
    )
    rows.append(
        {
            "check": "diagnosis_patient_reports_duplicate_patientenid",
            "status": "ok" if dup_reports == 0 else "fail",
            "value": str(dup_reports),
            "detail": "must be 0",
        }
    )
    rows.append(
        {
            "check": "diagnosis_patient_reports_missing_patientenid",
            "status": "ok" if missing_pid_reports == 0 else "fail",
            "value": str(missing_pid_reports),
            "detail": "",
        }
    )

    diag_ids = set(reports_df["PatientenID"].tolist()) if n_reports else set()

    icd10_df = _load_icd_patients(ICD10_PATH, ("PatientenID", "PatientID"))
    icdsc_df = _load_icd_patients(ICDSC_PATH, ("PatientenID", "PatientID"))
    icd10_ids = set(icd10_df["PatientenID"].tolist()) if len(icd10_df) else set()
    icdsc_ids = set(icdsc_df["PatientenID"].tolist()) if len(icdsc_df) else set()

    rows.append(
        {
            "check": "icd10_file_exists",
            "status": "ok" if ICD10_PATH.exists() else "warn",
            "value": str(ICD10_PATH.exists()),
            "detail": str(ICD10_PATH),
        }
    )
    rows.append(
        {
            "check": "icdsc_file_exists",
            "status": "ok" if ICDSC_PATH.exists() else "warn",
            "value": str(ICDSC_PATH.exists()),
            "detail": str(ICDSC_PATH),
        }
    )

    only_diag = diag_ids - icd10_ids
    only_icd10 = icd10_ids - diag_ids
    only_icdsc = icdsc_ids - diag_ids
    triple = diag_ids & icd10_ids & icdsc_ids

    rows.append(
        {
            "check": "patient_id_set_size_diagnosis",
            "status": "ok",
            "value": str(len(diag_ids)),
            "detail": "",
        }
    )
    rows.append(
        {
            "check": "patient_id_set_size_icd10",
            "status": "ok",
            "value": str(len(icd10_ids)),
            "detail": "",
        }
    )
    rows.append(
        {
            "check": "patient_id_set_size_icdsc",
            "status": "ok",
            "value": str(len(icdsc_ids)),
            "detail": "",
        }
    )
    rows.append(
        {
            "check": "patient_ids_diagnosis_not_in_icd10",
            "status": "ok" if len(only_diag) == 0 else "warn",
            "value": str(len(only_diag)),
            "detail": ",".join(sorted(list(only_diag))[:20]) + ("..." if len(only_diag) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_icd10_not_in_diagnosis",
            "status": "ok" if len(only_icd10) == 0 else "warn",
            "value": str(len(only_icd10)),
            "detail": ",".join(sorted(list(only_icd10))[:20]) + ("..." if len(only_icd10) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_icdsc_not_in_diagnosis",
            "status": "ok" if len(only_icdsc) == 0 else "warn",
            "value": str(len(only_icdsc)),
            "detail": ",".join(sorted(list(only_icdsc))[:20]) + ("..." if len(only_icdsc) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_in_all_three_sources",
            "status": "ok",
            "value": str(len(triple)),
            "detail": "intersection diagnosis & icd10 & icdsc",
        }
    )

    if STRUCTURED_BASELINE_PATH.exists():
        base = pd.read_csv(STRUCTURED_BASELINE_PATH)
        base["PatientenID"] = _norm_pid(base["PatientenID"])
        if "baseline_reference_class" not in base.columns:
            base = add_reference_class(base)
        dist = base["baseline_reference_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "check": "structured_baseline_row_count",
                "status": "ok",
                "value": str(len(base)),
                "detail": str(STRUCTURED_BASELINE_PATH),
            }
        )
        rows.append(
            {
                "check": "structured_baseline_class_distribution",
                "status": "ok",
                "value": str(dist),
                "detail": "baseline_reference_class counts",
            }
        )
    else:
        rows.append(
            {
                "check": "structured_baseline_exists",
                "status": "warn",
                "value": "false",
                "detail": str(STRUCTURED_BASELINE_PATH),
            }
        )

    return rows


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(VALIDATION_RESULTS_CSV_PATH, index=False)
    lines = ["=== Pipeline data validation ===", ""]
    for r in rows:
        lines.append(f"[{r['status'].upper()}] {r['check']}: {r['value']}")
        if r.get("detail"):
            lines.append(f"    {r['detail']}")
    VALIDATION_SUMMARY_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = run_checks()
    write_outputs(rows)
    print(f"Validation CSV:  {VALIDATION_RESULTS_CSV_PATH}")
    print(f"Validation text: {VALIDATION_SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
