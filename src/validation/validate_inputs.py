"""
Deterministic validation of Berichte, ICD, ICDSC, and optional baseline artifacts.

Reads paths only from src.pipeline.paths (DATA_MODE selects real CSV files vs optional synthetic CSVs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.pipeline.tabular_io import read_tabular
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    ICD10_PATH,
    ICDSC_PATH,
    STRUCTURED_BASELINE_PATH,
    VALIDATION_DIR,
    VALIDATION_RESULTS_CSV_PATH,
    VALIDATION_SUMMARY_TXT_PATH,
)
from src.pipeline.prepare_structured_data import add_reference_class, add_binary_baselines
from src.pipeline.schema_normalize import normalize_patient_id_columns
from src.preprocessing.berichte_mapper import build_patient_level_berichte_report_records

LOGGER = logging.getLogger(__name__)


def _norm_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _load_patient_ids(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["PatientenID"])
    df = read_tabular(path)
    df = normalize_patient_id_columns(df)
    if "PatientenID" not in df.columns:
        return pd.DataFrame(columns=["PatientenID"])
    out = df[["PatientenID"]].copy()
    out["PatientenID"] = _norm_pid(out["PatientenID"])
    return out


def _berichte_patient_reports() -> pd.DataFrame:
    """Patient-level rows from Berichte.csv (same as production pipeline)."""
    if not BERICHTE_INPUT_PATH.exists():
        return pd.DataFrame(columns=["PatientenID", "bericht", "report_text"])
    records = build_patient_level_berichte_report_records()
    return pd.DataFrame(records)


def run_checks() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    reports_df = _berichte_patient_reports()
    n_reports = len(reports_df)
    dup_reports = int(reports_df["PatientenID"].duplicated().sum()) if n_reports else 0
    if n_reports:
        pid = reports_df["PatientenID"].astype(str).str.strip()
        missing_pid_reports = int((reports_df["PatientenID"].isna() | (pid == "") | (pid.str.lower() == "nan")).sum())
    else:
        missing_pid_reports = 0

    rows.append(
        {
            "check": "berichte_patient_reports_row_count",
            "status": "ok" if n_reports > 0 else "warn",
            "value": str(n_reports),
            "detail": "patient-level rows from Berichte.csv",
        }
    )
    rows.append(
        {
            "check": "berichte_patient_reports_duplicate_patientenid",
            "status": "ok" if dup_reports == 0 else "fail",
            "value": str(dup_reports),
            "detail": "must be 0",
        }
    )
    rows.append(
        {
            "check": "berichte_patient_reports_missing_patientenid",
            "status": "ok" if missing_pid_reports == 0 else "fail",
            "value": str(missing_pid_reports),
            "detail": "",
        }
    )

    berichte_ids = set(reports_df["PatientenID"].tolist()) if n_reports else set()

    icd10_df = _load_patient_ids(ICD10_PATH, "ICD.csv")
    icdsc_df = _load_patient_ids(ICDSC_PATH, "ICDSC.csv")
    icd10_ids = set(icd10_df["PatientenID"].tolist()) if len(icd10_df) else set()
    icdsc_ids = set(icdsc_df["PatientenID"].tolist()) if len(icdsc_df) else set()

    rows.append(
        {
            "check": "berichte_file_exists",
            "status": "ok" if BERICHTE_INPUT_PATH.exists() else "warn",
            "value": str(BERICHTE_INPUT_PATH.exists()),
            "detail": str(BERICHTE_INPUT_PATH),
        }
    )
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

    only_berichte = berichte_ids - icd10_ids
    only_icd10 = icd10_ids - berichte_ids
    only_icdsc = icdsc_ids - berichte_ids
    triple = berichte_ids & icd10_ids & icdsc_ids

    rows.append(
        {
            "check": "patient_id_set_size_berichte",
            "status": "ok",
            "value": str(len(berichte_ids)),
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
            "check": "patient_ids_berichte_not_in_icd10",
            "status": "ok" if len(only_berichte) == 0 else "warn",
            "value": str(len(only_berichte)),
            "detail": ",".join(sorted(list(only_berichte))[:20]) + ("..." if len(only_berichte) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_icd10_not_in_berichte",
            "status": "ok" if len(only_icd10) == 0 else "warn",
            "value": str(len(only_icd10)),
            "detail": ",".join(sorted(list(only_icd10))[:20]) + ("..." if len(only_icd10) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_icdsc_not_in_berichte",
            "status": "ok" if len(only_icdsc) == 0 else "warn",
            "value": str(len(only_icdsc)),
            "detail": ",".join(sorted(list(only_icdsc))[:20]) + ("..." if len(only_icdsc) > 20 else ""),
        }
    )
    rows.append(
        {
            "check": "patient_ids_in_berichte_icd10_icdsc",
            "status": "ok",
            "value": str(len(triple)),
            "detail": "intersection Berichte & ICD & ICDSC",
        }
    )

    if STRUCTURED_BASELINE_PATH.exists():
        base = pd.read_csv(STRUCTURED_BASELINE_PATH)
        base = normalize_patient_id_columns(base)
        base["PatientenID"] = _norm_pid(base["PatientenID"])
        if "baseline_reference_class" not in base.columns:
            base = add_reference_class(base)
        dist = base["baseline_reference_class"].value_counts().sort_index().to_dict()
        bb = add_binary_baselines(base.copy())
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
                "check": "structured_baseline_binary_icd10_positive_patients",
                "status": "ok",
                "value": str(int((bb["baseline_icd10"] == 1).sum())) if "baseline_icd10" in bb.columns else "n/a",
                "detail": "baseline_icd10==1 (main diagnosis F05.0/F05.8/F05.9)",
            }
        )
        rows.append(
            {
                "check": "structured_baseline_binary_icdsc_ge_4_positive_patients",
                "status": "ok",
                "value": str(int((bb["baseline_icdsc_ge_4"] == 1).sum())) if "baseline_icdsc_ge_4" in bb.columns else "n/a",
                "detail": "baseline_icdsc_ge_4==1 (ICDSC_Max >= 4)",
            }
        )
        rows.append(
            {
                "check": "structured_baseline_legacy_reference_class_distribution",
                "status": "ok",
                "value": str(dist),
                "detail": "LEGACY multiclass baseline_reference_class — not primary evaluation",
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
