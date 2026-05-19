"""Tests for structured baseline source schema normalization."""

import pandas as pd
import pytest

from src.pipeline.prepare_structured_data import prepare_icd10, prepare_icdsc
from src.pipeline.schema_normalize import (
    SchemaValidationError,
    is_valid_delir_icd10_code,
    normalize_icd10_source_columns,
    normalize_icdsc_source_columns,
    normalize_patient_id_columns,
    require_columns,
)


def test_normalize_patient_id_from_patientid():
    df = pd.DataFrame({"PatientID": [" 1001 ", "1002"], "icd_code": ["F05.0", "I10"]})
    out = normalize_patient_id_columns(df)
    assert "PatientID" not in out.columns
    assert list(out["PatientenID"]) == ["1001", "1002"]


def test_normalize_patient_id_keeps_patientenid():
    df = pd.DataFrame({"PatientenID": ["p1"], "x": [1]})
    out = normalize_patient_id_columns(df)
    assert list(out["PatientenID"]) == ["p1"]


def test_normalize_icdsc_icdsc_max_preserved():
    df = pd.DataFrame({"PatientID": ["1"], "ICDSC_Max": [4]})
    out = normalize_icdsc_source_columns(df)
    assert "ICDSC_Max" in out.columns
    assert out.loc[0, "ICDSC_Max"] == 4


def test_normalize_icd10_icd_code_and_icd_hd():
    df = pd.DataFrame(
        {
            "PatientID": [1001, 1001],
            "icd_code": ["F05.0", "I10"],
            "icd_hd": ["1", "0"],
        }
    )
    out = normalize_icd10_source_columns(normalize_patient_id_columns(df))
    assert "icd_code" not in out.columns
    assert "icd_hd" not in out.columns
    result = prepare_icd10(out)
    row = result.loc[result["PatientenID"] == "1001"].iloc[0]
    assert row["has_delir_icd10"] == 1


def test_prepare_icdsc_with_patientid_and_icdsc_max():
    df = pd.DataFrame(
        {
            "PatientID": [1001, 1002],
            "ICDSC_Max": [5, 3],
        }
    )
    out = normalize_icdsc_source_columns(normalize_patient_id_columns(df))
    result = prepare_icdsc(out).sort_values("PatientenID").reset_index(drop=True)
    assert result.loc[0, "PatientenID"] == "1001"
    assert result.loc[0, "max_icdsc"] == 5
    assert result.loc[1, "PatientenID"] == "1002"
    assert result.loc[1, "max_icdsc"] == 3


def test_prepare_icd10_legacy_code_is_hauptdiagn():
    df = pd.DataFrame(
        {
            "PatientenID": [1001, 1002],
            "Code": ["F05.0", "J44.1"],
            "IsHauptDiagn": ["1", "0"],
        }
    )
    result = prepare_icd10(df)
    assert result.loc[0, "has_delir_icd10"] == 1
    assert result.loc[1, "has_delir_icd10"] == 0


def test_f051_excluded():
    assert not is_valid_delir_icd10_code("F05.1")


def test_require_columns_lists_available():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(SchemaValidationError) as exc:
        require_columns(df, ("PatientenID", "Code"), context="ICD input")
    msg = str(exc.value)
    assert "PatientenID" in msg
    assert "Available columns" in msg


def test_compare_load_normalizes_patientid_baseline(tmp_path):
    from src.pipeline.compare_reports_vs_baseline import load_data

    baseline = pd.DataFrame(
        {
            "PatientID": ["p1"],
            "has_delir_icd10": [0],
            "max_icdsc": [2],
            "baseline_icd10": [0],
            "baseline_icdsc_ge_1": [1],
            "baseline_icdsc_ge_2": [1],
            "baseline_icdsc_ge_3": [0],
            "baseline_icdsc_ge_4": [0],
            "baseline_icdsc_ge_5": [0],
            "baseline_icdsc_0": [0],
            "baseline_icdsc_1_to_3": [1],
            "baseline_icdsc_ge_4_grouped": [0],
        }
    )
    pred = pd.DataFrame({"PatientenID": ["p1"], "klasse": [0]})
    bpath = tmp_path / "baseline.csv"
    ppath = tmp_path / "pred.csv"
    baseline.to_csv(bpath, index=False)
    pred.to_csv(ppath, index=False)

    b, r = load_data(bpath, ppath)
    assert "PatientenID" in b.columns
    assert "PatientID" not in b.columns
    assert list(b["PatientenID"]) == ["p1"]
