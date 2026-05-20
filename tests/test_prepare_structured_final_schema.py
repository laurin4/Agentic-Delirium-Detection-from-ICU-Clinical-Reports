"""Tests for final raw data structure (ICD.csv + ICDSC.csv only)."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.prepare_structured_data import (
    add_binary_baselines,
    build_structured_baseline,
    load_data,
    prepare_icd10,
    prepare_icdsc,
)
from src.pipeline.schema_normalize import (
    SchemaValidationError,
    assert_structured_baseline_columns,
    is_valid_delir_icd10_code,
    structured_baseline_output_columns,
)


def test_baseline_composite_icdsc_ge4_or_icd10(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "OR")
    df = pd.DataFrame({"has_delir_icd10": [0, 1, 0], "max_icdsc": [0, 2, 5]})
    out = add_binary_baselines(df)
    assert list(out["baseline_composite"]) == [0, 1, 1]


def test_icdsc_patient_level_icdsc_max_and_thresholds():
    df = pd.DataFrame({"PatientID": ["p1", "p2", "p3"], "ICDSC_Max": [0, 3, 5]})
    out = prepare_icdsc(df).sort_values("PatientenID").reset_index(drop=True)
    assert list(out["PatientenID"]) == ["p1", "p2", "p3"]
    assert list(out["max_icdsc"]) == [0, 3, 5]

    merged = out.merge(
        pd.DataFrame({"PatientenID": ["p1", "p2", "p3"], "has_delir_icd10": [0, 0, 0]}),
        on="PatientenID",
    )
    bb = add_binary_baselines(merged)
    row0 = bb.loc[bb["PatientenID"] == "p1"].iloc[0]
    row2 = bb.loc[bb["PatientenID"] == "p2"].iloc[0]
    row3 = bb.loc[bb["PatientenID"] == "p3"].iloc[0]
    assert row0["baseline_icdsc_0"] == 1
    assert row0["baseline_icdsc_ge_4"] == 0
    assert row2["baseline_icdsc_1_to_3"] == 1
    assert row2["baseline_icdsc_ge_4_grouped"] == 0
    assert row3["baseline_icdsc_ge_4"] == 1
    assert row3["baseline_icdsc_ge_4_grouped"] == 1


def test_icd_only_main_diagnosis_counts_delir():
    df = pd.DataFrame(
        {
            "PatientID": ["p1", "p1", "p2", "p3"],
            "icd_hd": [1, 0, 1, 1],
            "icd_code": ["F05.0", "F05.0", "F05.8", "I10"],
        }
    )
    out = prepare_icd10(df).sort_values("PatientenID").reset_index(drop=True)
    assert int(out.loc[out["PatientenID"] == "p1", "has_delir_icd10"].iloc[0]) == 1
    assert int(out.loc[out["PatientenID"] == "p2", "has_delir_icd10"].iloc[0]) == 1
    assert int(out.loc[out["PatientenID"] == "p3", "has_delir_icd10"].iloc[0]) == 0


def test_icd_valid_delir_codes_thesis_definition():
    assert is_valid_delir_icd10_code("F05.0")
    assert is_valid_delir_icd10_code("f05.8")
    assert is_valid_delir_icd10_code("F05.9")
    assert not is_valid_delir_icd10_code("F05.1")
    assert not is_valid_delir_icd10_code("F05")
    assert not is_valid_delir_icd10_code("F05.2")
    assert not is_valid_delir_icd10_code("I10")


@pytest.mark.parametrize(
    "icd_code,expected",
    [
        ("F05.0", 1),
        ("F05.8", 1),
        ("F05.9", 1),
        ("F05.1", 0),
        ("F05.2", 0),
        ("F05.5", 0),
    ],
)
def test_icd10_main_diagnosis_thesis_codes(icd_code, expected):
    """Thesis ICD-10 delir: icd_hd==1 and F05.0 / F05.8 / F05.9 only."""
    df = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": [icd_code]})
    out = prepare_icd10(df)
    assert int(out.loc[0, "has_delir_icd10"]) == expected


def test_icd_f051_excluded_alcohol_delir():
    """F05.1 (alcohol-related delirium) is outside the intended cohort."""
    df = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": ["F05.1"]})
    out = prepare_icd10(df)
    assert int(out.loc[0, "has_delir_icd10"]) == 0


def test_icd_non_f05_excluded_even_as_main():
    df = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": ["I10"]})
    out = prepare_icd10(df)
    assert int(out.loc[0, "has_delir_icd10"]) == 0


def test_icd_f05_not_counted_when_not_main_diagnosis():
    df = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [0], "icd_code": ["F05.1"]})
    out = prepare_icd10(df)
    assert int(out.loc[0, "has_delir_icd10"]) == 0


def test_icd_f050_not_counted_when_icd_hd_not_main():
    df = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [0], "icd_code": ["F05.0"]})
    out = prepare_icd10(df)
    assert int(out.loc[0, "has_delir_icd10"]) == 0


def test_build_structured_baseline_standard_columns():
    icd = pd.DataFrame({"PatientID": ["p1"], "icd_hd": [0], "icd_code": ["I10"]})
    icdsc = pd.DataFrame({"PatientID": ["p1"], "ICDSC_Max": [2]})
    merged = build_structured_baseline(icd, icdsc)
    assert tuple(structured_baseline_output_columns()) == tuple(
        c for c in structured_baseline_output_columns() if c in merged.columns
    )
    assert_structured_baseline_columns(merged)


def test_load_data_no_diagnosenliste_required(monkeypatch, tmp_path):
    icd_path = tmp_path / "ICD.csv"
    icdsc_path = tmp_path / "ICDSC.csv"
    pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": ["F05.0"]}).to_csv(
        icd_path, index=False, sep=";"
    )
    pd.DataFrame({"PatientID": ["p1"], "ICDSC_Max": [4]}).to_csv(icdsc_path, index=False, sep=";")

    import src.pipeline.prepare_structured_data as psd

    monkeypatch.setattr(psd, "ICD10_PATH", icd_path)
    monkeypatch.setattr(psd, "ICDSC_PATH", icdsc_path)
    icd10, icdsc = load_data()
    assert "PatientenID" in icd10.columns
    assert "ICDSC_Max" in icdsc.columns
    assert not (tmp_path / "Diagnosenliste.csv").exists()


def test_icdsc_all_non_numeric_raises():
    df = pd.DataFrame({"PatientID": ["p1"], "ICDSC_Max": ["n/a"]})
    with pytest.raises(SchemaValidationError):
        prepare_icdsc(df)


def test_icdsc_duplicate_patient_takes_max():
    df = pd.DataFrame({"PatientID": ["p1", "p1"], "ICDSC_Max": [2, 6]})
    out = prepare_icdsc(df)
    assert float(out.loc[0, "max_icdsc"]) == 6.0
