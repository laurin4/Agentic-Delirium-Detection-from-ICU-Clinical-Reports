"""Tests for baseline vs manual comparison summary export."""

import pandas as pd

from src.analysis.export_baseline_manual_comparison_summary import (
    build_baseline_manual_comparison_summary,
    export_baseline_manual_comparison_summary,
    resolve_comparison_columns,
)


def _patient_row(
    vpid: str,
    *,
    manual: int,
    icdsc: int = 0,
    icd10: int = 0,
    comp_or: int = 0,
    comp_and: int = 0,
    v2: int = 0,
    hospital_id: str = "",
) -> dict:
    return {
        "validation_patient_id": vpid,
        "PatientenID": hospital_id or vpid.replace("Patient_", "H"),
        "n_reports_total": 2,
        "n_reports_labeled": 2,
        "n_reports_missing_label": 0,
        "is_patient_complete": 1,
        "n_positive_reports_manual": manual,
        "derived_manual_patient_ground_truth": manual,
        "model_patient_positive": v2,
        "baseline_icdsc_ge_4": icdsc,
        "baseline_icd10": icd10,
        "baseline_composite_or": comp_or,
        "baseline_composite_and": comp_and,
    }


def test_resolve_comparison_columns_detects_expected_names():
    df = pd.DataFrame([_patient_row("Patient_0001", manual=1)])
    resolved, errors = resolve_comparison_columns(df)
    assert errors == []
    assert resolved is not None
    assert resolved.manual == "derived_manual_patient_ground_truth"
    assert resolved.v2 == "model_patient_positive"
    assert resolved.icdsc == "baseline_icdsc_ge_4"


def test_summary_categories_and_counts(tmp_path):
    df = pd.DataFrame(
        [
            # manual+, ICDSC miss
            _patient_row("Patient_0001", manual=1, icdsc=0, icd10=1, comp_or=1, v2=1),
            # manual+, ICD10 miss
            _patient_row("Patient_0002", manual=1, icdsc=1, icd10=0, comp_or=1, v2=0),
            # manual+, V2+, all baselines negative
            _patient_row("Patient_0003", manual=1, v2=1),
            # ICDSC false positive
            _patient_row("Patient_0004", manual=0, icdsc=1, comp_or=1),
            # ICD10 false positive
            _patient_row("Patient_0005", manual=0, icd10=1, comp_or=1),
            # incomplete — excluded
            {
                **_patient_row("Patient_0006", manual=1, v2=1),
                "is_patient_complete": 0,
                "n_reports_missing_label": 1,
            },
        ]
    )
    text = build_baseline_manual_comparison_summary(df, prompt_version="v2")
    assert "Manual positives missed by ICDSC" in text
    assert "Patient_0001" in text
    assert "Patient_0003" in text
    assert "Count: 1" in text
    assert "Manual positives found by V2 but missed by all baseline rules" in text
    assert "False positives from ICDSC" in text
    assert "Patient_0004" in text
    assert "False positives from ICD10" in text
    assert "Patient_0005" in text
    assert "Patient_0006" not in text

    out = tmp_path / "baseline_manual_comparison_summary.txt"
    export_baseline_manual_comparison_summary(df, output_path=out)
    assert out.exists()
    assert out.name == "baseline_manual_comparison_summary.txt"


def test_summary_reports_missing_columns():
    df = pd.DataFrame({"validation_patient_id": ["Patient_0001"]})
    text = build_baseline_manual_comparison_summary(df)
    assert "COLUMN MAPPING FAILED" in text
    assert "Available columns:" in text
