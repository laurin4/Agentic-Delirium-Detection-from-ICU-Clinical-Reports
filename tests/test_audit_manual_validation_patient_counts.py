"""Tests for manual validation patient count audit."""

import pandas as pd

from src.analysis.audit_manual_validation_patient_counts import audit_patient_counts


def _row(vpid: str, *, manual: int, v2=0, icdsc=0, icd10=0, comp_or=0, comp_and=0, complete=1):
    return {
        "validation_patient_id": vpid,
        "PatientenID": vpid.replace("Patient_", "H"),
        "is_patient_complete": complete,
        "derived_manual_patient_ground_truth": manual if complete else pd.NA,
        "model_patient_positive": v2,
        "baseline_icdsc_ge_4": icdsc,
        "baseline_icd10": icd10,
        "baseline_composite_or": comp_or,
        "baseline_composite_and": comp_and,
    }


def test_audit_shows_tier2_vs_tier3_gap():
    df = pd.DataFrame(
        [
            _row("Patient_0001", manual=1, v2=1, icdsc=1, comp_or=1),
            _row("Patient_0002", manual=0),
            _row("Patient_0003", manual=0, v2=pd.NA),
            _row("Patient_0004", manual="", complete=0),
        ]
    )
    report, dropped = audit_patient_counts(df)
    assert "complete_patients:               3" in report
    assert "patients_with_all_signals:       2" in report
    assert "dropped_from_baseline_summary:   1" in report
    assert len(dropped) == 1
    assert dropped.iloc[0]["validation_patient_id"] == "Patient_0003"
