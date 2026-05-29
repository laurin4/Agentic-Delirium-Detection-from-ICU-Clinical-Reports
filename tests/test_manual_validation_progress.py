"""Tests for manual validation progress aggregation."""

import pandas as pd

from src.analysis.build_manual_validation_progress import (
    assign_confusion_group,
    build_manual_validation_progress,
    format_progress_report,
)


def _cohort_row(
    vpid: str,
    rid: str,
    gt: object,
    model_report: int = 0,
    model_patient: int = 0,
) -> dict:
    return {
        "validation_patient_id": vpid,
        "validation_report_id": rid,
        "PatientenID": vpid.replace("Patient_", "P"),
        "manual_report_ground_truth": gt,
        "model_report_prediction": model_report,
        "model_patient_positive": model_patient,
    }


def test_incomplete_patient_derived_gt_is_na():
    df = pd.DataFrame(
        [
            _cohort_row("Patient_0001", "Patient_0001_Report_0001", 1),
            _cohort_row("Patient_0001", "Patient_0001_Report_0002", ""),
        ]
    )
    prog = build_manual_validation_progress(df)
    row = prog.loc[prog["validation_patient_id"] == "Patient_0001"].iloc[0]
    assert row["is_patient_complete"] == 0
    assert pd.isna(row["derived_manual_patient_ground_truth"])
    assert row["confusion_group"] == ""


def test_complete_negative_patient_derived_gt_is_zero():
    df = pd.DataFrame(
        [
            _cohort_row("Patient_0002", "Patient_0002_Report_0001", 0, model_report=0),
            _cohort_row("Patient_0002", "Patient_0002_Report_0002", 0, model_report=0),
        ]
    )
    prog = build_manual_validation_progress(df)
    row = prog.loc[prog["validation_patient_id"] == "Patient_0002"].iloc[0]
    assert row["is_patient_complete"] == 1
    assert row["derived_manual_patient_ground_truth"] == 0
    assert row["n_positive_reports_manual"] == 0


def test_any_positive_report_makes_patient_gt_one():
    df = pd.DataFrame(
        [
            _cohort_row("Patient_0003", "Patient_0003_Report_0001", 0),
            _cohort_row("Patient_0003", "Patient_0003_Report_0002", 1),
        ]
    )
    prog = build_manual_validation_progress(df)
    row = prog.loc[prog["validation_patient_id"] == "Patient_0003"].iloc[0]
    assert row["is_patient_complete"] == 1
    assert row["derived_manual_patient_ground_truth"] == 1
    assert row["n_positive_reports_manual"] == 1


def test_confusion_groups_when_model_available():
    assert assign_confusion_group(1, 1) == "TP"
    assert assign_confusion_group(1, 0) == "FP"
    assert assign_confusion_group(0, 0) == "TN"
    assert assign_confusion_group(0, 1) == "FN"
    assert assign_confusion_group(1, pd.NA) == ""
    assert assign_confusion_group(pd.NA, 0) == ""


def test_progress_confusion_from_cohort():
    df = pd.DataFrame(
        [
            _cohort_row(
                "Patient_0004",
                "Patient_0004_Report_0001",
                1,
                model_report=1,
                model_patient=1,
            ),
            _cohort_row(
                "Patient_0005",
                "Patient_0005_Report_0001",
                0,
                model_report=1,
                model_patient=1,
            ),
            _cohort_row(
                "Patient_0006",
                "Patient_0006_Report_0001",
                0,
                model_report=0,
                model_patient=0,
            ),
            _cohort_row(
                "Patient_0007",
                "Patient_0007_Report_0001",
                1,
                model_report=0,
                model_patient=0,
            ),
        ]
    )
    prog = build_manual_validation_progress(df)
    by_id = prog.set_index("validation_patient_id")["confusion_group"].to_dict()
    assert by_id["Patient_0004"] == "TP"
    assert by_id["Patient_0005"] == "FP"
    assert by_id["Patient_0006"] == "TN"
    assert by_id["Patient_0007"] == "FN"


def test_empty_manual_label_not_treated_as_zero():
    df = pd.DataFrame(
        [
            _cohort_row("Patient_0008", "Patient_0008_Report_0001", ""),
            _cohort_row("Patient_0008", "Patient_0008_Report_0002", 0),
        ]
    )
    prog = build_manual_validation_progress(df)
    row = prog.loc[prog["validation_patient_id"] == "Patient_0008"].iloc[0]
    assert row["n_reports_labeled"] == 1
    assert row["n_reports_missing_label"] == 1
    assert row["is_patient_complete"] == 0
    assert pd.isna(row["derived_manual_patient_ground_truth"])


def test_format_progress_report_counts():
    prog = pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "PatientenID": ["a", "b"],
            "n_reports_total": [2, 1],
            "n_reports_labeled": [2, 1],
            "n_reports_missing_label": [0, 0],
            "is_patient_complete": [1, 1],
            "n_positive_reports_manual": [1, 0],
            "derived_manual_patient_ground_truth": [1, 0],
            "model_patient_positive": [1, 0],
            "confusion_group": ["TP", "TN"],
        }
    )
    report = format_progress_report(prog)
    assert "complete_patients=2" in report
    assert "manual_positive_patients=1" in report
    assert "TP=1" in report
    assert "TN=1" in report


def test_merge_labels_overrides_cohort_gt(tmp_path):
    from src.analysis.manual_report_labels import merge_manual_report_labels

    cohort = pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "validation_report_id": ["Patient_0001_Report_0001"],
            "manual_report_ground_truth": [""],
            "model_report_prediction": [0],
        }
    )
    labels = pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001"],
            "manual_report_ground_truth": [1],
        }
    )
    merged = merge_manual_report_labels(cohort, labels)
    prog = build_manual_validation_progress(merged)
    assert prog.iloc[0]["derived_manual_patient_ground_truth"] == 1
