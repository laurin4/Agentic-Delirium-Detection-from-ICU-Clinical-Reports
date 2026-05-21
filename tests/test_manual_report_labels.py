"""Tests for slim manual_report_labels export and merge into cohort evaluation."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.evaluate_manual_validation import main as eval_main
from src.analysis.export_manual_report_labels import main as export_labels_main
from src.analysis.manual_report_labels import (
    build_manual_report_labels_sheet,
    load_cohort_for_manual_evaluation,
    merge_manual_report_labels,
)
from src.analysis.manual_validation_eval import evaluate_annotated_cohort


def _cohort_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0001", "Patient_0002"],
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0001_Report_0002",
                "Patient_0002_Report_0001",
            ],
            "report_nr_within_patient": [1, 2, 1],
            "PatientenID": ["p1", "p1", "p2"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02", "2024-02-01"],
            "model_report_prediction": [1, 0, 0],
            "evidence_snippets": ['[{"text": "x"}]', "[]", "[]"],
            "manual_report_ground_truth": ["", "", ""],
            "manual_comment": ["", "", ""],
            "model_patient_positive": [1, 1, 0],
            "baseline_icdsc_ge_4": [0, 0, 0],
            "baseline_icd10": [0, 0, 0],
        }
    )


def test_build_manual_report_labels_empty_manual_fields():
    from src.analysis.manual_report_labels import SLIM_LABEL_EXPORT_COLUMNS

    sheet = build_manual_report_labels_sheet(_cohort_df())
    assert len(sheet) == 3
    for col in (
        "validation_patient_id",
        "validation_report_id",
        "model_report_prediction",
        "manual_report_ground_truth",
    ):
        assert col in sheet.columns
    assert (sheet["manual_report_ground_truth"].astype(str).str.strip() == "").all()


def test_merge_manual_report_labels_updates_cohort():
    cohort = _cohort_df()
    labels = pd.DataFrame(
        {
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0001_Report_0002",
            ],
            "manual_report_ground_truth": [1, 0],
            "manual_comment": ["clear delir", ""],
        }
    )
    merged = merge_manual_report_labels(cohort, labels)
    assert str(merged.loc[0, "manual_report_ground_truth"]) == "1"
    assert str(merged.loc[1, "manual_report_ground_truth"]) == "0"
    assert str(merged.loc[2, "manual_report_ground_truth"]).strip() in ("", "nan")
    assert merged.loc[0, "manual_comment"] == "clear delir"


def test_merge_rejects_duplicate_validation_report_id():
    cohort = _cohort_df()
    labels = pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0001_Report_0001"],
            "manual_report_ground_truth": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        merge_manual_report_labels(cohort, labels)


def test_load_cohort_merges_labels_file(tmp_path):
    cohort_path = tmp_path / "cohort.csv"
    labels_path = tmp_path / "manual_report_labels.csv"
    _cohort_df().to_csv(cohort_path, index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "manual_report_ground_truth": [1, 0],
            "manual_comment": ["", "tn"],
        }
    ).to_csv(labels_path, index=False)

    loaded = load_cohort_for_manual_evaluation(
        cohort_path,
        labels_path=labels_path,
        auto_merge_default_labels=False,
    )
    assert str(loaded.loc[0, "manual_report_ground_truth"]) == "1"
    assert str(loaded.loc[2, "manual_report_ground_truth"]) == "0"


def test_export_and_evaluate_with_labels(tmp_path):
    cohort_path = tmp_path / "patient_validation_cohort.csv"
    labels_path = tmp_path / "manual_report_labels.csv"
    eval_dir = tmp_path / "evaluation"
    _cohort_df().to_csv(cohort_path, index=False)

    export_labels_main(cohort_path=cohort_path, output_path=labels_path)
    labels = pd.read_csv(labels_path)
    labels.loc[labels["validation_report_id"] == "Patient_0001_Report_0001", "manual_report_ground_truth"] = 1
    labels.loc[labels["validation_report_id"] == "Patient_0002_Report_0001", "manual_report_ground_truth"] = 0
    labels.to_csv(labels_path, index=False)

    # Cohort still unannotated; evaluation uses labels merge
    eval_main(cohort_path=cohort_path, labels_path=labels_path, output_dir=eval_dir)
    assert (eval_dir / "tables" / "metrics_summary.csv").exists()

    merged = load_cohort_for_manual_evaluation(
        cohort_path, labels_path=labels_path, auto_merge_default_labels=False
    )
    summary, _ = evaluate_annotated_cohort(merged, eval_dir / "run2")
    assert not summary.empty
    assert int(summary.loc[summary["level"] == "report", "n"].iloc[0]) == 2
