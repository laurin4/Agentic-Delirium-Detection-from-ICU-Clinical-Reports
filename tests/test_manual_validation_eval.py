"""Tests for manual validation evaluation."""

import pandas as pd

from src.analysis.evaluate_manual_validation import main as eval_main
from src.analysis.manual_validation_eval import (
    binary_metrics,
    evaluate_annotated_cohort,
)


def test_binary_metrics_perfect():
    m = binary_metrics(pd.Series([1, 0, 1]), pd.Series([1, 0, 1]))
    assert m["tp"] == 2 and m["tn"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["f1"] == 1.0


def test_report_and_patient_level_evaluation(tmp_path):
    df = pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0001", "Patient_0002"],
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0001_Report_0002", "Patient_0002_Report_0001"],
            "PatientenID": ["a", "a", "b"],
            "manual_report_ground_truth": [1, 0, 0],
            "model_report_prediction": [1, 0, 0],
            "model_patient_positive": [1, 1, 0],
            "baseline_icdsc_ge_4": [1, 1, 0],
            "baseline_icd10": [0, 0, 0],
        }
    )
    out_dir = tmp_path / "eval"
    summary, report = evaluate_annotated_cohort(df, out_dir)
    assert not summary.empty
    assert (out_dir / "tables" / "metrics_summary.csv").exists()
    assert "patient-level" in report.lower() or "Patient-level" in report


def test_evaluate_main(tmp_path):
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "validation_report_id": ["Patient_0001_Report_0001"],
            "manual_report_ground_truth": [1],
            "model_report_prediction": [1],
            "model_patient_positive": [1],
            "baseline_icdsc_ge_4": [0],
            "baseline_icd10": [0],
        }
    ).to_csv(cohort, index=False)
    out = tmp_path / "evaluation"
    eval_main(cohort_path=cohort, output_dir=out)
    assert (out / "evaluation_report.txt").exists()
