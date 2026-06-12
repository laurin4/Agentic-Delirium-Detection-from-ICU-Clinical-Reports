"""Tests for final manual validation evaluation."""

import pandas as pd
import pytest

from src.analysis.final_manual_validation_evaluation import (
    ERROR_EXPORT_COLUMNS,
    attach_structured_baseline,
    build_patient_level_ground_truth,
    compute_method_metrics,
    derive_composite_baselines,
    export_model_error_slices,
    load_merged_frozen_cohort,
    primary_evaluation_cohort,
    run_final_evaluation,
)


def _report_row(
    vpid: str,
    rid: str,
    gt: object,
    *,
    model_report: int = 0,
    model_patient: int = 0,
    icdsc: int = 0,
    icd10: int = 0,
    comment: str = "",
    evidence: str = "",
) -> dict:
    return {
        "validation_patient_id": vpid,
        "validation_report_id": rid,
        "PatientenID": vpid.replace("Patient_", "P"),
        "manual_report_ground_truth": gt,
        "model_report_prediction": model_report,
        "model_patient_positive": model_patient,
        "baseline_icdsc_ge_4": icdsc,
        "baseline_icd10": icd10,
        "manual_comment": comment,
        "evidence_snippets": evidence,
    }


def test_incomplete_patients_excluded_from_primary():
    df = pd.DataFrame(
        [
            _report_row("Patient_0001", "Patient_0001_Report_0001", 1),
            _report_row("Patient_0001", "Patient_0001_Report_0002", ""),
            _report_row("Patient_0002", "Patient_0002_Report_0001", 0, model_patient=0),
        ]
    )
    gt = build_patient_level_ground_truth(df)
    primary = primary_evaluation_cohort(gt)
    assert len(primary) == 1
    assert primary.iloc[0]["validation_patient_id"] == "Patient_0002"


def test_empty_manual_labels_not_treated_as_zero():
    df = pd.DataFrame(
        [
            _report_row("Patient_0001", "Patient_0001_Report_0001", ""),
            _report_row("Patient_0001", "Patient_0001_Report_0002", 0),
        ]
    )
    gt = build_patient_level_ground_truth(df)
    row = gt.iloc[0]
    assert row["n_reports_labeled"] == 1
    assert row["is_patient_complete"] == 0
    assert pd.isna(row["derived_manual_patient_ground_truth"])


def test_stale_frozen_model_patient_positive_does_not_override_report_predictions():
    """Frozen cohort may carry old model_patient_positive; evaluation uses max(model_report_prediction)."""
    df = pd.DataFrame(
        [
            _report_row("Patient_0019", "Patient_0019_Report_0001", 0, model_report=0, model_patient=1),
            _report_row("Patient_0019", "Patient_0019_Report_0002", 0, model_report=0, model_patient=1),
        ]
    )
    gt = build_patient_level_ground_truth(df)
    assert int(gt.iloc[0]["model_patient_positive"]) == 0


def test_derived_manual_patient_gt_rules():
    complete_neg = pd.DataFrame(
        [
            _report_row("Patient_0001", "Patient_0001_Report_0001", 0),
            _report_row("Patient_0001", "Patient_0001_Report_0002", 0),
        ]
    )
    gt_neg = build_patient_level_ground_truth(complete_neg).iloc[0]
    assert gt_neg["derived_manual_patient_ground_truth"] == 0
    assert gt_neg["n_positive_reports_manual"] == 0

    complete_pos = pd.DataFrame(
        [
            _report_row("Patient_0002", "Patient_0002_Report_0001", 0),
            _report_row("Patient_0002", "Patient_0002_Report_0002", 1),
        ]
    )
    gt_pos = build_patient_level_ground_truth(complete_pos).iloc[0]
    assert gt_pos["derived_manual_patient_ground_truth"] == 1
    assert gt_pos["n_positive_reports_manual"] == 1


def test_derive_composite_or_and_when_missing():
    df = pd.DataFrame(
        {
            "baseline_icdsc_ge_4": [1, 0, 1],
            "baseline_icd10": [0, 1, 1],
        }
    )
    out = derive_composite_baselines(df)
    assert list(out["baseline_composite_or"]) == [1, 1, 1]
    assert list(out["baseline_composite_and"]) == [0, 0, 1]


def test_derive_composite_preserves_existing_values():
    df = pd.DataFrame(
        {
            "baseline_icdsc_ge_4": [1, 0],
            "baseline_icd10": [0, 1],
            "baseline_composite_or": [0, pd.NA],
            "baseline_composite_and": [1, pd.NA],
        }
    )
    out = derive_composite_baselines(df)
    assert out.iloc[0]["baseline_composite_or"] == 0
    assert out.iloc[0]["baseline_composite_and"] == 1
    assert out.iloc[1]["baseline_composite_or"] == 1
    assert out.iloc[1]["baseline_composite_and"] == 0


def test_metrics_computed_correctly():
    manual = pd.Series([1, 0, 1, 0])
    pred = pd.Series([1, 1, 0, 0])
    m = compute_method_metrics(manual, pred, method_name="model")
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["fn"] == 1
    assert m["sensitivity"] == 0.5
    assert m["specificity"] == 0.5
    assert m["precision"] == 0.5
    assert m["npv"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_run_final_evaluation_outputs(tmp_path):
    df = pd.DataFrame(
        [
            _report_row(
                "Patient_0001",
                "Patient_0001_Report_0001",
                1,
                model_report=1,
                model_patient=1,
                icdsc=1,
                icd10=0,
                comment="clear delirium",
                evidence="confusion noted",
            ),
            _report_row(
                "Patient_0002",
                "Patient_0002_Report_0001",
                0,
                model_report=1,
                model_patient=1,
                icdsc=0,
                icd10=0,
            ),
            _report_row(
                "Patient_0003",
                "Patient_0003_Report_0001",
                0,
                model_report=0,
                model_patient=0,
                icdsc=0,
                icd10=0,
            ),
            _report_row(
                "Patient_0004",
                "Patient_0004_Report_0001",
                1,
                model_report=0,
                model_patient=0,
                icdsc=1,
                icd10=1,
            ),
            _report_row("Patient_0005", "Patient_0005_Report_0001", 1),
            _report_row("Patient_0005", "Patient_0005_Report_0002", ""),
        ]
    )
    out_dir = tmp_path / "final_evaluation"
    patient_gt, metrics, confusion, report = run_final_evaluation(df, output_dir=out_dir)

    assert len(patient_gt) == 5
    assert (out_dir / "patient_level_ground_truth.csv").exists()
    assert (out_dir / "final_metrics_summary.csv").exists()
    assert (out_dir / "confusion_counts.csv").exists()
    assert (out_dir / "report.txt").exists()
    assert (out_dir / "plots" / "confusion_matrix_model_vs_manual.png").exists()
    assert (out_dir / "plots" / "confusion_matrix_icdsc_vs_manual.png").exists()
    assert (out_dir / "plots" / "confusion_matrix_icd10_vs_manual.png").exists()
    assert (out_dir / "plots" / "confusion_matrix_composite_or_vs_manual.png").exists()
    assert (out_dir / "plots" / "confusion_matrix_composite_and_vs_manual.png").exists()

    assert "incomplete_patients=1" in report
    assert "WARNING" in report
    assert len(metrics) == 5

    model_row = metrics[metrics["method"] == "model"].iloc[0]
    assert model_row["n_patients"] == 4
    assert model_row["tp"] == 1
    assert model_row["fp"] == 1
    assert model_row["tn"] == 1
    assert model_row["fn"] == 1

    for label in ("TP", "FP", "TN", "FN"):
        path = out_dir / f"model_{label}.csv"
        assert path.exists(), f"missing {path.name}"
        exported = pd.read_csv(path)
        assert list(exported.columns) == list(ERROR_EXPORT_COLUMNS)

    tp_df = pd.read_csv(out_dir / "model_TP.csv")
    assert len(tp_df) == 1
    assert tp_df.iloc[0]["validation_patient_id"] == "Patient_0001"
    assert "clear delirium" in str(tp_df.iloc[0]["manual_comments_summary"])


def test_primary_metrics_ignore_incomplete_in_counts(tmp_path):
    df = pd.DataFrame(
        [
            _report_row("Patient_0001", "Patient_0001_Report_0001", ""),
            _report_row("Patient_0002", "Patient_0002_Report_0001", 1, model_report=1, model_patient=1),
        ]
    )
    _, metrics, _, _ = run_final_evaluation(df, output_dir=tmp_path / "out")
    model_row = metrics[metrics["method"] == "model"].iloc[0]
    assert model_row["n_patients"] == 1
    assert model_row["tp"] == 1


def test_final_evaluation_refreshes_baseline_from_structured_baseline(tmp_path):
    cohort_path = tmp_path / "patient_validation_cohort_frozen.csv"
    labels_path = tmp_path / "manual_report_labels_frozen.csv"
    baseline_path = tmp_path / "structured_baseline.csv"
    predictions_path = tmp_path / "validation_cohort_predictions.csv"

    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "PatientenID": ["p1", "p2"],
            "model_report_prediction": [1, 0],
            "model_patient_positive": [1, 0],
            "baseline_icdsc_ge_4": [0, 0],
            "baseline_icd10": [0, 0],
        }
    ).to_csv(cohort_path, index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "manual_report_ground_truth": [1, 0],
            "manual_comment": ["pos", ""],
        }
    ).to_csv(labels_path, index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "PatientenID": ["p1", "p2"],
            "klasse": [1, 0],
            "status": ["processed", "skipped"],
            "llm_called": [1, 0],
            "skipped_reason": ["direct", "no_evidence"],
            "evidence_snippets": ["[]", "[]"],
            "signalstaerke": ["hoch", "niedrig"],
            "delir_probability_estimate": [70, 0],
            "decision_rule_applied": ["direct", "no_evidence"],
        }
    ).to_csv(predictions_path, index=False)
    labels_before = labels_path.read_bytes()

    pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "baseline_icd10": [1, 0],
            "baseline_icdsc_ge_4": [0, 1],
            "max_icdsc": [2, 5],
        }
    ).to_csv(baseline_path, index=False)

    merged, _, _ = load_merged_frozen_cohort(
        cohort_path, labels_path, baseline_path, predictions_path
    )
    gt = build_patient_level_ground_truth(merged)
    p1 = gt.loc[gt["validation_patient_id"] == "Patient_0001"].iloc[0]
    p2 = gt.loc[gt["validation_patient_id"] == "Patient_0002"].iloc[0]
    assert int(p1["baseline_icd10"]) == 1
    assert int(p2["baseline_icdsc_ge_4"]) == 1
    assert int(p1["derived_manual_patient_ground_truth"]) == 1
    assert labels_path.read_bytes() == labels_before

    out_dir = tmp_path / "final_evaluation"
    _, metrics, _, report = run_final_evaluation(
        merged, output_dir=out_dir, baseline_source=baseline_path
    )
    assert f"baseline_source={baseline_path}" in report
    icd10_row = metrics[metrics["method"] == "icd10"].iloc[0]
    assert icd10_row["tp"] == 1


def test_attach_structured_baseline_does_not_change_manual_labels(tmp_path):
    baseline_path = tmp_path / "structured_baseline.csv"
    pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "baseline_icd10": [1],
            "baseline_icdsc_ge_4": [0],
        }
    ).to_csv(baseline_path, index=False)

    merged = pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "validation_report_id": ["Patient_0001_Report_0001"],
            "manual_report_ground_truth": ["1"],
            "baseline_icd10": [0],
        }
    )
    out = attach_structured_baseline(merged, baseline_path)
    assert str(out.iloc[0]["manual_report_ground_truth"]) == "1"
    assert int(out.iloc[0]["baseline_icd10"]) == 1
