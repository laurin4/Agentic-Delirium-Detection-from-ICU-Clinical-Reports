"""Frozen manual validation cohort workflow."""

import json
import os

import pandas as pd
import pytest

from src.analysis.freeze_validation_cohort import freeze_validation_cohort
from src.analysis.frozen_validation_cohort import (
    assert_can_write_frozen,
    frozen_cohort_exists,
    resolve_validation_input_paths,
    sha256_file,
)
from src.analysis.evaluate_manual_validation import main as eval_main


def _cohort_and_labels(tmp_path):
    cohort = tmp_path / "patient_validation_cohort.csv"
    labels = tmp_path / "manual_report_labels.csv"
    preds = tmp_path / "predictions.csv"
    base = tmp_path / "baseline.csv"
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
    ).to_csv(cohort, index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "manual_report_ground_truth": [1, 0],
            "manual_comment": ["", ""],
        }
    ).to_csv(labels, index=False)
    preds.write_text("pred\n", encoding="utf-8")
    base.write_text("base\n", encoding="utf-8")
    return cohort, labels, preds, base


def test_freeze_creates_files_and_metadata(tmp_path, monkeypatch):
    cohort, labels, preds, base = _cohort_and_labels(tmp_path)
    frozen_dir = tmp_path / "frozen_validation_cohort"
    monkeypatch.setattr("src.pipeline.paths.FROZEN_VALIDATION_COHORT_DIR", frozen_dir)
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_PATIENT_VALIDATION_COHORT_PATH",
        frozen_dir / "patient_validation_cohort_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_MANUAL_REPORT_LABELS_PATH",
        frozen_dir / "manual_report_labels_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_COHORT_METADATA_PATH",
        frozen_dir / "frozen_cohort_metadata.json",
    )

    meta_path = freeze_validation_cohort(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        baseline_path=base,
        output_dir=frozen_dir,
    )
    assert (frozen_dir / "patient_validation_cohort_frozen.csv").exists()
    assert (frozen_dir / "manual_report_labels_frozen.csv").exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_unique_patients"] == 2
    assert meta["n_report_rows"] == 2
    assert meta["checksum_sha256"]["patient_validation_cohort"] == sha256_file(cohort)
    assert "fixed dataset" in meta["note"].lower()


def test_freeze_blocks_overwrite_without_env(tmp_path, monkeypatch):
    cohort, labels, preds, base = _cohort_and_labels(tmp_path)
    frozen_dir = tmp_path / "frozen_validation_cohort"
    monkeypatch.setattr("src.pipeline.paths.FROZEN_VALIDATION_COHORT_DIR", frozen_dir)
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_PATIENT_VALIDATION_COHORT_PATH",
        frozen_dir / "patient_validation_cohort_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_MANUAL_REPORT_LABELS_PATH",
        frozen_dir / "manual_report_labels_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_COHORT_METADATA_PATH",
        frozen_dir / "frozen_cohort_metadata.json",
    )

    freeze_validation_cohort(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        baseline_path=base,
        output_dir=frozen_dir,
    )
    assert frozen_cohort_exists()

    monkeypatch.delenv("OVERWRITE_FROZEN_VALIDATION", raising=False)
    with pytest.raises(FileExistsError, match="OVERWRITE_FROZEN_VALIDATION"):
        freeze_validation_cohort(
            cohort_path=cohort,
            labels_path=labels,
            predictions_path=preds,
            baseline_path=base,
            output_dir=frozen_dir,
        )


def test_freeze_allows_overwrite_with_env(tmp_path, monkeypatch):
    cohort, labels, preds, base = _cohort_and_labels(tmp_path)
    frozen_dir = tmp_path / "frozen_validation_cohort"
    monkeypatch.setattr("src.pipeline.paths.FROZEN_VALIDATION_COHORT_DIR", frozen_dir)
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_PATIENT_VALIDATION_COHORT_PATH",
        frozen_dir / "patient_validation_cohort_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_MANUAL_REPORT_LABELS_PATH",
        frozen_dir / "manual_report_labels_frozen.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_COHORT_METADATA_PATH",
        frozen_dir / "frozen_cohort_metadata.json",
    )

    freeze_validation_cohort(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        baseline_path=base,
        output_dir=frozen_dir,
    )
    monkeypatch.setenv("OVERWRITE_FROZEN_VALIDATION", "true")
    freeze_validation_cohort(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        baseline_path=base,
        output_dir=frozen_dir,
    )


def test_evaluation_prefers_frozen_cohort(tmp_path, monkeypatch):
    cohort, labels, preds, base = _cohort_and_labels(tmp_path)
    frozen_dir = tmp_path / "frozen_validation_cohort"
    eval_dir = tmp_path / "evaluation"

    monkeypatch.setattr("src.pipeline.paths.FROZEN_VALIDATION_COHORT_DIR", frozen_dir)
    fc = frozen_dir / "patient_validation_cohort_frozen.csv"
    fl = frozen_dir / "manual_report_labels_frozen.csv"
    monkeypatch.setattr("src.pipeline.paths.FROZEN_PATIENT_VALIDATION_COHORT_PATH", fc)
    monkeypatch.setattr("src.pipeline.paths.FROZEN_MANUAL_REPORT_LABELS_PATH", fl)
    monkeypatch.setattr(
        "src.pipeline.paths.FROZEN_COHORT_METADATA_PATH",
        frozen_dir / "frozen_cohort_metadata.json",
    )
    monkeypatch.setattr("src.pipeline.paths.PATIENT_VALIDATION_COHORT_PATH", tmp_path / "mutable_cohort.csv")
    monkeypatch.setattr("src.pipeline.paths.MANUAL_REPORT_LABELS_PATH", tmp_path / "mutable_labels.csv")

    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "validation_report_id": ["Patient_0001_Report_0001"],
            "PatientenID": ["p1"],
            "model_report_prediction": [0],
            "model_patient_positive": [0],
            "baseline_icdsc_ge_4": [0],
            "baseline_icd10": [0],
        }
    ).to_csv(tmp_path / "mutable_cohort.csv", index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001"],
            "manual_report_ground_truth": [0],
        }
    ).to_csv(tmp_path / "mutable_labels.csv", index=False)

    freeze_validation_cohort(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        baseline_path=base,
        output_dir=frozen_dir,
    )

    resolved_cohort, resolved_labels, using_frozen = resolve_validation_input_paths()
    assert using_frozen
    assert resolved_cohort == fc
    assert resolved_labels == fl

    eval_main(output_dir=eval_dir)
    assert (eval_dir / "tables" / "metrics_summary.csv").exists()
