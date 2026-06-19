"""Tests for patient-level biostatistical evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.biostatistical_evaluation import (
    MANUAL_GT_COL,
    MCNEMAR_EXACT_MAX_DISCORDANT,
    PatientMethodTable,
    align_method_pair,
    confusion_counts,
    diagnostic_metrics_with_ci,
    load_all_method_tables,
    mcnemar_comparison_row,
    mcnemar_test,
    run_biostatistical_evaluation,
    wilson_ci,
)
from src.pipeline.validation_report_identity import VALIDATION_PATIENT_ID_COL


def test_wilson_ci_known_proportion():
    prop, low, high = wilson_ci(8, 10)
    assert prop == pytest.approx(0.8)
    assert 0.0 < low < prop < high < 1.0


def test_wilson_ci_zero_denominator():
    prop, low, high = wilson_ci(0, 0)
    assert pd.isna(prop)
    assert pd.isna(low)
    assert pd.isna(high)


def test_confusion_counts():
    y_true = pd.Series([1, 0, 1, 0, 1])
    y_pred = pd.Series([1, 0, 0, 1, 1])
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"n": 5, "TP": 2, "FP": 1, "TN": 1, "FN": 1}


def test_diagnostic_metrics_with_ci():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 1, 1])
    m = diagnostic_metrics_with_ci(y_true, y_pred, method_name="test")
    assert m["method"] == "test"
    assert m["TP"] == 2
    assert m["FP"] == 1
    assert m["TN"] == 1
    assert m["FN"] == 0
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["sensitivity_ci_low"] <= m["sensitivity"] <= m["sensitivity_ci_high"]


def test_mcnemar_exact_test():
    test_type, statistic, p_value = mcnemar_test(3, 7)
    assert test_type == "exact_binomial"
    assert statistic == 3.0
    assert 0.0 < p_value < 1.0


def test_mcnemar_chi_square_when_many_discordant():
    b = 14
    c = 16
    assert b + c > MCNEMAR_EXACT_MAX_DISCORDANT
    test_type, statistic, p_value = mcnemar_test(b, c)
    assert test_type == "chi_square_cc"
    assert statistic == pytest.approx(((abs(b - c) - 1) ** 2) / (b + c))
    assert 0.0 < p_value < 1.0


def test_align_method_pair_by_validation_patient_id():
    a = PatientMethodTable(
        method_name="a",
        frame=pd.DataFrame(
            {
                VALIDATION_PATIENT_ID_COL: ["Patient_0001", "Patient_0002"],
                "y_true": [1, 0],
                "y_pred": [1, 0],
            }
        ),
    )
    b = PatientMethodTable(
        method_name="b",
        frame=pd.DataFrame(
            {
                VALIDATION_PATIENT_ID_COL: ["Patient_0002", "Patient_0003"],
                "y_true": [0, 1],
                "y_pred": [1, 1],
            }
        ),
    )
    aligned = align_method_pair(a, b)
    assert len(aligned) == 1
    assert aligned.iloc[0][VALIDATION_PATIENT_ID_COL] == "Patient_0002"


def test_mcnemar_comparison_row_counts():
    a = PatientMethodTable(
        method_name="a",
        frame=pd.DataFrame(
            {
                VALIDATION_PATIENT_ID_COL: ["P1", "P2", "P3", "P4"],
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 0, 0, 1],
            }
        ),
    )
    b = PatientMethodTable(
        method_name="b",
        frame=pd.DataFrame(
            {
                VALIDATION_PATIENT_ID_COL: ["P1", "P2", "P3", "P4"],
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 0, 1, 0],
            }
        ),
    )
    row = mcnemar_comparison_row(a, b)
    assert row["n_common"] == 4
    assert row["a_correct_b_wrong"] == 0
    assert row["a_wrong_b_correct"] == 2
    assert row["discordant_total"] == 2


def _write_patient_file(path: Path, preds: dict[str, int]) -> None:
    rows = []
    for i, (pid, pred) in enumerate(preds.items(), start=1):
        manual = 1 if i <= 2 else 0
        rows.append(
            {
                VALIDATION_PATIENT_ID_COL: pid,
                "is_patient_complete": 1,
                MANUAL_GT_COL: manual,
                "model_patient_positive": pred,
                "baseline_icdsc_ge_4": pred,
                "baseline_icd10": 0,
                "baseline_composite_or": pred,
                "baseline_composite_and": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_missing_method_files_do_not_crash(tmp_path):
    baseline = tmp_path / "baseline.csv"
    v1_path = tmp_path / "v1.csv"
    _write_patient_file(
        baseline,
        {"Patient_0001": 1, "Patient_0002": 0, "Patient_0003": 0},
    )
    _write_patient_file(v1_path, {"Patient_0001": 1, "Patient_0002": 0, "Patient_0003": 1})

    specs = (
        ("v1", v1_path, "model_patient_positive"),
        ("v2_run_02", tmp_path / "missing.csv", "model_patient_positive"),
    )
    tables = load_all_method_tables(method_specs=specs, baseline_source=baseline)
    assert "v1" in tables
    assert "icdsc" in tables
    assert "v2_run_02" not in tables

    out = tmp_path / "bio"
    report = run_biostatistical_evaluation(
        out,
        method_specs=specs,
        baseline_source=baseline,
        comparisons=(("v1", "icdsc"), ("v1", "v2_run_02")),
    )
    assert "Diagnostic metrics summary" in report
    assert (out / "diagnostic_metrics_with_ci.csv").exists()
    assert (out / "mcnemar_tests.csv").exists()
    mcnemar = pd.read_csv(out / "mcnemar_tests.csv")
    assert len(mcnemar) == 1
    assert mcnemar.iloc[0]["method_a"] == "v1"
    assert mcnemar.iloc[0]["method_b"] == "icdsc"
