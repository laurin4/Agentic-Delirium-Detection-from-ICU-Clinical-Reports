"""Tests for final evaluation alignment check script."""

import pandas as pd

from src.analysis.check_final_eval_alignment import (
    compute_alignment_verdict,
    run_final_eval_alignment_check,
    write_final_eval_alignment_check,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL


def _write_aligned_pair(tmp_path, *, pred_berdat=None):
    cohort_path = tmp_path / "cohort.csv"
    preds_path = tmp_path / "preds.csv"
    berdat_c = "2024-01-01"
    berdat_p = berdat_c if pred_berdat is None else pred_berdat
    pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: ["Patient_0001_Report_0001", "Patient_0001_Report_0002"],
            "PatientenID": ["p1", "p1"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht"],
            "berdat": [berdat_c, "2024-01-02"],
        }
    ).to_csv(cohort_path, index=False)
    pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: ["Patient_0001_Report_0001", "Patient_0001_Report_0002"],
            "PatientenID": ["p1", "p1"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht"],
            "berdat": [berdat_p, "2024-01-02"],
        }
    ).to_csv(preds_path, index=False)
    return cohort_path, preds_path


def test_alignment_pass(tmp_path):
    cohort_path, preds_path = _write_aligned_pair(tmp_path)
    result = run_final_eval_alignment_check(cohort_path, preds_path)
    assert result.verdict == "PASS"
    assert result.patient_id_mismatch == 0
    assert result.bertyp_mismatch == 0
    assert result.berdat_mismatch == 0


def test_alignment_warning_berdat_format_only(tmp_path):
    cohort_path, preds_path = _write_aligned_pair(tmp_path, pred_berdat="01.01.2024")
    result = run_final_eval_alignment_check(cohort_path, preds_path)
    assert result.verdict == "WARNING"
    assert result.berdat_format_only_mismatch >= 1
    assert result.berdat_mismatch == 0


def test_alignment_fail_patient_mismatch(tmp_path):
    cohort_path, preds_path = _write_aligned_pair(tmp_path)
    preds = pd.read_csv(preds_path)
    preds.loc[0, "PatientenID"] = "p2"
    preds.to_csv(preds_path, index=False)
    result = run_final_eval_alignment_check(cohort_path, preds_path)
    assert result.verdict == "FAIL"
    assert result.patient_id_mismatch == 1


def test_alignment_fail_id_set_mismatch(tmp_path):
    cohort_path, preds_path = _write_aligned_pair(tmp_path)
    preds = pd.read_csv(preds_path)
    preds = preds.iloc[:1]
    preds.to_csv(preds_path, index=False)
    result = run_final_eval_alignment_check(cohort_path, preds_path)
    assert result.verdict == "FAIL"
    assert len(result.missing_in_predictions) == 1


def test_write_report_file(tmp_path):
    cohort_path, preds_path = _write_aligned_pair(tmp_path)
    out = tmp_path / "final_eval_alignment_check.txt"
    result = write_final_eval_alignment_check(cohort_path, preds_path, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "VERDICT: PASS" in text
    assert result.verdict == "PASS"


def test_compute_verdict_fail_on_errors():
    from src.analysis.check_final_eval_alignment import AlignmentCheckResult

    res = AlignmentCheckResult(errors=["missing file"])
    assert compute_alignment_verdict(res) == "FAIL"
