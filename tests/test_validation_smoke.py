from src.validation.validate_inputs import run_checks, write_outputs


def test_run_checks_returns_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("src.validation.validate_inputs.VALIDATION_DIR", tmp_path)
    monkeypatch.setattr("src.validation.validate_inputs.VALIDATION_RESULTS_CSV_PATH", tmp_path / "validation_results.csv")
    monkeypatch.setattr("src.validation.validate_inputs.VALIDATION_SUMMARY_TXT_PATH", tmp_path / "validation_summary.txt")
    rows = run_checks()
    assert any(r["check"] == "diagnosis_patient_reports_row_count" for r in rows)
    write_outputs(rows)
    assert (tmp_path / "validation_results.csv").exists()
