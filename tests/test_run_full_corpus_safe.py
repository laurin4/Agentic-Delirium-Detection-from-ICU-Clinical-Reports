"""Tests for safe full-corpus launcher."""

import os

import pytest

from src.pipeline import run_full_corpus_safe as safe
from src.pipeline.paths import FULL_PREDICTIONS_PATH, PREDICTIONS_DIR


def test_refuses_validation_cohort_only(monkeypatch):
    monkeypatch.setenv("VALIDATION_COHORT_ONLY", "true")
    with pytest.raises(SystemExit, match="VALIDATION_COHORT_ONLY"):
        safe.run_safe(smoke=True)


def test_refuses_max_reports_without_flag(monkeypatch):
    monkeypatch.delenv("VALIDATION_COHORT_ONLY", raising=False)
    monkeypatch.setenv("MAX_REPORTS", "60")
    with pytest.raises(SystemExit, match="MAX_REPORTS"):
        safe.run_safe(smoke=False)


def test_smoke_configures_env_and_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("VALIDATION_COHORT_ONLY", raising=False)
    monkeypatch.delenv("MAX_REPORTS", raising=False)
    monkeypatch.setattr(safe, "BERICHTE_INPUT_PATH", tmp_path / "Berichte.csv")
    safe.BERICHTE_INPUT_PATH.write_text("PatientID;bertyp;bername;diag\n", encoding="utf-8")

    smoke_out = tmp_path / "predictions" / "full_corpus_smoke_5.csv"
    monkeypatch.setattr(safe, "SMOKE_OUTPUT_PATH", smoke_out)
    monkeypatch.setattr(safe, "PREDICTIONS_DIR", tmp_path / "predictions")

    records = [
        {
            "PatientenID": f"p{i}",
            "bericht": f"doc_{i}",
            "bertyp": "Verlaufseintrag",
            "berdat": "2024-01-01",
            "report_text": "Delir",
        }
        for i in range(10)
    ]

    monkeypatch.setattr(
        safe,
        "build_report_level_berichte_records",
        lambda: (records, 0),
    )

    captured: dict = {}

    def fake_main():
        captured["output"] = os.environ.get(safe.RUN_PIPELINE_OUTPUT_PATH_ENV)
        captured["override"] = os.environ.get(safe.RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV)
        captured["flush"] = os.environ.get(safe.RUN_PIPELINE_PROGRESS_FLUSH_ENV)
        captured["skip_copy"] = os.environ.get(safe.RUN_PIPELINE_SKIP_MODEL_COPY_ENV)
        captured["checkpoint"] = os.environ.get("PIPELINE_CHECKPOINT_EVERY")

    monkeypatch.setattr(safe, "run_pipeline_main", fake_main)

    rc = safe.run_safe(smoke=True, checkpoint_every=2)
    assert rc == 0
    assert captured["output"] == str(smoke_out.resolve())
    assert captured["override"] == "5"
    assert captured["flush"] == "true"
    assert captured["skip_copy"] == "true"
    assert captured["checkpoint"] == "2"

    assert smoke_out.with_name(smoke_out.name + ".completed").exists()
    assert not smoke_out.with_name(smoke_out.name + ".running").exists()


def test_full_run_backs_up_main_csv(tmp_path, monkeypatch):
    monkeypatch.delenv("VALIDATION_COHORT_ONLY", raising=False)
    monkeypatch.delenv("MAX_REPORTS", raising=False)
    monkeypatch.setattr(safe, "BERICHTE_INPUT_PATH", tmp_path / "Berichte.csv")
    safe.BERICHTE_INPUT_PATH.write_text("x", encoding="utf-8")

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    main_csv = pred_dir / "agent1_agent2_agent3_results_prompt.csv"
    main_csv.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(safe, "FULL_PREDICTIONS_PATH", main_csv)
    monkeypatch.setattr(safe, "PREDICTIONS_DIR", pred_dir)
    monkeypatch.setattr(safe, "build_report_level_berichte_records", lambda: ([{"PatientenID": "p1", "bericht": "d", "report_text": "t"}], 0))
    monkeypatch.setattr(safe, "run_pipeline_main", lambda: None)

    safe.run_safe(smoke=False, checkpoint_every=50)
    backups = list(pred_dir.glob("agent1_agent2_agent3_results_prompt.backup_*.csv"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old\n"


def test_failure_keeps_failed_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("VALIDATION_COHORT_ONLY", raising=False)
    monkeypatch.delenv("MAX_REPORTS", raising=False)
    monkeypatch.setattr(safe, "BERICHTE_INPUT_PATH", tmp_path / "Berichte.csv")
    safe.BERICHTE_INPUT_PATH.write_text("x", encoding="utf-8")
    smoke_out = tmp_path / "predictions" / "full_corpus_smoke_5.csv"
    monkeypatch.setattr(safe, "SMOKE_OUTPUT_PATH", smoke_out)
    monkeypatch.setattr(safe, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(safe, "build_report_level_berichte_records", lambda: ([{"PatientenID": "p1", "bericht": "d", "report_text": "t"}], 0))

    def boom():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(safe, "run_pipeline_main", boom)

    with pytest.raises(RuntimeError, match="simulated failure"):
        safe.run_safe(smoke=True, checkpoint_every=1)

    assert smoke_out.with_name(smoke_out.name + ".failed").exists()
    assert not smoke_out.with_name(smoke_out.name + ".running").exists()
