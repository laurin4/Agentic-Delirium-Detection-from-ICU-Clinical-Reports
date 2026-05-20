"""Tests for configurable baseline_composite OR vs AND mode."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.baseline_composite import (
    baseline_composite_definition_text,
    compute_baseline_composite,
    format_baseline_composite_mode_banner,
    resolve_baseline_composite_mode,
)
from src.pipeline.prepare_structured_data import add_binary_baselines


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "has_delir_icd10": [0, 1, 0, 1],
            "max_icdsc": [0, 2, 5, 5],
        }
    )


def test_or_mode_composite(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "OR")
    out = add_binary_baselines(_sample_df())
    assert list(out["baseline_composite"]) == [0, 1, 1, 1]


def test_and_mode_composite(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "AND")
    out = add_binary_baselines(_sample_df())
    # rows: (0,0), (icd10 only), (icdsc>=4 only), (both)
    assert list(out["baseline_composite"]) == [0, 0, 0, 1]


def test_switching_mode_changes_composite():
    ge4 = pd.Series([0, 0, 1, 1])
    icd10 = pd.Series([0, 1, 0, 1])
    assert list(compute_baseline_composite(ge4, icd10, mode="OR")) == [0, 1, 1, 1]
    assert list(compute_baseline_composite(ge4, icd10, mode="AND")) == [0, 0, 0, 1]


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        resolve_baseline_composite_mode("XOR")


def test_mode_banner_or_and(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "OR")
    assert "OR" in format_baseline_composite_mode_banner()
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "AND")
    assert "AND" in format_baseline_composite_mode_banner()
    assert "high-confidence" in format_baseline_composite_mode_banner().lower()


def test_definition_text_reflects_mode(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "AND")
    assert "AND" in baseline_composite_definition_text()
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "OR")
    assert "OR" in baseline_composite_definition_text()


def test_evaluate_predictions_runs(tmp_path, monkeypatch):
    """Smoke: evaluation + plots with AND mode."""
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "AND")
    comp = tmp_path / "report_vs_baseline_comparison.csv"
    eval_dir = tmp_path / "evaluation"
    plots = eval_dir / "binary_baselines" / "plots"
    tables = eval_dir / "binary_baselines" / "tables"
    plots.mkdir(parents=True)
    tables.mkdir(parents=True)

    pd.DataFrame(
        {
            "PatientenID": ["p1", "p2", "p3"],
            "klasse": [1, 0, 1],
            "has_delir_icd10": [1, 0, 1],
            "max_icdsc": [5, 2, 5],
        }
    ).to_csv(comp, index=False)

    import src.pipeline.evaluate_predictions as ev

    monkeypatch.setattr(ev, "REPORT_VS_BASELINE_PATH", comp)
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINES_DIR", eval_dir / "binary_baselines")
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINES_TABLES_DIR", tables)
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINES_PLOTS_DIR", plots)
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINE_SUMMARY_PATH", tables / "summary.csv")
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH", tables / "confusion.csv")
    monkeypatch.setattr(ev, "EVALUATION_BINARY_BASELINE_REPORT_PATH", eval_dir / "binary_baselines" / "report.txt")
    monkeypatch.setattr(ev, "EVALUATION_SUMMARY_PATH", eval_dir / "evaluation_summary.csv")

    ev.main()
    assert (plots / "confusion_matrix_baseline_composite.png").exists()
    assert "Delirkandidaten" in (eval_dir / "binary_baselines" / "report.txt").read_text(encoding="utf-8")


def test_prepare_structured_data_prints_mode(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "AND")
    icd = tmp_path / "ICD.csv"
    icdsc = tmp_path / "ICDSC.csv"
    out = tmp_path / "baseline.csv"
    pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": ["F05.0"]}).to_csv(
        icd, index=False, sep=";"
    )
    pd.DataFrame({"PatientID": ["p1"], "ICDSC_Max": [5]}).to_csv(icdsc, index=False, sep=";")

    import src.pipeline.prepare_structured_data as psd

    monkeypatch.setattr(psd, "ICD10_PATH", icd)
    monkeypatch.setattr(psd, "ICDSC_PATH", icdsc)
    monkeypatch.setattr(psd, "STRUCTURED_BASELINE_PATH", out)
    psd.main()
    captured = capsys.readouterr().out
    assert "[Baseline Composite Mode]" in captured
    assert "AND" in captured
