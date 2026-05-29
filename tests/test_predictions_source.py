"""Tests for PREDICTIONS_SOURCE resolution."""

import pandas as pd
import pytest

from src.pipeline import predictions_source as ps


@pytest.fixture
def pred_dir(tmp_path, monkeypatch):
    """Isolated predictions directory with full and cohort CSV stubs."""
    full = tmp_path / "predictions" / "agent1_agent2_agent3_results_prompt.csv"
    cohort = tmp_path / "predictions" / "validation_cohort_predictions.csv"
    full.parent.mkdir(parents=True)
    full.write_text("PatientenID,klasse\np1,0\n", encoding="utf-8")
    cohort.write_text("PatientenID,klasse\np1,1\n", encoding="utf-8")
    monkeypatch.setattr(ps, "FULL_PREDICTIONS_PATH", full)
    monkeypatch.setattr(ps, "VALIDATION_COHORT_PREDICTIONS_PATH", cohort)
    return full, cohort


def test_default_full_source(pred_dir, monkeypatch):
    full, cohort = pred_dir
    monkeypatch.delenv(ps.PREDICTIONS_SOURCE_ENV, raising=False)
    assert ps.get_predictions_source() == ps.PREDICTIONS_SOURCE_FULL
    assert ps.resolve_predictions_path() == full
    assert ps.resolve_predictions_path() != cohort


def test_validation_cohort_source(pred_dir, monkeypatch):
    full, cohort = pred_dir
    monkeypatch.setenv(ps.PREDICTIONS_SOURCE_ENV, "validation_cohort")
    assert ps.get_predictions_source() == ps.PREDICTIONS_SOURCE_VALIDATION_COHORT
    assert ps.resolve_predictions_path() == cohort
    assert ps.resolve_predictions_path() != full


def test_explicit_path_overrides_env(pred_dir, monkeypatch):
    full, cohort = pred_dir
    custom = full.parent / "custom_predictions.csv"
    custom.write_text("x\n", encoding="utf-8")
    monkeypatch.setenv(ps.PREDICTIONS_SOURCE_ENV, "validation_cohort")
    assert ps.resolve_predictions_path(custom) == custom


def test_invalid_source_raises(monkeypatch):
    monkeypatch.setenv(ps.PREDICTIONS_SOURCE_ENV, "unknown")
    with pytest.raises(ValueError, match="PREDICTIONS_SOURCE"):
        ps.get_predictions_source()


def test_compare_uses_validation_cohort_file(pred_dir, monkeypatch, tmp_path):
    """run_compare reads validation_cohort_predictions.csv when env is set."""
    full, cohort = pred_dir
    from src.pipeline.compare_reports_vs_baseline import run_compare

    baseline = tmp_path / "baseline.csv"
    baseline.write_text(
        "PatientenID,has_delir_icd10,max_icdsc,baseline_icd10,"
        "baseline_icdsc_ge_1,baseline_icdsc_ge_2,baseline_icdsc_ge_3,"
        "baseline_icdsc_ge_4,baseline_icdsc_ge_5,baseline_icdsc_0,"
        "baseline_icdsc_1_to_3,baseline_icdsc_ge_4_grouped,baseline_composite\n"
        "p1,0,2,0,1,1,0,0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    out = tmp_path / "cmp.csv"
    excl = tmp_path / "excl.csv"

    monkeypatch.setenv(ps.PREDICTIONS_SOURCE_ENV, "validation_cohort")
    run_compare(baseline_path=baseline, output_path=out, excluded_path=excl)

    cmp_df = pd.read_csv(out)
    assert len(cmp_df) == 1
    assert int(cmp_df.iloc[0]["klasse"]) == 1

    monkeypatch.setenv(ps.PREDICTIONS_SOURCE_ENV, "full")
    run_compare(baseline_path=baseline, output_path=out, excluded_path=excl)
    cmp_df = pd.read_csv(out)
    assert int(cmp_df.iloc[0]["klasse"]) == 0
