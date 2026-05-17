"""Tests for patient-level cohort counts and data coverage analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import cohort_counts, run_data_coverage_analysis
from src.pipeline.paths import STRUCTURED_BASELINE_PATH


def test_no_stale_hardcoded_cohort_counts_in_analysis_sources():
    root = Path(__file__).resolve().parents[1] / "src" / "analysis"
    forbidden = ("25800", "193")
    for py in root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Stale hardcoded value {token!r} in {py.name}"


def test_structured_baseline_path_is_outputs_baseline():
    assert STRUCTURED_BASELINE_PATH.name == "structured_baseline.csv"
    assert STRUCTURED_BASELINE_PATH.parent.name == "baseline"
    assert "outputs" in STRUCTURED_BASELINE_PATH.parts


def test_compute_current_cohort_counts(tmp_path):
    berichte = tmp_path / "Berichte.csv"
    baseline = tmp_path / "structured_baseline.csv"
    berichte.write_text(
        "PatientID;berdat;diag\n"
        "p1;2024-01-01;text\n"
        "p2;2024-01-02;text\n"
        "p1;2024-01-03;text2\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "PatientenID": ["p1", "p3", "p3"],
            "has_delir_icd10": [0, 1, 1],
            "max_icdsc": [2, 4, 5],
        }
    ).to_csv(baseline, index=False)

    b_df = cohort_counts.load_berichte_rows(berichte)
    m_df = cohort_counts.load_structured_baseline_rows(baseline)
    counts = cohort_counts.compute_current_cohort_counts(b_df, m_df)

    assert counts["berichte_rows"] == 3
    assert counts["berichte_unique_patientids"] == 2
    assert counts["structured_baseline_rows"] == 2  # deduped p3
    assert counts["structured_baseline_unique_patientids"] == 2
    assert counts["overlap_patientids"] == 1
    assert counts["berichte_without_baseline"] == 1
    assert counts["baseline_without_berichte"] == 1


def test_data_coverage_writes_current_cohort_counts_and_plots(tmp_path, monkeypatch):
    berichte = tmp_path / "Berichte.csv"
    baseline = tmp_path / "structured_baseline.csv"
    berichte.write_text("PatientID;diag\np1;x\np2;y\n", encoding="utf-8")
    pd.DataFrame({"PatientenID": ["p1", "p2"], "has_delir_icd10": [0, 0], "max_icdsc": [0, 0]}).to_csv(
        baseline, index=False
    )

    icd = tmp_path / "ICD.csv"
    icdsc = tmp_path / "ICDSC.csv"
    icd.write_text("PatientID;icd_hd;icd_code\np1;1;I10\n", encoding="utf-8")
    icdsc.write_text("PatientID;ICDSC_Max\np1;0\n", encoding="utf-8")

    out_root = tmp_path / "analysis" / "data_coverage"
    plots = out_root / "plots"
    tables = out_root / "tables"

    monkeypatch.setattr(run_data_coverage_analysis, "BERICHTE_INPUT_PATH", berichte)
    monkeypatch.setattr(run_data_coverage_analysis, "STRUCTURED_BASELINE_PATH", baseline)
    monkeypatch.setattr(run_data_coverage_analysis, "ICD10_PATH", icd)
    monkeypatch.setattr(run_data_coverage_analysis, "ICDSC_PATH", icdsc)
    monkeypatch.setattr(run_data_coverage_analysis, "DATA_COVERAGE_ANALYSIS_DIR", out_root)
    monkeypatch.setattr(run_data_coverage_analysis, "DATA_COVERAGE_PLOTS_DIR", plots)
    monkeypatch.setattr(run_data_coverage_analysis, "DATA_COVERAGE_TABLES_DIR", tables)

    stale = plots / "stale_old_plot.png"
    plots.mkdir(parents=True)
    stale.write_bytes(b"stale")

    run_data_coverage_analysis.main()

    assert not stale.exists()
    counts_csv = tables / "current_cohort_counts.csv"
    assert counts_csv.exists()
    counts_df = pd.read_csv(counts_csv)
    assert list(counts_df["metric"]) == list(cohort_counts.CURRENT_COHORT_METRIC_NAMES)
    assert int(counts_df.loc[counts_df["metric"] == "overlap_patientids", "value"].iloc[0]) == 2

    for name in (
        "patient_level_cohort_sizes.png",
        "cohort_size_context.png",
        "overlap_distribution.png",
        "unmatched_counts.png",
        "raw_source_sizes.png",
    ):
        assert (plots / name).exists(), f"missing plot {name}"


def test_load_structured_baseline_requires_outputs_path(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="structured_baseline"):
        cohort_counts.load_structured_baseline_rows(missing)
