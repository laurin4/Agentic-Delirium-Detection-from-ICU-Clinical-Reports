"""Exploration module works without Diagnosenliste.csv."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import run_exploration
from src.preprocessing.diagnosis_mapper import load_diagnosis_dataframe


def test_load_diagnosis_dataframe_none_does_not_crash():
    df = load_diagnosis_dataframe(None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_run_exploration_without_diagnosenliste(monkeypatch, tmp_path):
    berichte = tmp_path / "Berichte.csv"
    icd = tmp_path / "ICD.csv"
    icdsc = tmp_path / "ICDSC.csv"
    baseline = tmp_path / "structured_baseline.csv"
    tables = tmp_path / "tables"
    plots = tmp_path / "plots"
    report = tmp_path / "exploration_report.txt"

    pd.DataFrame(
        {
            "PatientID": ["p1"],
            "diag": ["Hyperaktives Delir"],
            "epikrise": ["Stabil"],
            "jetziges_leiden": [""],
            "prozedere": [""],
        }
    ).to_csv(berichte, index=False, sep=";")
    pd.DataFrame({"PatientID": ["p1"], "icd_hd": [1], "icd_code": ["F05.0"]}).to_csv(icd, index=False, sep=";")
    pd.DataFrame({"PatientID": ["p1"], "ICDSC_Max": [4]}).to_csv(icdsc, index=False, sep=";")
    pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "has_delir_icd10": [1],
            "max_icdsc": [4],
            "baseline_icd10": [1],
            "baseline_icdsc_ge_1": [1],
            "baseline_icdsc_ge_2": [1],
            "baseline_icdsc_ge_3": [1],
            "baseline_icdsc_ge_4": [1],
            "baseline_icdsc_ge_5": [0],
            "baseline_icdsc_0": [0],
            "baseline_icdsc_1_to_3": [0],
            "baseline_icdsc_ge_4_grouped": [1],
        }
    ).to_csv(baseline, index=False)

    monkeypatch.setattr(run_exploration, "BERICHTE_INPUT_PATH", berichte)
    monkeypatch.setattr(run_exploration, "ICD10_PATH", icd)
    monkeypatch.setattr(run_exploration, "ICDSC_PATH", icdsc)
    monkeypatch.setattr(run_exploration, "STRUCTURED_BASELINE_PATH", baseline)
    monkeypatch.setattr(run_exploration, "DIAGNOSIS_INPUT_PATH", None)
    monkeypatch.setattr(run_exploration, "LEGACY_DIAGNOSIS_INPUT_PATH", tmp_path / "Diagnosenliste.csv")
    monkeypatch.setattr(run_exploration, "EXPLORATION_TABLES_DIR", tables)
    monkeypatch.setattr(run_exploration, "EXPLORATION_PLOTS_DIR", plots)
    monkeypatch.setattr(run_exploration, "EXPLORATION_REPORT_PATH", report)
    monkeypatch.setattr(run_exploration, "PATIENT_LEVEL_REPORTS_PATH", tmp_path / "missing_prepared.csv")
    monkeypatch.setattr(run_exploration, "REPORT_VS_BASELINE_PATH", tmp_path / "missing_cmp.csv")
    monkeypatch.setattr(run_exploration, "PREDICTIONS_PROMPT_PATH", tmp_path / "missing_pred.csv")

    run_exploration.main()

    assert report.exists()
    assert (tables / "dataset_overview.csv").exists()
    assert (tables / "berichte_section_overview.csv").exists()
    text = report.read_text(encoding="utf-8")
    assert "Berichte.csv" in text
    assert "deprecated" in text.lower() or "Diagnosenliste" in text
