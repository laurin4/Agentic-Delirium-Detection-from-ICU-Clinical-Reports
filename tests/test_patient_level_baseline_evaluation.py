"""Tests for patient-level baseline comparison and evaluation."""

import pandas as pd

from src.pipeline.compare_patients_vs_baseline import run_compare
from src.pipeline.evaluate_predictions_patient_level import main as eval_patient_main
from src.pipeline.patient_prediction_aggregate import aggregate_predictions_to_patient_level


def _baseline_row(pid: str, **kwargs) -> dict:
    base = {
        "PatientenID": pid,
        "has_delir_icd10": 0,
        "max_icdsc": 0.0,
        "baseline_icd10": 0,
        "baseline_icdsc_ge_4": 0,
        "baseline_composite_or": 0,
        "baseline_composite_and": 0,
        "baseline_composite": 0,
    }
    base.update(kwargs)
    return base


def test_aggregate_patient_positive_is_max_report_klasse():
    reports = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1", "p2", "p2"],
            "klasse": [0, 1, 0, 0],
            "bertyp": ["Verlaufseintrag"] * 4,
        }
    )
    out = aggregate_predictions_to_patient_level(reports)
    assert len(out) == 2
    p1 = out[out["PatientenID"] == "p1"].iloc[0]
    p2 = out[out["PatientenID"] == "p2"].iloc[0]
    assert int(p1["model_patient_positive"]) == 1
    assert int(p1["n_reports_positive"]) == 1
    assert int(p2["model_patient_positive"]) == 0


def test_aggregate_excludes_dokumentationsblatt():
    reports = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1"],
            "klasse": [1, 0],
            "bertyp": ["Dokumentationsblatt", "Verlaufseintrag"],
        }
    )
    out = aggregate_predictions_to_patient_level(reports)
    assert len(out) == 1
    assert int(out.iloc[0]["model_patient_positive"]) == 0
    assert int(out.iloc[0]["n_reports_in_aggregate"]) == 1


def test_compare_and_evaluate_patient_level(tmp_path, monkeypatch):
    baseline = pd.DataFrame(
        [
            _baseline_row("p1", max_icdsc=5.0, baseline_icdsc_ge_4=1, baseline_composite_or=1),
            _baseline_row("p2", has_delir_icd10=1, baseline_icd10=1, baseline_composite_or=1),
            _baseline_row("p3"),
        ]
    )
    preds = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1", "p2", "p3"],
            "klasse": [1, 0, 0, 1],
            "bertyp": ["Verlaufseintrag"] * 4,
        }
    )
    bpath = tmp_path / "baseline.csv"
    ppath = tmp_path / "pred.csv"
    cmp_out = tmp_path / "patient_cmp.csv"
    excl = tmp_path / "patient_excl.csv"
    agg_out = tmp_path / "patient_agg.csv"
    baseline.to_csv(bpath, index=False)
    preds.to_csv(ppath, index=False)

    run_compare(
        baseline_path=bpath,
        predictions_path=ppath,
        output_path=cmp_out,
        excluded_path=excl,
        aggregate_output_path=agg_out,
    )

    cmp_df = pd.read_csv(cmp_out)
    assert len(cmp_df) == 3
    p1 = cmp_df[cmp_df["PatientenID"] == "p1"].iloc[0]
    assert int(p1["model_patient_positive"]) == 1
    assert bool(p1["agreement_patient_vs_baseline_icdsc_ge_4"])

    eval_dir = tmp_path / "evaluation" / "patient_level"
    plots = eval_dir / "plots"
    tables = eval_dir / "tables"
    plots.mkdir(parents=True)
    tables.mkdir(parents=True)

    import src.pipeline.evaluate_predictions_patient_level as evp

    monkeypatch.setattr(evp, "PATIENT_VS_BASELINE_PATH", cmp_out)
    monkeypatch.setattr(evp, "EVALUATION_PATIENT_LEVEL_DIR", eval_dir)
    monkeypatch.setattr(evp, "EVALUATION_PATIENT_LEVEL_TABLES_DIR", tables)
    monkeypatch.setattr(evp, "EVALUATION_PATIENT_LEVEL_PLOTS_DIR", plots)
    monkeypatch.setattr(evp, "EVALUATION_PATIENT_LEVEL_SUMMARY_PATH", tables / "summary.csv")
    monkeypatch.setattr(
        evp, "EVALUATION_PATIENT_LEVEL_CONFUSION_COUNTS_PATH", tables / "confusion.csv"
    )
    monkeypatch.setattr(evp, "EVALUATION_PATIENT_LEVEL_REPORT_PATH", eval_dir / "report.txt")
    monkeypatch.setattr(evp, "EVALUATION_SUMMARY_PATH", tmp_path / "eval_summary.csv")

    eval_patient_main()

    summary = pd.read_csv(tables / "summary.csv")
    assert len(summary) == 4
    assert (plots / "confusion_matrix_baseline_icdsc_ge_4.png").exists()
    assert "n_patients: 3" in (eval_dir / "report.txt").read_text(encoding="utf-8")
