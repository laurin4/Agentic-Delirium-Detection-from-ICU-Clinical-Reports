"""Tests for patient-level manual validation cohort export."""

import json

import pandas as pd

from src.analysis.export_patient_validation_cohort import (
    COHORT_COLUMNS,
    build_patient_validation_cohort,
    select_validation_patient_ids,
)
from src.preprocessing.berichte_filters import DOKUMENTATIONSBLATT_BERTYP


def _predictions() -> pd.DataFrame:
    rows = []
    for pid, klasse in [("p1", 1), ("p1", 0), ("p2", 0)]:
        rows.append(
            {
                "PatientenID": pid,
                "bericht": f"{pid}_v.txt",
                "bertyp": "Verlaufseintrag",
                "klasse": klasse,
                "signalstaerke": "mittel" if klasse else "niedrig",
                "delir_probability_estimate": 50,
                "manual_review_candidate": "False",
                "decision_rule_applied": "test",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "k",
                "begruendung": "b",
                "original_report_text_length": 100,
                "llm_report_text_length": 50,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        )
    rows.append(
        {
            "PatientenID": "p1",
            "bericht": "p1_a.txt",
            "bertyp": "Austrittsbericht",
            "klasse": 1,
            "signalstaerke": "hoch",
            "delir_probability_estimate": 80,
            "manual_review_candidate": "True",
            "decision_rule_applied": "direct",
            "evidence_snippets": json.dumps([{"text": "Delir"}]),
            "delir_signale": "",
            "kontext": "",
            "begruendung": "",
            "original_report_text_length": 200,
            "llm_report_text_length": 100,
            "llm_text_reduction_method": "structured_evidence_extraction",
        }
    )
    rows.append(
        {
            "PatientenID": "p3",
            "bericht": "p3_doc",
            "bertyp": DOKUMENTATIONSBLATT_BERTYP,
            "klasse": 1,
            "signalstaerke": "hoch",
            "delir_probability_estimate": 90,
            "manual_review_candidate": "False",
            "decision_rule_applied": "x",
            "evidence_snippets": "[]",
            "delir_signale": "",
            "kontext": "",
            "begruendung": "",
            "original_report_text_length": 10,
            "llm_report_text_length": 0,
            "llm_text_reduction_method": "no_evidence_prefilter_skip",
        }
    )
    return pd.DataFrame(rows)


def _patient_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": ["p1", "p2", "p3"],
            "baseline_composite": [1, 0, 0],
            "ICDSC_max": [5, 0, 0],
            "ICD10": [0, 0, 0],
            "baseline_icdsc_ge_4": [1, 0, 0],
            "model_patient_positive": [1, 0, 0],
            "n_verlaufseintrag": [2, 1, 0],
            "n_verlegungsbericht": [0, 0, 0],
            "n_austrittsbericht": [1, 0, 0],
            "any_manual_review_candidate": [1, 0, 0],
        }
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "max_icdsc": [5, 0],
            "baseline_icd10": [0, 0],
            "baseline_icdsc_ge_4": [1, 0],
            "baseline_composite": [1, 0],
        }
    )


def test_select_n_unique_patients_exports_all_reports():
    preds = _predictions()
    matrix = _patient_matrix()
    selected, _ = select_validation_patient_ids(matrix, target_n=2)
    assert len(selected) == 2
    cohort = build_patient_validation_cohort(preds, _baseline(), matrix, selected)
    p1_rows = cohort[cohort["PatientenID"] == "p1"]
    assert len(p1_rows) == 3
    assert DOKUMENTATIONSBLATT_BERTYP not in set(cohort["bertyp"])


def test_patient_with_multiple_reports_all_included():
    cohort = build_patient_validation_cohort(
        _predictions(),
        _baseline(),
        _patient_matrix(),
        ["p1"],
    )
    assert len(cohort) == 3
    assert set(cohort["bertyp"]) == {"Verlaufseintrag", "Austrittsbericht"}


def test_manual_columns_exist():
    cohort = build_patient_validation_cohort(
        _predictions(),
        _baseline(),
        _patient_matrix(),
        ["p1"],
    )
    for col in (
        "manual_report_ground_truth",
        "manual_patient_ground_truth",
        "reviewer",
    ):
        assert col in cohort.columns
        assert (cohort[col].astype(str).str.strip() == "").all()


def test_missing_baseline_keeps_reports():
    cohort = build_patient_validation_cohort(
        _predictions(),
        None,
        _patient_matrix(),
        ["p2"],
    )
    assert len(cohort) == 1
    assert int(cohort.iloc[0]["missing_structured_baseline"]) == 1
    assert str(cohort.iloc[0]["ICDSC_max"]).strip() == ""


def test_patientenid_int64_merge():
    preds = _predictions().copy()
    preds["PatientenID"] = preds["PatientenID"].replace({"p1": 1, "p2": 2, "p3": 3})
    matrix = _patient_matrix().copy()
    matrix["PatientenID"] = ["1", "2", "3"]
    selected, _ = select_validation_patient_ids(matrix, target_n=1)
    cohort = build_patient_validation_cohort(preds, _baseline(), matrix, selected)
    assert len(cohort) >= 1


def test_balanced_sampling_groups_created():
    matrix = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(8)],
            "baseline_composite": [1, 1, 0, 0, 1, 0, 0, 0],
            "model_patient_positive": [1, 0, 1, 0, 1, 1, 0, 0],
            "n_verlaufseintrag": [1] * 8,
            "n_verlegungsbericht": [0] * 8,
            "n_austrittsbericht": [0] * 8,
            "any_manual_review_candidate": [0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    selected, subset = select_validation_patient_ids(matrix, target_n=6)
    assert len(selected) == 6
    groups = set(subset["suggested_patient_sampling_group"])
    assert "TP_composite" in groups or "FN_composite" in groups


def test_report_warning_when_baseline_positive_report_negative():
    cohort = build_patient_validation_cohort(
        _predictions(),
        _baseline(),
        _patient_matrix(),
        ["p1"],
    )
    neg = cohort[(cohort["model_report_prediction"] == 0) & (cohort["baseline_composite"] == 1)]
    assert len(neg) >= 1
    assert "correctly negative" in str(neg.iloc[0]["report_patient_level_warning"]).lower()


def test_column_order():
    cohort = build_patient_validation_cohort(
        _predictions(),
        _baseline(),
        _patient_matrix(),
        ["p1"],
    )
    assert list(cohort.columns) == COHORT_COLUMNS


def test_main_writes_files(tmp_path, monkeypatch):
    pred = tmp_path / "pred.csv"
    base = tmp_path / "base.csv"
    mat = tmp_path / "mat.csv"
    out = tmp_path / "cohort.csv"
    rep = tmp_path / "report.txt"
    _predictions().to_csv(pred, index=False)
    _baseline().to_csv(base, index=False)
    _patient_matrix().to_csv(mat, index=False)

    import src.analysis.export_patient_validation_cohort as mod

    monkeypatch.setenv("PATIENT_VALIDATION_N", "2")
    mod.main(
        predictions_path=pred,
        baseline_path=base,
        matrix_path=mat,
        output_path=out,
        report_path=rep,
    )
    assert out.exists()
    assert rep.exists()
    df = pd.read_csv(out)
    assert df["PatientenID"].nunique() == 2
