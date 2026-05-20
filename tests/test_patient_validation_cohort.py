"""Tests for PRIMARY patient-level manual validation cohort export."""

import json

import pandas as pd

from src.analysis.export_patient_validation_cohort import (
    COHORT_COLUMNS,
    build_patient_validation_cohort,
    load_patient_level_context,
    select_validation_patient_ids,
)
from src.analysis.manual_validation_eval import (
    DERIVED_PATIENT_GT_COL,
    derive_patient_manual_labels,
)
from src.analysis.validation_ids import format_validation_patient_id, format_validation_report_id
from src.preprocessing.berichte_filters import DOKUMENTATIONSBLATT_BERTYP


def _predictions() -> pd.DataFrame:
    rows = []
    for pid, klasse, berdat in [
        ("p1", 1, "2024-01-02"),
        ("p1", 0, "2024-01-01"),
        ("p2", 0, "2024-02-01"),
    ]:
        rows.append(
            {
                "PatientenID": pid,
                "bericht": f"{pid}_v_{berdat}.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": berdat,
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
            "berdat": "2024-01-03",
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
            "berdat": "2024-03-01",
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
            "baseline_icd10": [0, 0, 0],
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


def test_validation_ids_deterministic():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1", "p2"]
    )
    assert cohort.iloc[0]["validation_patient_id"] == format_validation_patient_id(1)
    assert cohort.iloc[0]["validation_report_id"] == format_validation_report_id(
        format_validation_patient_id(1), 1
    )
    assert list(cohort["validation_patient_id"].unique()) == [
        format_validation_patient_id(1),
        format_validation_patient_id(2),
    ]


def test_chronological_order_by_berdat():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1"]
    )
    p1 = cohort[cohort["PatientenID"] == "p1"].reset_index(drop=True)
    dates = pd.to_datetime(p1["berdat"], errors="coerce")
    assert dates.is_monotonic_increasing
    assert list(p1["report_nr_within_patient"]) == [1, 2, 3]


def test_model_patient_positive_is_max_report():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1"]
    )
    assert (cohort["model_patient_positive"] == 1).all()


def test_derived_manual_patient_ground_truth_from_reports():
    df = pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"] * 3,
            "manual_report_ground_truth": [0, 1, 0],
        }
    )
    out = derive_patient_manual_labels(df)
    assert int(out[DERIVED_PATIENT_GT_COL].iloc[0]) == 1
    assert int(out["n_positive_reports_manual"].iloc[0]) == 1


def test_manual_columns_no_required_patient_gt():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1"]
    )
    assert "manual_report_ground_truth" in cohort.columns
    assert "manual_patient_ground_truth" not in cohort.columns
    assert (cohort["manual_report_ground_truth"].astype(str).str.strip() == "").all()


def test_baseline_composite_or_and_columns():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1"]
    )
    assert "baseline_composite_or" in cohort.columns
    assert "baseline_composite_and" in cohort.columns
    row = cohort.iloc[0]
    assert int(row["baseline_composite_or"]) == 1
    assert int(row["baseline_composite_and"]) == 0


def test_missing_baseline_keeps_reports():
    cohort = build_patient_validation_cohort(
        _predictions(),
        None,
        _patient_matrix(),
        ["p2"],
    )
    assert len(cohort) == 1
    assert int(cohort.iloc[0]["missing_structured_baseline"]) == 1


def test_matrix_icdsc_max_enables_icdsc_positive_sampling_bucket(tmp_path):
    """Legacy matrix CSV without baseline_icdsc_ge_4 still fills bucket via ICDSC_max >= 4."""
    legacy_matrix = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(10)],
            "ICDSC_max": [5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
            "ICD10": [0] * 10,
            "model_patient_positive": [0] * 10,
            "n_verlaufseintrag": [1] * 10,
            "n_verlegungsbericht": [0] * 10,
            "n_austrittsbericht": [0] * 10,
            "any_manual_review_candidate": [0] * 10,
        }
    )
    mat_path = tmp_path / "matrix_legacy.csv"
    legacy_matrix.to_csv(mat_path, index=False)

    preds = _predictions()
    baseline = _baseline()
    ctx = load_patient_level_context(mat_path, preds, baseline)
    assert "baseline_icdsc_ge_4" in ctx.columns
    assert int(ctx.loc[ctx["PatientenID"] == "p0", "baseline_icdsc_ge_4"].iloc[0]) == 1

    selected, subset = select_validation_patient_ids(ctx, target_n=6)
    assert len(selected) == 6
    assert (subset["baseline_icdsc_ge_4"] == 1).any()


def test_validation_sampling_includes_icdsc_positive_from_built_matrix():
    preds = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(10)],
            "bertyp": ["Verlaufseintrag"] * 10,
            "klasse": [0] * 10,
        }
    )
    baseline = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(10)],
            "max_icdsc": [5, 5, 4, 0, 0, 0, 0, 0, 0, 0],
            "baseline_icd10": [0] * 10,
            "baseline_composite": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    from src.analysis.patient_reporttype_matrix import build_patient_reporttype_matrix

    matrix = build_patient_reporttype_matrix(preds, baseline)
    assert (matrix["baseline_icdsc_ge_4"] == 1).sum() >= 3
    selected, subset = select_validation_patient_ids(matrix, target_n=6)
    assert (subset["baseline_icdsc_ge_4"] == 1).any()


def test_balanced_sampling_groups():
    matrix = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(10)],
            "baseline_icd10": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "baseline_icdsc_ge_4": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            "model_patient_positive": [1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
            "n_verlaufseintrag": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            "n_verlegungsbericht": [0] * 10,
            "n_austrittsbericht": [0] * 10,
            "any_manual_review_candidate": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        }
    )
    selected, _ = select_validation_patient_ids(matrix, target_n=6)
    assert len(selected) == 6


def test_column_order():
    cohort = build_patient_validation_cohort(
        _predictions(), _baseline(), _patient_matrix(), ["p1"]
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
    assert df["validation_patient_id"].nunique() == 2
