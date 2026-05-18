"""Tests for report-level manual annotation sheet export."""

import pandas as pd

from src.analysis.export_manual_annotation_sheet import (
    ANNOTATION_SHEET_COLUMNS,
    build_manual_annotation_sheet,
)
from src.preprocessing.berichte_filters import DOKUMENTATIONSBLATT_BERTYP


def _predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": ["p1", "p1", "p2", "p2"],
            "bericht": ["r1.txt", "r2.txt", "r3.txt", "r4.txt"],
            "bertyp": [
                "Verlaufseintrag",
                "Austrittsbericht",
                "Verlaufseintrag",
                DOKUMENTATIONSBLATT_BERTYP,
            ],
            "klasse": [1, 0, 0, 1],
            "klassifikation": ["delir", "kein_delir", "kein_delir", "delir"],
            "signalstaerke": ["mittel", "niedrig", "niedrig", "hoch"],
            "delir_probability_estimate": [50, 10, 5, 80],
            "manual_review_candidate": ["False"] * 4,
            "decision_rule_applied": ["", "", "", ""],
            "evidence_snippets": ["[]"] * 4,
            "delir_signale": ["", "", "", ""],
            "kontext": ["k1", "k2", "k3", "k4"],
            "begruendung": ["b1", "b2", "b3", "b4"],
            "original_report_text_length": [100, 200, 300, 400],
            "llm_report_text_length": [50, 0, 0, 0],
            "llm_text_reduction_method": ["evidence"] * 4,
            "llm_skipped_by_prefilter": ["False", "True", "True", "False"],
            "has_direct_delir_evidence": ["True", "False", "False", "True"],
            "has_indirect_delir_evidence": ["False"] * 4,
            "has_negated_delir_evidence": ["False"] * 4,
            "has_prophylaxis_or_risk_only": ["False"] * 4,
            "has_alternative_explanation": ["False"] * 4,
        }
    )


def _baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": ["p1", "p3"],
            "max_icdsc": [5, 2],
            "baseline_icd10": [0, 1],
            "baseline_icdsc_ge_4": [1, 0],
            "baseline_composite": [1, 1],
        }
    )


def test_multiple_reports_per_patient():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    assert len(sheet) == 3
    assert (sheet["PatientenID"] == "p1").sum() == 2


def test_patient_level_aggregation_columns():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    p1 = sheet[sheet["PatientenID"] == "p1"].iloc[0]
    assert int(p1["model_any_verlaufseintrag"]) == 1
    assert int(p1["model_any_austrittsbericht"]) == 0
    assert int(p1["model_patient_positive"]) == 1


def test_manual_columns_exist_and_empty():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    for col in (
        "manual_report_ground_truth",
        "manual_patient_ground_truth",
        "manual_discrepancy_type",
        "reviewer",
    ):
        assert col in sheet.columns
        assert (sheet[col].astype(str).str.strip() == "").all()


def test_dokumentationsblatt_excluded():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    assert DOKUMENTATIONSBLATT_BERTYP not in set(sheet["bertyp"])


def test_baseline_missing_keeps_report_row():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    p2 = sheet[sheet["PatientenID"] == "p2"].iloc[0]
    assert int(p2["missing_structured_baseline"]) == 1
    assert str(p2["ICDSC_max"]).strip() == ""
    assert str(p2["baseline_composite"]).strip() == ""


def test_patientenid_int64_vs_object_merge():
    preds = _predictions_df().copy()
    preds["PatientenID"] = pd.Series([100, 100, 200, 200], dtype="int64")
    baseline = _baseline_df().copy()
    baseline["PatientenID"] = ["100", "300"]
    sheet = build_manual_annotation_sheet(preds, baseline)
    assert len(sheet) == 3
    assert set(sheet["PatientenID"].astype(str)) == {"100", "200"}


def test_column_order_matches_spec():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    assert list(sheet.columns) == ANNOTATION_SHEET_COLUMNS


def test_report_patient_level_warning():
    sheet = build_manual_annotation_sheet(_predictions_df(), _baseline_df())
    p1_neg = sheet[(sheet["PatientenID"] == "p1") & (sheet["model_report_prediction"] == 0)]
    assert len(p1_neg) == 1
    assert "patient-level" in str(p1_neg.iloc[0]["report_patient_level_warning"]).lower()


def test_main_writes_files(tmp_path, monkeypatch):
    pred = tmp_path / "pred.csv"
    base = tmp_path / "base.csv"
    out = tmp_path / "sheet.csv"
    rep = tmp_path / "report.txt"
    _predictions_df().to_csv(pred, index=False)
    _baseline_df().to_csv(base, index=False)

    import src.analysis.export_manual_annotation_sheet as mod

    mod.main(
        predictions_path=pred,
        baseline_path=base,
        output_path=out,
        report_path=rep,
    )
    assert out.exists()
    assert rep.exists()
    assert "rows=" in rep.read_text(encoding="utf-8")
