"""Validation cohort must include skipped / prefilter-negative reports."""

import pandas as pd

from src.analysis.export_patient_validation_cohort import build_patient_validation_cohort
from src.analysis.validation_cohort_reports import (
    build_complete_validation_reports_frame,
    derive_report_processing_fields,
)
from src.preprocessing.evidence_extraction import METHOD_NO_EVIDENCE


def _matrix():
    return pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "baseline_icd10": [0],
            "baseline_icdsc_ge_4": [0],
            "model_patient_positive": [0],
            "n_verlaufseintrag": [2],
            "n_verlegungsbericht": [0],
            "n_austrittsbericht": [0],
            "any_manual_review_candidate": [0],
        }
    )


def test_skipped_prefilter_report_in_cohort():
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": "r_llm.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-02",
                "klasse": 1,
                "signalstaerke": "hoch",
                "llm_skipped_by_prefilter": False,
                "llm_text_reduction_method": "structured_evidence_extraction",
                "decision_rule_applied": "direct",
                "delir_probability_estimate": 80,
                "manual_review_candidate": "False",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 100,
                "llm_report_text_length": 50,
            },
            {
                "PatientenID": "p1",
                "bericht": "r_skip.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "llm_skipped_by_prefilter": True,
                "llm_text_reduction_method": METHOD_NO_EVIDENCE,
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "delir_probability_estimate": 5,
                "manual_review_candidate": "False",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "Keine verwertbare Delir-Evidenz",
                "begruendung": "",
                "original_report_text_length": 80,
                "llm_report_text_length": 0,
            },
        ]
    )
    berichte = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1"],
            "bername": ["r_llm.txt", "r_skip.txt"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-02", "2024-01-01"],
            "diag": ["Delir", "x"],
        }
    )
    cohort = build_patient_validation_cohort(
        preds,
        None,
        _matrix(),
        ["p1"],
        berichte_reports=berichte,
    )
    assert len(cohort) == 2
    skip = cohort[cohort["bericht"] == "r_skip.txt"].iloc[0]
    assert int(skip["model_report_prediction"]) == 0
    assert skip["status"] == "skipped"
    assert int(skip["llm_called"]) == 0
    assert "no_evidence" in str(skip["skipped_reason"])


def test_berichte_spine_adds_missing_report():
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": "r1.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "klasse": 0,
                "llm_skipped_by_prefilter": True,
                "llm_text_reduction_method": METHOD_NO_EVIDENCE,
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 10,
                "llm_report_text_length": 0,
            }
        ]
    )
    berichte = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1"],
            "bername": ["r1.txt", "r2_only_berichte.txt"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "diag": ["x", "y"],
        }
    )
    merged, stats = build_complete_validation_reports_frame(
        preds, ["p1"], berichte_df=berichte
    )
    assert len(merged) == 2
    assert stats["only_in_berichte"] == 1
    missing = merged[merged["pipeline_bericht"] == "r2_only_berichte.txt"].iloc[0]
    assert missing["status"] == "missing_prediction"
    assert missing["skipped_reason"] == "missing_prediction_implicit_negative"
    assert int(missing["model_report_prediction"]) == 0

    cohort = build_patient_validation_cohort(
        preds,
        None,
        _matrix(),
        ["p1"],
        berichte_reports=berichte,
    )
    assert len(cohort) == 2


def test_derive_processing_fields_processed_guardrail():
    row = pd.Series(
        {
            "_has_prediction_row": True,
            "llm_skipped_by_prefilter": False,
            "llm_text_reduction_method": "structured_evidence_extraction",
            "decision_rule_applied": "prophylaxis_only_not_positive",
            "kontext": "x",
        }
    )
    meta = derive_report_processing_fields(row)
    assert meta["status"] == "processed"
    assert meta["llm_called"] == 1


def test_patient_level_aggregation_includes_skipped_positive():
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": "a.txt",
                "bertyp": "Verlaufseintrag",
                "klasse": 0,
                "llm_skipped_by_prefilter": True,
                "llm_text_reduction_method": METHOD_NO_EVIDENCE,
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 0,
            },
            {
                "PatientenID": "p1",
                "bericht": "b.txt",
                "bertyp": "Austrittsbericht",
                "klasse": 1,
                "llm_skipped_by_prefilter": False,
                "llm_text_reduction_method": "structured_evidence_extraction",
                "decision_rule_applied": "x",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 90,
                "manual_review_candidate": "False",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 1,
            },
        ]
    )
    cohort = build_patient_validation_cohort(preds, None, _matrix(), ["p1"])
    assert int(cohort["model_patient_positive"].iloc[0]) == 1
