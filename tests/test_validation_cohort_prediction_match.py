"""Prediction ↔ cohort merge must preserve pipeline status (not missing_prediction)."""

from pathlib import Path

import pandas as pd

from src.analysis.validation_cohort_reports import (
    build_complete_validation_reports_frame,
    load_raw_included_report_spine,
)
from src.preprocessing.evidence_extraction import METHOD_NO_EVIDENCE
from src.preprocessing.report_identity import PIPELINE_BERICHT_COL, SOURCE_REPORT_ROW_ID_COL


def _load_spine_df(raw: pd.DataFrame) -> pd.DataFrame:
    return load_raw_included_report_spine(Path("."), berichte_df=raw)


def test_source_report_row_id_merge_preserves_processed_status():
    raw = pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "bericht": "display.txt",
            "bername": "pipeline_doc_1",
            "bertyp": "Verlaufseintrag",
            "berdat": "2024-05-01",
            "diag": "Delir",
        }
    )
    spine = _load_spine_df(raw)
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": spine.iloc[0][PIPELINE_BERICHT_COL],
                SOURCE_REPORT_ROW_ID_COL: spine.iloc[0][SOURCE_REPORT_ROW_ID_COL],
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-05-01",
                "klasse": 1,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "direct_delir_evidence",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 80,
                "manual_review_candidate": "False",
                "decision_rule_applied": "direct",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 10,
                "llm_report_text_length": 5,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        ]
    )
    merged, stats = build_complete_validation_reports_frame(preds, ["p1"], berichte_df=spine)
    assert stats["prediction_matched_reports"] == 1
    assert merged.iloc[0]["status"] == "processed"
    assert int(merged.iloc[0]["llm_called"]) == 1
    assert merged.iloc[0]["skipped_reason"] == "direct_delir_evidence"


def test_skipped_prediction_not_missing_prediction():
    raw = pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "bername": "skip_doc",
            "bertyp": "Verlaufseintrag",
            "berdat": "2024-05-02",
            "diag": "x",
        }
    )
    spine = _load_spine_df(raw)
    pber = spine.iloc[0][PIPELINE_BERICHT_COL]
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": pber,
                "bertyp": "Verlaufseintrag",
                "klasse": 0,
                "status": "skipped",
                "llm_called": 0,
                "skipped_reason": "no_evidence_prefilter_skip",
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
    merged, _ = build_complete_validation_reports_frame(preds, ["p1"], berichte_df=spine)
    row = merged.iloc[0]
    assert row["status"] == "skipped"
    assert row["skipped_reason"] != "missing_prediction_implicit_negative"


def test_legacy_pipeline_bericht_merge_without_source_id():
    raw = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1"],
            "bername": ["v1", "v2"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-06-01", "2024-06-02"],
            "diag": ["a", "b"],
        }
    )
    spine = _load_spine_df(raw)
    assert len(spine) == 2
    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": spine.iloc[0][PIPELINE_BERICHT_COL],
                "bertyp": "Verlaufseintrag",
                "klasse": 1,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "x",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 50,
                "manual_review_candidate": "False",
                "decision_rule_applied": "x",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 1,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        ]
    )
    merged, stats = build_complete_validation_reports_frame(preds, ["p1"], berichte_df=spine)
    assert len(merged) == 2
    assert stats["prediction_matched_reports"] == 1
    assert int((merged["status"] == "missing_prediction").sum()) == 1
    assert len(merged[merged["status"] == "processed"]) == 1


def test_seven_raw_two_predictions_pipeline_bericht_match():
    pid = "p7"
    raw_rows = [
        {
            "PatientenID": pid,
            "bername": f"doc_{i}",
            "bertyp": "Verlaufseintrag",
            "berdat": f"2024-07-{i + 1:02d}",
            "diag": "text",
        }
        for i in range(7)
    ]
    spine = _load_spine_df(pd.DataFrame(raw_rows))
    pred_rows = []
    for i in (0, 3):
        pred_rows.append(
            {
                "PatientenID": pid,
                "bericht": spine.iloc[i][PIPELINE_BERICHT_COL],
                "bertyp": "Verlaufseintrag",
                "klasse": i,
                "status": "skipped" if i == 0 else "processed",
                "llm_called": 0 if i == 0 else 1,
                "skipped_reason": "no_evidence_prefilter_skip" if i == 0 else "direct",
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "decision_rule_applied": "x",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 0,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        )
    merged, stats = build_complete_validation_reports_frame(
        pd.DataFrame(pred_rows), [pid], berichte_df=spine
    )
    assert len(merged) == 7
    assert stats["prediction_matched_reports"] == 2
    assert merged.iloc[0]["status"] == "skipped"
    assert merged.iloc[3]["status"] == "processed"
