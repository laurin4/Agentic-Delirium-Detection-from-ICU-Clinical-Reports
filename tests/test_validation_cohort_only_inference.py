"""VALIDATION_COHORT_ONLY mode: inference limited to frozen cohort reports."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.export_patient_validation_cohort import (
    build_patient_validation_cohort,
    resolve_predictions_path_for_export,
)
from src.analysis.validation_cohort_reports import build_complete_validation_reports_frame
from src.pipeline import run_pipeline
from src.pipeline.paths import VALIDATION_COHORT_PREDICTIONS_PATH
from src.pipeline.validation_cohort_filter import (
    build_cohort_filter_spec,
    filter_report_records_for_validation_cohort,
    validation_cohort_only_enabled,
)
from src.preprocessing.evidence_extraction import METHOD_NO_EVIDENCE
from src.preprocessing.report_identity import PIPELINE_BERICHT_COL, SOURCE_REPORT_ROW_ID_COL


def _frozen_cohort_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0001", "Patient_0002"],
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0001_Report_0002",
                "Patient_0002_Report_0001",
            ],
            "PatientenID": ["p1", "p1", "p2"],
            SOURCE_REPORT_ROW_ID_COL: [
                "berichte_row_0",
                "berichte_row_1",
                "berichte_row_5",
            ],
            "bericht": ["doc_a", "doc_b", "doc_c"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02", "2024-02-01"],
            "model_report_prediction": [0, 1, 0],
        }
    )


def _all_report_records() -> list[dict]:
    rows = []
    entries = [
        ("p1", "doc_a", "Verlaufseintrag", "2024-01-01", "berichte_row_0"),
        ("p1", "doc_b", "Austrittsbericht", "2024-01-02", "berichte_row_1"),
        ("p2", "doc_c", "Verlaufseintrag", "2024-02-01", "berichte_row_5"),
        ("p3", "doc_other", "Verlaufseintrag", "2024-03-01", "berichte_row_9"),
    ]
    for pid, bername, bertyp, berdat, sid in entries:
        rows.append(
            {
                "PatientenID": pid,
                "bericht": bername,
                "bertyp": bertyp,
                "berdat": berdat,
                SOURCE_REPORT_ROW_ID_COL: sid,
                "report_text": "[Diagnosen]\nDelir Verdacht",
            }
        )
    return rows


def test_validation_cohort_only_enabled_env(monkeypatch):
    monkeypatch.delenv("VALIDATION_COHORT_ONLY", raising=False)
    assert validation_cohort_only_enabled() is False
    monkeypatch.setenv("VALIDATION_COHORT_ONLY", "true")
    assert validation_cohort_only_enabled() is True


def test_filter_by_source_report_row_id():
    frozen = _frozen_cohort_df()
    spec = build_cohort_filter_spec(frozen)
    assert spec.filter_mode == "source_report_row_id"
    all_recs = _all_report_records()
    filtered, _ = filter_report_records_for_validation_cohort(all_recs, cohort_df=frozen)
    assert len(filtered) == 3
    assert {r[SOURCE_REPORT_ROW_ID_COL] for r in filtered} == {
        "berichte_row_0",
        "berichte_row_1",
        "berichte_row_5",
    }


def test_filter_fallback_without_source_id():
    frozen = _frozen_cohort_df().drop(columns=[SOURCE_REPORT_ROW_ID_COL])
    all_recs = _all_report_records()
    filtered, spec = filter_report_records_for_validation_cohort(all_recs, cohort_df=frozen)
    assert len(filtered) == 3
    assert spec.filter_mode == "patientenid_bertyp_berdat_bericht"


def test_get_output_path_cohort_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VALIDATION_COHORT_ONLY", "true")
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(
        run_pipeline,
        "VALIDATION_COHORT_PREDICTIONS_PATH",
        tmp_path / "predictions" / "validation_cohort_predictions.csv",
    )
    assert run_pipeline._get_output_path().name == "validation_cohort_predictions.csv"


def test_run_pipeline_cohort_only_writes_separate_file(monkeypatch, tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    full_pred = pred_dir / "agent1_agent2_agent3_results_prompt.csv"
    full_pred.write_text("PatientenID,bericht\nold,row\n", encoding="utf-8")

    monkeypatch.setenv("VALIDATION_COHORT_ONLY", "true")
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)
    monkeypatch.setattr(
        run_pipeline,
        "VALIDATION_COHORT_PREDICTIONS_PATH",
        pred_dir / "validation_cohort_predictions.csv",
    )
    monkeypatch.setattr(run_pipeline, "FROZEN_PATIENT_VALIDATION_COHORT_PATH", tmp_path / "frozen.csv")
    _frozen_cohort_df().to_csv(tmp_path / "frozen.csv", index=False)

    records = [
        {
            "PatientenID": "p1",
            "bericht": "doc_a",
            "bertyp": "Verlaufseintrag",
            "berdat": "2024-01-01",
            SOURCE_REPORT_ROW_ID_COL: "berichte_row_0",
            "report_text": "[Diagnosen]\nx",
        }
    ]
    monkeypatch.setattr(run_pipeline, "_get_report_records", lambda: records)

    def _fake_run(report, idx, total):
        row = {
            "PatientenID": report["PatientenID"],
            "bericht": report["bericht"],
            "bertyp": report["bertyp"],
            "klasse": 0,
            "status": "skipped",
            "llm_called": 0,
            "skipped_reason": "no_evidence_prefilter_skip",
            "signalstaerke": "niedrig",
            "delir_probability_estimate": 0,
            "manual_review_candidate": "False",
            "decision_rule_applied": "no_evidence_prefilter_skip",
            "evidence_snippets": "[]",
            "delir_signale": "",
            "kontext": "",
            "begruendung": "",
            "original_report_text_length": 1,
            "llm_report_text_length": 0,
            "llm_text_reduction_method": METHOD_NO_EVIDENCE,
            "llm_skipped_by_prefilter": True,
            "anzahl_treffer": 0,
            "alternative_erklaerung": False,
            "alternative_erklaerung_keywords": "",
            "klassifikation": "kein_delir",
            "klassifikation_begruendung": "x",
            "has_direct_delir_evidence": "False",
            "has_indirect_delir_evidence": "False",
            "has_negated_delir_evidence": "False",
            "has_prophylaxis_or_risk_only": "False",
            "has_alternative_explanation": "False",
            "delir_keyword_hits_count": 0,
        }
        return row, True, False

    monkeypatch.setattr(run_pipeline, "_run_single_report", _fake_run)
    run_pipeline.main()

    assert (pred_dir / "validation_cohort_predictions.csv").exists()
    out = pd.read_csv(pred_dir / "validation_cohort_predictions.csv")
    assert len(out) == 1
    assert out.iloc[0]["status"] == "skipped"
    assert "old" in full_pred.read_text(encoding="utf-8")


def test_resolve_predictions_prefers_cohort_file(tmp_path, monkeypatch):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    full_path = pred_dir / "agent1_agent2_agent3_results_prompt.csv"
    cohort_path = pred_dir / "validation_cohort_predictions.csv"
    full_path.write_text("x", encoding="utf-8")
    cohort_path.write_text("y", encoding="utf-8")

    import src.analysis.export_patient_validation_cohort as mod

    monkeypatch.setattr(mod, "PREDICTIONS_DIR", pred_dir)
    monkeypatch.setattr(mod, "VALIDATION_COHORT_PREDICTIONS_PATH", cohort_path)
    assert resolve_predictions_path_for_export() == cohort_path


def test_cohort_predictions_improve_match_rate():
    frozen = _frozen_cohort_df()
    spine = frozen.copy()
    spine[PIPELINE_BERICHT_COL] = spine["bericht"]

    legacy_preds = pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "bericht": "wrong_id",
            "bertyp": "Verlaufseintrag",
            "klasse": 0,
        }
    )
    _, legacy_stats = build_complete_validation_reports_frame(
        legacy_preds, ["p1", "p2"], berichte_df=spine
    )

    cohort_preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": "doc_a",
                SOURCE_REPORT_ROW_ID_COL: "berichte_row_0",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "klasse": 0,
                "status": "skipped",
                "llm_called": 0,
                "skipped_reason": "no_evidence_prefilter_skip",
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 0,
                "llm_text_reduction_method": METHOD_NO_EVIDENCE,
            },
            {
                "PatientenID": "p1",
                "bericht": "doc_b",
                SOURCE_REPORT_ROW_ID_COL: "berichte_row_1",
                "bertyp": "Austrittsbericht",
                "berdat": "2024-01-02",
                "klasse": 1,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "direct",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 80,
                "manual_review_candidate": "False",
                "decision_rule_applied": "direct",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 1,
                "llm_text_reduction_method": "structured_evidence_extraction",
            },
        ]
    )
    _, cohort_stats = build_complete_validation_reports_frame(
        cohort_preds, ["p1", "p2"], berichte_df=spine
    )

    assert cohort_stats["prediction_matched_reports"] > legacy_stats["prediction_matched_reports"]
    assert cohort_stats["missing_prediction_reports"] < legacy_stats["missing_prediction_reports"]
