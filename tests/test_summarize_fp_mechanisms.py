"""Tests for FP mechanism summary."""

import pandas as pd

from src.analysis.summarize_fp_mechanisms import (
    _classify_report_row,
    build_fp_mechanism_summary,
    run_fp_mechanism_summary,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL


def _preds_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: [
                "Patient_0001_Report_0001",
                "Patient_0001_Report_0002",
                "Patient_0002_Report_0001",
            ],
            "validation_patient_id": ["Patient_0001", "Patient_0001", "Patient_0002"],
            "PatientenID": ["p1", "p1", "p2"],
            "klasse": [1, 0, 1],
            "decision_rule_applied": ["llm_classification", "no_evidence_prefilter_skip", "direct_delir_positive"],
            "signalstaerke": ["mittel", "niedrig", "hoch"],
            "llm_text_reduction_method": [
                "short_report_no_evidence_fulltext",
                "no_evidence_prefilter_skip",
                "structured_evidence_extraction",
            ],
            "llm_called": [1, 0, 1],
            "evidence_snippets": [
                "[]",
                "[]",
                '[{"keyword":"delir","text":"Delir dokumentiert","evidence_type":"direct_delir"}]',
            ],
            "delir_signale": ["desorientiert", "", "delir"],
            "kontext": ["", "", ""],
            "begruendung": ["", "", ""],
        }
    )


def test_classify_short_report_fulltext():
    row = pd.Series(
        {
            "llm_text_reduction_method": "short_report_no_evidence_fulltext",
            "evidence_snippets": "[]",
            "decision_rule_applied": "llm_classification",
            "delir_signale": "desorientiert",
        }
    )
    assert _classify_report_row(row) == "short_report_fulltext_llm_positive"


def test_classify_direct_delir():
    row = pd.Series(
        {
            "llm_text_reduction_method": "structured_evidence_extraction",
            "evidence_snippets": '[{"keyword":"delir","evidence_type":"direct_delir","text":"x"}]',
            "decision_rule_applied": "direct_delir_positive",
        }
    )
    assert _classify_report_row(row) == "direct_delir_mention_but_manual_negative"


def test_build_fp_mechanism_summary(tmp_path):
    fp = pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "PatientenID": ["p1", "p2"],
            "derived_manual_patient_ground_truth": [0, 0],
            "model_patient_positive": [1, 1],
        }
    )
    preds = _preds_df()
    summary = build_fp_mechanism_summary(fp, preds)
    assert len(summary) == 2
    p1 = summary[summary["validation_patient_id"] == "Patient_0001"].iloc[0]
    assert p1["n_positive_model_reports"] == 1
    assert "Patient_0001_Report_0001" in p1["positive_validation_report_ids"]
    assert p1["evidence_snippets_empty_count"] == 1
    p2 = summary[summary["validation_patient_id"] == "Patient_0002"].iloc[0]
    assert p2["suggested_error_category"] == "direct_delir_mention_but_manual_negative"


def test_run_fp_mechanism_summary_writes_files(tmp_path):
    eval_dir = tmp_path / "final_evaluation"
    eval_dir.mkdir(parents=True)
    fp_path = eval_dir / "model_FP.csv"
    preds_path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "PatientenID": ["p1"],
            "derived_manual_patient_ground_truth": [0],
            "model_patient_positive": [1],
        }
    ).to_csv(fp_path, index=False)
    _preds_df().iloc[:1].to_csv(preds_path, index=False)

    summary, report = run_fp_mechanism_summary(
        fp_path=fp_path,
        predictions_path=preds_path,
        cohort_path=tmp_path / "missing_cohort.csv",
        output_csv=eval_dir / "fp_mechanism_summary.csv",
        output_report=eval_dir / "fp_mechanism_report.txt",
    )
    assert (eval_dir / "fp_mechanism_summary.csv").exists()
    assert "fp_patients=1" in report
    assert summary.iloc[0]["n_positive_model_reports"] == 1
