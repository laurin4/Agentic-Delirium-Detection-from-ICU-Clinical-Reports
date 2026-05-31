"""Tests for validation report trace and positive prediction audit."""

import json

import pandas as pd

from src.analysis.audit_all_positive_predictions_matching import (
    run_positive_prediction_matching_audit,
)
from src.analysis.trace_validation_report import trace_validation_report
from src.analysis.validation_report_trace import (
    build_report_trace,
    evaluate_evidence_against_raw,
    load_trace_inputs,
)
from src.analysis.validation_cohort_reports import load_raw_included_report_spine
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL


def _fixtures(tmp_path):
    berichte = tmp_path / "Berichte.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "p1",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "bername": "r1.txt",
                "diag": "Patient ist orientiert und stabil.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "PatientID": "p2",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-02",
                "bername": "r2.txt",
                "diag": "hypoaktives Delir dokumentiert.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    ).to_csv(berichte, index=False, sep=";")

    spine = load_raw_included_report_spine(berichte, patient_ids=["p1", "p2"])
    r1 = spine.iloc[0]
    r2 = spine.iloc[1]

    cohort = tmp_path / "cohort.csv"
    labels = tmp_path / "labels.csv"
    preds = tmp_path / "preds.csv"

    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "PatientenID": ["p1", "p2"],
            SOURCE_REPORT_ROW_ID_COL: [r1[SOURCE_REPORT_ROW_ID_COL], r2[SOURCE_REPORT_ROW_ID_COL]],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "bericht": [r1["pipeline_bericht"], r2["pipeline_bericht"]],
            "pipeline_bericht": [r1["pipeline_bericht"], r2["pipeline_bericht"]],
            "status": ["processed", "processed"],
            "model_report_prediction": [1, 1],
            "evidence_snippets": [
                json.dumps([{"keyword": "orientiert", "text": "orientiert", "evidence_type": "indirect_symptom"}]),
                json.dumps(
                    [{"keyword": "hypoaktives Delir", "text": "hypoaktives Delir", "evidence_type": "direct_delir"}]
                ),
            ],
        }
    ).to_csv(cohort, index=False)

    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "manual_report_ground_truth": [0, 1],
            "manual_comment": ["FP candidate", ""],
        }
    ).to_csv(labels, index=False)

    pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            SOURCE_REPORT_ROW_ID_COL: [r1[SOURCE_REPORT_ROW_ID_COL], r2[SOURCE_REPORT_ROW_ID_COL]],
            "bericht": [r1["pipeline_bericht"], r2["pipeline_bericht"]],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "klasse": [1, 1],
            "status": ["processed", "processed"],
            "llm_called": [1, 1],
            "skipped_reason": ["", ""],
            "evidence_snippets": [
                json.dumps([{"keyword": "orientiert", "text": "orientiert", "evidence_type": "indirect_symptom"}]),
                json.dumps(
                    [{"keyword": "hypoaktives Delir", "text": "hypoaktives Delir", "evidence_type": "direct_delir"}]
                ),
            ],
            "begruendung": ["test", "test2"],
            "kontext": ["k1", "k2"],
        }
    ).to_csv(preds, index=False)

    return berichte, cohort, labels, preds, spine


def test_trace_detects_correct_match(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    c, lab, pr, spine, raw = load_trace_inputs(cohort, labels, preds, berichte)
    trace = build_report_trace("Patient_0002_Report_0001", c, lab, pr, spine, raw)
    assert trace.verdict == "MATCH_OK"
    assert trace.prediction_row is not None
    assert trace.raw_berichte_row is not None


def test_trace_detects_wrong_prediction_match(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    pr = pd.read_csv(preds)
    pr.loc[0, "PatientenID"] = "p2"
    pr.to_csv(preds, index=False)

    c, lab, pr_df, spine, raw = load_trace_inputs(cohort, labels, preds, berichte)
    trace = build_report_trace("Patient_0001_Report_0001", c, lab, pr_df, spine, raw)
    assert trace.verdict == "MATCH_FAIL"
    assert any("PatientenID_mismatch" in i for i in trace.issues)


def test_trace_flags_evidence_keyword_not_in_raw_text(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    c = pd.read_csv(cohort)
    c.loc[0, "evidence_snippets"] = json.dumps(
        [{"keyword": "desorientiert", "text": "desorientiert", "evidence_type": "indirect_symptom"}]
    )
    c.to_csv(cohort, index=False)

    _, lab, pr, spine, raw = load_trace_inputs(cohort, labels, preds, berichte)
    trace = build_report_trace("Patient_0001_Report_0001", c, lab, pr, spine, raw)
    assert trace.verdict == "MATCH_FAIL"
    assert any("evidence_not_in_raw_report" in i for i in trace.issues)


def test_trace_works_with_missing_optional_fields(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    c = pd.read_csv(cohort)
    c = c.drop(columns=[SOURCE_REPORT_ROW_ID_COL], errors="ignore")
    c.to_csv(cohort, index=False)
    pr = pd.read_csv(preds)
    pr = pr.drop(columns=[SOURCE_REPORT_ROW_ID_COL], errors="ignore")
    pr.to_csv(preds, index=False)

    _, lab, pr_df, spine, raw = load_trace_inputs(cohort, labels, preds, berichte)
    trace = build_report_trace("Patient_0001_Report_0001", c, lab, pr_df, spine, raw)
    assert trace.cohort_row is not None
    assert trace.verdict in ("MATCH_OK", "MATCH_SUSPICIOUS", "MATCH_FAIL")


def test_evaluate_evidence_against_raw():
    checks, issues = evaluate_evidence_against_raw(
        json.dumps([{"keyword": "desorientiert", "text": "desorientiert"}]),
        "Patient ist orientiert.",
    )
    assert checks[0]["found_in_raw_report"] is False
    assert issues


def test_trace_validation_report_writes_file(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    out_dir = tmp_path / "traces"
    path = trace_validation_report(
        "Patient_0001_Report_0001",
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        berichte_path=berichte,
        output_dir=out_dir,
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Frozen cohort row" in text
    assert "validation_report_id=Patient_0001_Report_0001" in text


def test_positive_prediction_audit(tmp_path):
    berichte, cohort, labels, preds, _ = _fixtures(tmp_path)
    c = pd.read_csv(cohort)
    c.loc[0, "evidence_snippets"] = json.dumps(
        [{"keyword": "desorientiert", "text": "desorientiert", "evidence_type": "indirect_symptom"}]
    )
    c.to_csv(cohort, index=False)
    out = tmp_path / "pos_audit"
    mismatches, report = run_positive_prediction_matching_audit(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        berichte_path=berichte,
        output_dir=out,
    )
    assert "positive_report_rows=2" in report
    assert (out / "positive_prediction_mismatches.csv").exists()
    assert len(mismatches) >= 1
