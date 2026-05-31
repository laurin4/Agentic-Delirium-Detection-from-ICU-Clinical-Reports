"""Tests for validation matching audit."""

import json

import pandas as pd
import pytest

from src.analysis.audit_validation_matching import (
    check_evidence_in_report,
    check_patient_report_counts,
    check_validation_report_id_uniqueness,
    compute_verdict,
    run_matching_audit,
    text_contains_phrase,
)
from src.analysis.validation_cohort_reports import load_raw_included_report_spine
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL


def _write_berichte(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, sep=";")


def _minimal_audit_fixtures(tmp_path):
    berichte = tmp_path / "Berichte.csv"
    _write_berichte(
        berichte,
        [
            {
                "PatientID": "p1",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "bername": "r1.txt",
                "diag": "Patient stabil ohne Delirzeichen.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "PatientID": "p2",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-02",
                "bername": "r2.txt",
                "diag": "hypoaktives Delir bei Pneumonie.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ],
    )
    spine = load_raw_included_report_spine(berichte, patient_ids=["p1", "p2"])
    row_p1 = spine.iloc[0]
    row_p2 = spine.iloc[1]

    cohort = tmp_path / "cohort.csv"
    labels = tmp_path / "labels.csv"
    preds = tmp_path / "preds.csv"
    out = tmp_path / "audit_out"

    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002", "Patient_0003"],
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0002_Report_0001",
                "Patient_0003_Report_0001",
            ],
            "PatientenID": ["p1", "p2", "p9"],
            SOURCE_REPORT_ROW_ID_COL: [
                row_p1[SOURCE_REPORT_ROW_ID_COL],
                row_p2[SOURCE_REPORT_ROW_ID_COL],
                "berichte_row_missing",
            ],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "bericht": [row_p1.get("pipeline_bericht", "r1.txt"), row_p2.get("pipeline_bericht", "r2.txt"), "x"],
            "pipeline_bericht": [
                row_p1.get("pipeline_bericht", "r1.txt"),
                row_p2.get("pipeline_bericht", "r2.txt"),
                "x",
            ],
            "status": ["processed", "processed", "missing_prediction"],
            "evidence_snippets": [
                json.dumps([{"keyword": "Delir", "text": "Delirzeichen", "evidence_type": "direct_delir"}]),
                json.dumps(
                    [
                        {
                            "keyword": "hypoaktives Delir",
                            "text": "hypoaktives Delir bei Pneumonie",
                            "evidence_type": "direct_delir",
                        }
                    ]
                ),
                "[]",
            ],
        }
    ).to_csv(cohort, index=False)

    pd.DataFrame(
        {
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0002_Report_0001",
                "Patient_0003_Report_0001",
            ],
            "PatientenID": ["p1", "p2", "p9"],
            "manual_report_ground_truth": [0, 1, ""],
        }
    ).to_csv(labels, index=False)

    pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            SOURCE_REPORT_ROW_ID_COL: [
                row_p1[SOURCE_REPORT_ROW_ID_COL],
                row_p2[SOURCE_REPORT_ROW_ID_COL],
            ],
            "bericht": [
                row_p1.get("pipeline_bericht", "r1.txt"),
                row_p2.get("pipeline_bericht", "r2.txt"),
            ],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "klasse": [0, 1],
            "evidence_snippets": [
                json.dumps([{"keyword": "Delir", "text": "Delirzeichen", "evidence_type": "direct_delir"}]),
                json.dumps(
                    [
                        {
                            "keyword": "hypoaktives Delir",
                            "text": "hypoaktives Delir bei Pneumonie",
                            "evidence_type": "direct_delir",
                        }
                    ]
                ),
            ],
            "status": ["processed", "processed"],
        }
    ).to_csv(preds, index=False)

    return berichte, cohort, labels, preds, out, spine


def test_correct_evidence_passes(tmp_path):
    berichte, cohort, labels, preds, out, spine = _minimal_audit_fixtures(tmp_path)
    from src.analysis.audit_validation_matching import build_report_text_index

    by_sid, by_fb = build_report_text_index(spine)
    cohort_df = pd.read_csv(cohort)
    _, high_risk, checked, _, _ = check_evidence_in_report(cohort_df.iloc[[0]], by_sid, by_fb)
    assert checked == 1
    assert high_risk == []


def test_wrong_report_evidence_fails(tmp_path):
    berichte, cohort, labels, preds, out, _ = _minimal_audit_fixtures(tmp_path)
    cohort_df = pd.read_csv(cohort)
    cohort_df.loc[0, "evidence_snippets"] = json.dumps(
        [
            {
                "keyword": "hypoaktives Delir",
                "text": "hypoaktives Delir",
                "evidence_type": "direct_delir",
            }
        ]
    )
    cohort_df.to_csv(cohort, index=False)
    result = run_matching_audit(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        berichte_path=berichte,
        output_dir=out,
    )
    assert result.high_risk_mismatch_count >= 1
    assert result.verdict == "FAIL"
    assert (out / "sample_mismatch_cases.csv").exists()


def test_duplicate_report_keys_detected():
    cohort = pd.DataFrame(
        {
            "validation_report_id": ["R1", "R1"],
            "PatientenID": ["p1", "p1"],
        }
    )
    issues = check_validation_report_id_uniqueness(cohort, pd.DataFrame())
    assert len(issues) == 1
    assert issues[0]["issue_type"] == "duplicate_validation_report_id"


def test_patient_report_count_mismatch_detected(tmp_path):
    berichte, _, _, _, _, spine = _minimal_audit_fixtures(tmp_path)
    cohort_df = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1"],
            "validation_report_id": ["R1", "R2"],
        }
    )
    mismatches = check_patient_report_counts(cohort_df, spine)
    assert any(m["PatientenID"] == "p1" for m in mismatches)


def test_missing_prediction_excluded_from_evidence_check(tmp_path):
    berichte, cohort, labels, preds, out, _ = _minimal_audit_fixtures(tmp_path)
    result = run_matching_audit(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        berichte_path=berichte,
        output_dir=out,
    )
    assert result.evidence_rows_checked == 2
    checked_ids = {r.get("validation_report_id") for r in result.evidence_not_found}
    assert "Patient_0003_Report_0001" not in checked_ids


def test_text_contains_phrase_whitespace_insensitive():
    assert text_contains_phrase("hypoaktives  Delir\nbei Infekt", "hypoaktives Delir")


def test_verdict_pass_when_clean():
    r = type("R", (), {})()
    from src.analysis.audit_validation_matching import AuditResult

    result = AuditResult(
        evidence_not_found_count=0,
        high_risk_mismatch_count=0,
        patient_report_count_mismatches=0,
        duplicate_key_issues=0,
        patientenid_label_mismatches=0,
        prediction_integrity_failures=0,
    )
    assert compute_verdict(result) == "PASS"
