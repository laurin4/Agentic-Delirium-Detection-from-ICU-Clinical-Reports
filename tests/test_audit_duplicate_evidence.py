"""Tests for duplicate evidence audit."""

import pandas as pd

from src.analysis.audit_duplicate_evidence import (
    _primary_evidence_text,
    _verdict_for_group,
    run_duplicate_evidence_audit,
    write_duplicate_evidence_audit,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL

SHARED = "Nach Extubation bestand eine unauffaellige Atemmechanik ohne Delirzeichen."


def test_primary_evidence_from_snippets():
    raw = (
        '[{"keyword":"extubation","text":"'
        + SHARED
        + '","evidence_type":"indirect_symptom"}]'
    )
    assert _primary_evidence_text(raw).startswith("Nach Extubation")


def test_verdict_fail_when_not_in_source():
    details = [
        {
            "PatientenID": "p1",
            "in_frozen_cohort_text": False,
            "in_berichte_for_patient": False,
            "likely_template_text": False,
            "evidence_text": SHARED,
        },
        {
            "PatientenID": "p2",
            "in_frozen_cohort_text": True,
            "in_berichte_for_patient": True,
            "likely_template_text": True,
            "evidence_text": SHARED,
        },
    ]
    assert _verdict_for_group(details) == "FAIL"


def test_verdict_warning_template():
    details = [
        {
            "PatientenID": "p1",
            "in_frozen_cohort_text": True,
            "in_berichte_for_patient": True,
            "likely_template_text": True,
            "evidence_text": SHARED,
        },
        {
            "PatientenID": "p2",
            "in_frozen_cohort_text": True,
            "in_berichte_for_patient": True,
            "likely_template_text": True,
            "evidence_text": SHARED,
        },
    ]
    assert _verdict_for_group(details) == "WARNING"


def test_run_duplicate_evidence_audit(tmp_path):
    eval_dir = tmp_path / "final_evaluation"
    eval_dir.mkdir(parents=True)
    cohort_dir = tmp_path / "frozen"
    cohort_dir.mkdir()

    shared_json = (
        '[{"keyword":"extubation","text":"'
        + SHARED
        + '","evidence_type":"indirect_symptom"}]'
    )

    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "PatientenID": ["p1", "p2"],
            "representative_evidence": [shared_json, shared_json],
            "model_patient_positive": [1, 1],
        }
    ).to_csv(eval_dir / "model_FP.csv", index=False)

    pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "PatientenID": ["p1", "p2"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "klasse": [1, 1],
            "evidence_snippets": [shared_json, shared_json],
        }
    ).to_csv(tmp_path / "preds.csv", index=False)

    pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "validation_patient_id": ["Patient_0001", "Patient_0002"],
            "PatientenID": ["p1", "p2"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "report_text": [
                "Anderer Text.",
                SHARED,
            ],
        }
    ).to_csv(cohort_dir / "patient_validation_cohort_frozen.csv", index=False)

    audit_df, report = run_duplicate_evidence_audit(
        fp_path=eval_dir / "model_FP.csv",
        tp_path=eval_dir / "model_TP.csv",
        predictions_path=tmp_path / "preds.csv",
        cohort_path=cohort_dir / "patient_validation_cohort_frozen.csv",
        berichte_path=tmp_path / "no_berichte.csv",
    )
    assert not audit_df.empty
    assert audit_df["duplicate_group_id"].nunique() >= 1
    assert "OVERALL_VERDICT" in report
    assert audit_df["PatientenID"].nunique() >= 2


def test_write_outputs(tmp_path):
    eval_dir = tmp_path / "final_evaluation"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "PatientenID": ["p1"],
            "representative_evidence": ["[]"],
            "model_patient_positive": [1],
        }
    ).to_csv(eval_dir / "model_FP.csv", index=False)
    pd.DataFrame(
        {
            VALIDATION_REPORT_ID_COL: ["Patient_0001_Report_0001"],
            "validation_patient_id": ["Patient_0001"],
            "PatientenID": ["p1"],
            "klasse": [0],
            "evidence_snippets": ["[]"],
        }
    ).to_csv(tmp_path / "preds.csv", index=False)

    _, report = write_duplicate_evidence_audit(
        fp_path=eval_dir / "model_FP.csv",
        tp_path=eval_dir / "model_TP.csv",
        predictions_path=tmp_path / "preds.csv",
        cohort_path=tmp_path / "missing_cohort.csv",
        output_csv=eval_dir / "duplicate_evidence_audit.csv",
        output_report=eval_dir / "duplicate_evidence_audit_report.txt",
    )
    assert (eval_dir / "duplicate_evidence_audit.csv").exists()
    assert "PASS" in report
