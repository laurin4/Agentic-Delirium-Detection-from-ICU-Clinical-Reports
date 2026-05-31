"""
Reproduce MATCH_FAIL from stale positional source_report_row_id.

Simulates: pipeline ran on Berichte v1; cohort spine built from Berichte v2 where
berichte_row_<N> now refers to a different clinical report.
"""

import json

import pandas as pd

from src.analysis.validation_cohort_reports import build_complete_validation_reports_frame
from src.analysis.validation_report_trace import build_report_trace, load_trace_inputs
from src.preprocessing.report_identity import PIPELINE_BERICHT_COL, SOURCE_REPORT_ROW_ID_COL


def test_match_fail_stale_positional_source_report_row_id(tmp_path):
    """
    Pipeline prediction targets Verlauf (desorientiert evidence).
    Cohort spine rebuilt after Berichte insert shifts berichte_row_0 to another patient/report.
    Merge by source_report_row_id attaches stale prediction → MATCH_FAIL.
    """
    berichte_v1 = tmp_path / "Berichte_v1.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "p1",
                "bername": "verlauf_doc",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-01",
                "diag": "Patient desorientiert, Vigilanz reduziert.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "PatientID": "p1",
                "bername": "austritt_doc",
                "bertyp": "Austrittsbericht",
                "berdat": "2024-03-10",
                "diag": "Entlassung in gutem AZ.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    ).to_csv(berichte_v1, index=False, sep=";")

    from src.analysis.validation_cohort_reports import load_raw_included_report_spine

    spine_v1 = load_raw_included_report_spine(berichte_v1, patient_ids=["p1"])
    target_sid = spine_v1.iloc[0][SOURCE_REPORT_ROW_ID_COL]
    target_pber = spine_v1.iloc[0][PIPELINE_BERICHT_COL]

    preds = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": target_pber,
                SOURCE_REPORT_ROW_ID_COL: target_sid,
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-01",
                "klasse": 1,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "direct",
                "evidence_snippets": json.dumps(
                    [
                        {
                            "keyword": "desorientiert",
                            "text": "Patient desorientiert",
                            "evidence_type": "indirect_symptom",
                        }
                    ]
                ),
                "signalstaerke": "hoch",
                "delir_probability_estimate": 70,
                "manual_review_candidate": "False",
                "decision_rule_applied": "direct",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 40,
                "llm_report_text_length": 20,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        ]
    )

    berichte_v2 = tmp_path / "Berichte_v2.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "p1",
                "bername": "austritt_doc",
                "bertyp": "Austrittsbericht",
                "berdat": "2024-03-10",
                "diag": "Entlassung in gutem AZ.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "PatientID": "p1",
                "bername": "verlauf_doc",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-01",
                "diag": "Patient desorientiert, Vigilanz reduziert.",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    ).to_csv(berichte_v2, index=False, sep=";")

    merged, _ = build_complete_validation_reports_frame(
        preds, ["p1"], berichte_path=berichte_v2
    )
    austritt = merged[merged["bertyp"] == "Austrittsbericht"].iloc[0]
    assert austritt[SOURCE_REPORT_ROW_ID_COL] == target_sid
    assert int(austritt["klasse"]) == 1
    assert "desorientiert" in str(austritt["evidence_snippets"]).lower()

    cohort = tmp_path / "cohort.csv"
    labels = tmp_path / "labels.csv"
    preds_path = tmp_path / "preds.csv"
    cohort_df = merged.assign(
        validation_patient_id="Patient_0001",
        validation_report_id=["Patient_0001_Report_0001", "Patient_0001_Report_0002"],
        model_report_prediction=merged["klasse"].astype(int),
    )
    cohort_df.to_csv(cohort, index=False)
    pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0001_Report_0002"],
            "manual_report_ground_truth": [0, 0],
        }
    ).to_csv(labels, index=False)
    preds.to_csv(preds_path, index=False)

    c, lab, pr, spine, raw = load_trace_inputs(cohort, labels, preds_path, berichte_v2)
    rid = cohort_df.loc[cohort_df["bertyp"] == "Austrittsbericht", "validation_report_id"].iloc[0]
    trace = build_report_trace(rid, c, lab, pr, spine, raw)

    assert trace.verdict == "MATCH_FAIL"
    assert any("bertyp" in i.lower() or "evidence_not_in_raw" in i for i in trace.issues)
    assert trace.cohort_row["bertyp"] == "Austrittsbericht"
    assert trace.prediction_row["bertyp"] == "Verlaufseintrag"


def test_diagnose_root_cause_report_written(tmp_path):
    from src.analysis.diagnose_matching_root_cause import run_matching_root_cause_diagnosis
    from src.analysis.validation_cohort_reports import load_raw_included_report_spine

    berichte = tmp_path / "Berichte.csv"
    pd.DataFrame(
        [
            {
                "PatientID": "p1",
                "bername": "v1",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "diag": "ok",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    ).to_csv(berichte, index=False, sep=";")
    spine = load_raw_included_report_spine(berichte, patient_ids=["p1"])
    sid = spine.iloc[0][SOURCE_REPORT_ROW_ID_COL]
    pber = spine.iloc[0][PIPELINE_BERICHT_COL]

    cohort = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001"],
            "validation_report_id": ["Patient_0001_Report_0001"],
            "PatientenID": ["p1"],
            SOURCE_REPORT_ROW_ID_COL: [sid],
            "bertyp": ["Verlaufseintrag"],
            "berdat": ["2024-01-01"],
            "bericht": [pber],
            "pipeline_bericht": [pber],
            "status": ["processed"],
            "model_report_prediction": [1],
            "evidence_snippets": ['[{"keyword":"desorientiert","text":"desorientiert"}]'],
        }
    ).to_csv(cohort, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {"validation_report_id": ["Patient_0001_Report_0001"], "manual_report_ground_truth": [0]}
    ).to_csv(labels, index=False)
    preds = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "bericht": [pber],
            SOURCE_REPORT_ROW_ID_COL: [sid],
            "bertyp": ["Verlaufseintrag"],
            "berdat": ["2024-01-01"],
            "klasse": [1],
            "status": ["processed"],
            "evidence_snippets": ['[{"keyword":"desorientiert","text":"desorientiert"}]'],
        }
    ).to_csv(preds, index=False)

    out = tmp_path / "matching_root_cause_report.txt"
    report = run_matching_root_cause_diagnosis(
        cohort_path=cohort,
        labels_path=labels,
        predictions_path=preds,
        berichte_path=berichte,
        output_path=out,
    )
    assert out.exists()
    assert "ROOT CAUSE" in report
