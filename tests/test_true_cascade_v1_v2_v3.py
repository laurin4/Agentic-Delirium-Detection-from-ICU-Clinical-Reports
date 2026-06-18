"""Tests for V1→V2→V3 cascade routing and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.final_manual_validation_evaluation import MANUAL_GT_COL
from src.analysis.run_true_cascade_v1_v2_v3 import (
    evaluate_cascade_methods,
    run_cascade_inference,
)
from src.pipeline.validation_report_identity import (
    VALIDATION_PATIENT_ID_COL,
    VALIDATION_REPORT_ID_COL,
)


def _record(report_id: str, patient_id: str, text: str = "Bericht") -> dict:
    return {
        VALIDATION_REPORT_ID_COL: report_id,
        VALIDATION_PATIENT_ID_COL: patient_id,
        "PatientenID": patient_id.replace("Patient_", "P"),
        "bericht": f"{report_id}.txt",
        "bertyp": "pflege",
        "report_text": text,
    }


def test_cascade_routing_v1_neg(tmp_path, monkeypatch):
    def fake_infer(record, version):
        return {"klasse": 0, "signalstaerke": "niedrig", "status": "skipped"}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    rows, queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["cascade_stage"] == "v1_negative"
    assert rows[0]["cascade_klasse"] == 0
    assert counts["v1_negative_final"] == 1
    assert counts["v1_positive_to_v2"] == 0
    assert queue == []


def test_cascade_routing_v2_confirms(tmp_path, monkeypatch):
    def fake_infer(record, version):
        return {"klasse": 1, "signalstaerke": "hoch", "status": "processed", "prompt_version": version}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    rows, queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
    )
    assert rows[0]["cascade_stage"] == "v2_confirmed"
    assert rows[0]["cascade_klasse"] == 1
    assert counts["v2_confirmed_final"] == 1
    assert queue == []


def test_cascade_routing_v3_adjudicates(tmp_path, monkeypatch):
    def fake_infer(record, version):
        klasse = 1 if version == "v1" else 0
        return {
            "klasse": klasse,
            "signalstaerke": "mittel",
            "status": "processed",
            "prompt_version": version,
            "evidence_snippets": "[]",
            "delir_signale": "",
            "decision_rule_applied": "test",
            "kontext": "k",
            "begruendung": "b",
        }

    def fake_v3(report_text, **kwargs):
        return {"klasse": 1, "signalstaerke": "hoch", "kontext": "restored", "begruendung": ["ok"]}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.adjudicate_cascade_v3",
        fake_v3,
    )
    rows, queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
    )
    assert rows[0]["cascade_stage"] == "v3_adjudicated"
    assert rows[0]["cascade_klasse"] == 1
    assert counts["v3_queue"] == 1
    assert counts["v3_calls_made"] == 1
    assert len(queue) == 1
    assert (tmp_path / "v3_outputs.jsonl").exists()


def test_cascade_dry_run_no_llm(tmp_path, monkeypatch):
    def fail_infer(*args, **kwargs):
        raise AssertionError("LLM should not be called in dry-run")

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fail_infer,
    )
    rows, queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
        dry_run=True,
    )
    assert rows[0]["cascade_stage"] == "v1_negative"
    assert counts["n_reports"] == 1


def test_cascade_resume_skips_v1(tmp_path, monkeypatch):
    calls = {"v1": 0, "v2": 0}

    def fake_infer(record, version):
        calls[version] += 1
        klasse = 1 if version == "v1" else 0
        return {
            "klasse": klasse,
            "signalstaerke": "mittel",
            "status": "processed",
            "prompt_version": version,
            "evidence_snippets": "[]",
            "delir_signale": "",
            "decision_rule_applied": "test",
            "kontext": "k",
            "begruendung": "b",
        }

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.adjudicate_cascade_v3",
        lambda *a, **k: {"klasse": 0, "signalstaerke": "niedrig", "kontext": "", "begruendung": []},
    )

    run_cascade_inference([_record("R1", "Patient_0001")], tmp_path, resume=False)
    assert calls["v1"] == 1
    calls["v1"] = 0
    run_cascade_inference([_record("R1", "Patient_0001")], tmp_path, resume=True)
    assert calls["v1"] == 0
    assert calls["v2"] == 1


def test_evaluate_cascade_methods_metrics():
    patient_gt = pd.DataFrame(
        [
            {
                VALIDATION_PATIENT_ID_COL: "Patient_0001",
                "is_patient_complete": 1,
                MANUAL_GT_COL: 1,
                "cascade_patient_positive": 1,
                "v1_patient_positive": 1,
                "v2_patient_positive": 0,
                "baseline_icdsc_ge_4": 0,
                "baseline_icd10": 0,
                "baseline_composite_or": 0,
                "baseline_composite_and": 0,
            },
            {
                VALIDATION_PATIENT_ID_COL: "Patient_0002",
                "is_patient_complete": 1,
                MANUAL_GT_COL: 0,
                "cascade_patient_positive": 0,
                "v1_patient_positive": 0,
                "v2_patient_positive": 0,
                "baseline_icdsc_ge_4": 0,
                "baseline_icd10": 0,
                "baseline_composite_or": 0,
                "baseline_composite_and": 0,
            },
        ]
    )
    metrics, confusion = evaluate_cascade_methods(patient_gt)
    cascade_row = metrics[metrics["method"] == "cascade"].iloc[0]
    assert int(cascade_row["tp"]) == 1
    assert int(cascade_row["tn"]) == 1
    assert int(cascade_row["fp"]) == 0
    assert int(cascade_row["fn"]) == 0
