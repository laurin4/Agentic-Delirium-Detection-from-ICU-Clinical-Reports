"""Tests for V1→stage2→V3 cascade routing and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.final_manual_validation_evaluation import MANUAL_GT_COL
from src.analysis.run_true_cascade_v1_v2_v3 import (
    CASCADE_REVIEWER_CHECKPOINT,
    V1_CHECKPOINT,
    default_output_dir_for_stage2_mode,
    evaluate_cascade_methods,
    export_stage_evaluation,
    run_cascade_inference,
    seed_v1_checkpoint_if_needed,
)
from src.pipeline.cascade_report_inference import STAGE2_MODE_CASCADE_REVIEWER, STAGE2_MODE_V2
from src.pipeline.paths import CASCADE_REVIEWER_RUN_01_DIR, CASCADE_V1_V2_V3_RUN_01_DIR
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


def test_default_output_dirs():
    assert default_output_dir_for_stage2_mode(STAGE2_MODE_V2) == CASCADE_V1_V2_V3_RUN_01_DIR
    assert default_output_dir_for_stage2_mode(STAGE2_MODE_CASCADE_REVIEWER) == CASCADE_REVIEWER_RUN_01_DIR


def test_cascade_routing_v1_neg(tmp_path, monkeypatch):
    def fake_infer(record, version):
        return {"klasse": 0, "signalstaerke": "niedrig", "status": "skipped"}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    rows, stage2_queue, v3_queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["cascade_stage"] == "v1_negative"
    assert rows[0]["cascade_klasse"] == 0
    assert counts["v1_negative_final"] == 1
    assert counts["v1_positive_to_stage2"] == 0
    assert stage2_queue == []
    assert v3_queue == []


def test_cascade_routing_stage2_confirms(tmp_path, monkeypatch):
    def fake_infer(record, version):
        return {"klasse": 1, "signalstaerke": "hoch", "status": "processed", "prompt_version": version}

    def fake_stage2(record, mode):
        return {"klasse": 1, "signalstaerke": "hoch", "status": "processed", "prompt_version": mode}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_stage2",
        fake_stage2,
    )
    rows, stage2_queue, v3_queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
        stage2_mode=STAGE2_MODE_CASCADE_REVIEWER,
    )
    assert rows[0]["cascade_stage"] == "stage2_confirmed"
    assert rows[0]["cascade_klasse"] == 1
    assert counts["stage2_confirmed_final"] == 1
    assert len(stage2_queue) == 1
    assert v3_queue == []
    assert (tmp_path / CASCADE_REVIEWER_CHECKPOINT).exists()


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

    def fake_stage2(record, mode):
        return {
            "klasse": 0,
            "signalstaerke": "niedrig",
            "status": "processed",
            "prompt_version": mode,
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
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_stage2",
        fake_stage2,
    )
    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.adjudicate_cascade_v3",
        fake_v3,
    )
    rows, stage2_queue, v3_queue, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
        stage2_mode=STAGE2_MODE_CASCADE_REVIEWER,
    )
    assert rows[0]["cascade_stage"] == "v3_adjudicated"
    assert rows[0]["cascade_klasse"] == 1
    assert counts["v3_queue"] == 1
    assert counts["v3_calls_made"] == 1
    assert len(v3_queue) == 1
    assert (tmp_path / "v3_outputs.jsonl").exists()


def test_cascade_dry_run_no_llm(tmp_path, monkeypatch):
    def fail_infer(*args, **kwargs):
        raise AssertionError("LLM should not be called in dry-run")

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fail_infer,
    )
    rows, _, _, counts = run_cascade_inference(
        [_record("R1", "Patient_0001")],
        tmp_path,
        dry_run=True,
    )
    assert rows[0]["cascade_stage"] == "v1_negative"
    assert counts["n_reports"] == 1


def test_cascade_resume_skips_v1(tmp_path, monkeypatch):
    calls = {"v1": 0, "stage2": 0}

    def fake_infer(record, version):
        calls["v1"] += 1
        return {
            "klasse": 1,
            "signalstaerke": "mittel",
            "status": "processed",
            "prompt_version": version,
            "evidence_snippets": "[]",
            "delir_signale": "",
            "decision_rule_applied": "test",
            "kontext": "k",
            "begruendung": "b",
        }

    def fake_stage2(record, mode):
        calls["stage2"] += 1
        return {
            "klasse": 0,
            "signalstaerke": "niedrig",
            "status": "processed",
            "prompt_version": mode,
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
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_stage2",
        fake_stage2,
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
    assert calls["stage2"] == 1


def test_max_stage2_limits_calls(tmp_path, monkeypatch):
    calls = {"stage2": 0}

    def fake_infer(record, version):
        return {
            "klasse": 1,
            "signalstaerke": "hoch",
            "status": "processed",
            "prompt_version": version,
            "evidence_snippets": "[]",
            "delir_signale": "",
            "decision_rule_applied": "test",
            "kontext": "k",
            "begruendung": "b",
        }

    def fake_stage2(record, mode):
        calls["stage2"] += 1
        return {"klasse": 1, "signalstaerke": "hoch", "status": "processed", "prompt_version": mode}

    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_with_prompt_version",
        fake_infer,
    )
    monkeypatch.setattr(
        "src.analysis.run_true_cascade_v1_v2_v3.infer_report_stage2",
        fake_stage2,
    )

    records = [_record("R1", "Patient_0001"), _record("R2", "Patient_0002")]
    rows, _, _, counts = run_cascade_inference(records, tmp_path, max_stage2=1)
    assert calls["stage2"] == 1
    assert counts["stage2_calls_made"] == 1
    pending = [r for r in rows if r["cascade_stage"] == "stage2_pending"]
    assert len(pending) == 1


def test_seed_v1_checkpoint(tmp_path):
    seed_dir = tmp_path / "run_01"
    out_dir = tmp_path / "cascade_reviewer_run_01"
    seed_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (seed_dir / "checkpoints").mkdir(parents=True)
    src = seed_dir / V1_CHECKPOINT
    src.write_text('{"validation_report_id":"R1","stage":"v1","klasse":0,"full_row":{"klasse":0}}\n', encoding="utf-8")

    assert seed_v1_checkpoint_if_needed(out_dir, resume=True, v1_seed_dir=seed_dir)
    assert (out_dir / V1_CHECKPOINT).exists()
    assert not seed_v1_checkpoint_if_needed(out_dir, resume=True, v1_seed_dir=seed_dir)


def test_evaluate_cascade_methods_metrics():
    patient_gt = pd.DataFrame(
        [
            {
                VALIDATION_PATIENT_ID_COL: "Patient_0001",
                "is_patient_complete": 1,
                MANUAL_GT_COL: 1,
                "cascade_patient_positive": 1,
                "v1_patient_positive": 1,
                "stage2_patient_positive": 0,
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
                "stage2_patient_positive": 0,
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


def test_export_stage_evaluation(tmp_path):
    patient_gt = pd.DataFrame(
        [
            {
                VALIDATION_PATIENT_ID_COL: "Patient_0001",
                "PatientenID": "P1",
                "is_patient_complete": 1,
                MANUAL_GT_COL: 1,
                "v1_patient_positive": 1,
                "stage2_patient_positive": 1,
                "cascade_patient_positive": 1,
                "baseline_icdsc_ge_4": 0,
                "baseline_icd10": 0,
                "baseline_composite_or": 0,
                "baseline_composite_and": 0,
                "n_reports_total": 1,
                "n_positive_reports_manual": 1,
            },
            {
                VALIDATION_PATIENT_ID_COL: "Patient_0002",
                "PatientenID": "P2",
                "is_patient_complete": 1,
                MANUAL_GT_COL: 0,
                "v1_patient_positive": 0,
                "stage2_patient_positive": 0,
                "cascade_patient_positive": 0,
                "baseline_icdsc_ge_4": 0,
                "baseline_icd10": 0,
                "baseline_composite_or": 0,
                "baseline_composite_and": 0,
                "n_reports_total": 1,
                "n_positive_reports_manual": 0,
            },
        ]
    )
    export_stage_evaluation(patient_gt, "v1_patient_positive", "v1", tmp_path)
    assert (tmp_path / "v1_patient_metrics.csv").exists()
    assert (tmp_path / "v1_confusion_counts.csv").exists()
    assert (tmp_path / "v1_TP.csv").exists()
    assert (tmp_path / "v1_TN.csv").exists()
