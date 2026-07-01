"""Tests for delirium presentation demo snapshots."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.demo_delirium_case import export_case_png, export_demo_html, export_demo_png, render_demo_html, run_demo
from src.analysis.demo_delirium_snapshot import (
    anonymize_snapshot,
    autopick_validation_report_id,
    build_curated_snapshot,
    build_snapshot_from_row,
    generate_snapshot_from_validation,
    save_snapshot,
)


def _validation_predictions() -> pd.DataFrame:
    direct_snippets = [
        {
            "section": "diag",
            "keyword": "hypoaktives delir",
            "evidence_type": "direct_delir",
            "text": "[Diagnosen]\nPatient mit hypoaktives Delir.",
            "priority": 1,
        }
    ]
    return pd.DataFrame(
        [
            {
                "validation_report_id": "Patient_0001_Report_0001",
                "validation_patient_id": "Patient_0001",
                "PatientenID": "p_pos",
                "bertyp": "Austrittsbericht",
                "berdat": "2024-01-02",
                "bericht": "austritt",
                "klasse": 1,
                "signalstaerke": "hoch",
                "delir_probability_estimate": 90,
                "decision_rule_applied": "direct_delir_positive",
                "has_direct_delir_evidence": True,
                "has_indirect_delir_evidence": False,
                "llm_called": True,
                "llm_skipped_by_prefilter": False,
                "manual_review_candidate": False,
                "status": "success",
                "evidence_snippets": json.dumps(direct_snippets, ensure_ascii=False),
                "kontext": "Explizites Delir.",
                "begruendung": "Delir dokumentiert.",
            },
            {
                "validation_report_id": "Patient_0002_Report_0001",
                "validation_patient_id": "Patient_0002",
                "PatientenID": "p_neg",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "bericht": "verlauf",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "has_direct_delir_evidence": False,
                "has_indirect_delir_evidence": False,
                "llm_called": False,
                "llm_skipped_by_prefilter": True,
                "skipped_reason": "no_evidence_prefilter_skip",
                "status": "skipped",
                "evidence_snippets": "[]",
            },
        ]
    )


def _validation_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_report_id": ["Patient_0001_Report_0001", "Patient_0002_Report_0001"],
            "manual_report_ground_truth": [1, 0],
        }
    )


def test_autopick_tp_and_tn():
    preds = _validation_predictions()
    labels = _validation_labels()
    assert autopick_validation_report_id(preds, labels, polarity="positive") == "Patient_0001_Report_0001"
    assert autopick_validation_report_id(preds, labels, polarity="negative") == "Patient_0002_Report_0001"


def test_curated_snapshots_have_pipeline_fields():
    pos = build_curated_snapshot(polarity="positive")
    neg = build_curated_snapshot(polarity="negative")
    assert pos["final"]["klasse"] == 1
    assert neg["final"]["klasse"] == 0
    assert len(pos["extraction"]["evidence_snippets"]) >= 1
    assert neg["final"]["llm_skipped_by_prefilter"] is True
    assert "PatientenID" not in pos["case"]
    assert pos["case"]["presentation_label"] == "Beispiel-Fall A (Delir positiv)"


def test_anonymize_scrubs_ids_from_text():
    snap = {
        "polarity": "positive",
        "case": {
            "PatientenID": "10234567",
            "validation_report_id": "Patient_0042_Report_0003",
            "validation_patient_id": "Patient_0042",
            "bericht": "austritt_x",
            "berdat": "2024-01-01",
            "bertyp": "Austrittsbericht",
            "manual_report_ground_truth": 1,
        },
        "report_text": "Patient 10234567 mit Delir.",
        "extraction": {"evidence_snippets": [], "llm_report_text": ""},
        "final": {"klasse": 1},
    }
    safe = anonymize_snapshot(snap)
    assert "10234567" not in safe["report_text"]
    assert "PatientenID" not in safe["case"]
    assert safe["case"]["presentation_label"].startswith("Beispiel-Fall")


def test_build_snapshot_from_row():
    preds = _validation_predictions()
    row = preds.iloc[0]
    report = "[Diagnosen]\nPatient mit hypoaktives Delir.\n"
    snap = build_snapshot_from_row(row, report_text=report, manual_gt=1)
    assert snap["case"]["manual_report_ground_truth"] == 1
    assert snap["verification"]["model_correct_vs_manual"] is True


def test_generate_snapshot_fallback_curated(tmp_path):
    stub = tmp_path / "stub_preds.csv"
    stub.write_text("validation_report_id\nonly_one\n", encoding="utf-8")
    out = tmp_path / "pos.json"
    snap = generate_snapshot_from_validation(
        polarity="positive",
        out_path=out,
        predictions_path=stub,
        labels_path=tmp_path / "missing.csv",
    )
    assert out.exists()
    assert snap["source"] == "curated_anonymized"


def test_html_export(tmp_path):
    pos = build_curated_snapshot(polarity="positive")
    neg = build_curated_snapshot(polarity="negative")
    pos_path = tmp_path / "pos.json"
    neg_path = tmp_path / "neg.json"
    save_snapshot(pos, pos_path)
    save_snapshot(neg, neg_path)
    html_out = render_demo_html(pos, neg)
    assert "Delirium Detection Pipeline" in html_out
    assert "direct_delir" in html_out
    assert "PatientenID" not in html_out
    assert "Beispiel-Fall A" in html_out
    out = export_demo_html(pos_path, neg_path, output_path=tmp_path / "demo.html")
    assert out.exists()


def test_png_export(tmp_path):
    pytest.importorskip("matplotlib")
    pos = build_curated_snapshot(polarity="positive")
    path = export_case_png(pos, tmp_path / "fall_a.png")
    assert path.exists()
    assert path.stat().st_size > 1000


def test_save_snapshot_serializes_numpy_int64(tmp_path):
    import numpy as np

    from src.analysis.demo_delirium_snapshot import save_snapshot, to_json_safe

    snap = {
        "polarity": "positive",
        "case": {"manual_report_ground_truth": np.int64(1)},
        "final": {"klasse": np.int64(1)},
        "interpretation": {"delir_probability_estimate": np.int64(92)},
        "anonymized_for_presentation": True,
    }
    assert to_json_safe(snap)["case"]["manual_report_ground_truth"] == 1
    path = tmp_path / "snap.json"
    save_snapshot(snap, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["interpretation"]["delir_probability_estimate"] == 92


def test_run_demo_no_pause(capsys, tmp_path):
    pos = build_curated_snapshot(polarity="positive")
    pos_path = tmp_path / "pos.json"
    save_snapshot(pos, pos_path)
    run_demo(positive=True, pause=False, positive_path=pos_path, negative_path=pos_path)
    captured = capsys.readouterr()
    assert "STEP 1" in captured.out
    assert "hypoaktives Delir" in captured.out or "Delir" in captured.out
