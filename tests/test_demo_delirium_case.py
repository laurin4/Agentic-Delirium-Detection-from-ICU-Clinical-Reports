"""Tests for delirium presentation demo snapshots."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.demo_delirium_case import export_demo_html, export_demo_txt, render_demo_html, run_demo
from src.analysis.demo_delirium_walkthrough import render_walkthrough_txt
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
            "keyword": "delir",
            "evidence_type": "direct_delir",
            "text": "[Diagnosen]\nPatient mit hypoaktives Delir.",
            "priority": 1,
        }
    ]
    indirect_snippets = [
        {
            "section": "epikrise",
            "keyword": "agitiert",
            "evidence_type": "indirect_symptom",
            "text": "Patient war agitiert bei Suizidalität.",
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
                "validation_report_id": "Patient_0057_Report_0001",
                "validation_patient_id": "Patient_0057",
                "PatientenID": "p_fn",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-03",
                "bericht": "verlauf_fn",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 22,
                "decision_rule_applied": "isolated_indirect_not_positive",
                "has_direct_delir_evidence": False,
                "has_indirect_delir_evidence": True,
                "llm_called": True,
                "llm_skipped_by_prefilter": False,
                "manual_review_candidate": True,
                "status": "success",
                "evidence_snippets": json.dumps(indirect_snippets, ensure_ascii=False),
                "kontext": "Schwache indirekte Hinweise.",
                "begruendung": "Vigilanz und Desorientierung",
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
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0057_Report_0001",
                "Patient_0002_Report_0001",
            ],
            "manual_report_ground_truth": [1, 1, 0],
        }
    )


def test_autopick_tp_and_fn():
    preds = _validation_predictions()
    labels = _validation_labels()
    assert autopick_validation_report_id(preds, labels, polarity="positive") == "Patient_0001_Report_0001"
    assert autopick_validation_report_id(preds, labels, polarity="false_negative") == "Patient_0057_Report_0001"


def test_curated_snapshots_have_pipeline_fields():
    pos = build_curated_snapshot(polarity="positive")
    fn = build_curated_snapshot(polarity="false_negative")
    assert pos["version"] >= 2
    assert "agent1" in pos and "agent2" in pos
    assert pos["final"]["klasse"] == 1
    assert pos["case"]["manual_report_ground_truth"] == 1
    assert pos["verification"]["model_correct_vs_manual"] is True
    assert fn["final"]["klasse"] == 0
    assert fn["case"]["manual_report_ground_truth"] == 1
    assert fn["verification"]["model_correct_vs_manual"] is False
    assert len(pos["extraction"]["evidence_snippets"]) >= 1
    assert fn["final"]["llm_skipped_by_prefilter"] is False
    assert "PatientenID" not in pos["case"]
    assert pos["case"]["presentation_label"] == "Beispiel-Fall A (Delir positiv · TP)"
    assert fn["case"]["presentation_label"] == "Beispiel-Fall B (Falsch negativ · FN)"


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
    fn = build_curated_snapshot(polarity="false_negative")
    pos_path = tmp_path / "pos.json"
    neg_path = tmp_path / "neg.json"
    save_snapshot(pos, pos_path)
    save_snapshot(fn, neg_path)
    html_out = render_demo_html(pos, fn)
    assert "Delirium Detection Pipeline" in html_out
    assert "direct_delir" in html_out
    assert "PatientenID" not in html_out
    assert "Beispiel-Fall A" in html_out
    out = export_demo_html(pos_path, neg_path, output_path=tmp_path / "demo.html")
    assert out.exists()


def test_walkthrough_txt_hemorrhage_structure():
    pos = build_curated_snapshot(polarity="positive")
    txt = render_walkthrough_txt(pos)
    assert "STEP 1" in txt and "STEP 2" in txt and "Final structured output" in txt
    assert "STEP 3" in txt and "Agent 1" in txt
    assert "STEP 6" in txt and "Agent 2" in txt
    assert "Beispiel-Fall A" in txt
    assert "PatientenID" not in txt


def test_export_demo_txt(tmp_path):
    pos = build_curated_snapshot(polarity="positive")
    fn = build_curated_snapshot(polarity="false_negative")
    pos_path = tmp_path / "pos.json"
    neg_path = tmp_path / "neg.json"
    save_snapshot(pos, pos_path)
    save_snapshot(fn, neg_path)
    paths = export_demo_txt(pos_path, neg_path, output_dir=tmp_path)
    assert len(paths) == 3
    assert paths[0].read_text(encoding="utf-8").startswith("=" * 76)


def test_png_export(tmp_path):
    pytest.importorskip("matplotlib")
    from src.analysis.demo_delirium_case import export_case_png

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


def test_thesis_summary_structure():
    from src.analysis.demo_delirium_thesis_summary import (
        CASE_A_HEADING,
        CASE_B_HEADING,
        clinical_report_excerpt,
        final_decision_rows,
        llm_interpretation_bullets,
        render_thesis_case_summary_markdown,
    )

    pos = build_curated_snapshot(polarity="positive")
    fn = build_curated_snapshot(polarity="false_negative")
    pos_md = render_thesis_case_summary_markdown(pos)
    fn_md = render_thesis_case_summary_markdown(fn)

    assert CASE_A_HEADING in pos_md
    assert CASE_B_HEADING in fn_md
    assert "### 1. Klinischer Berichtsauszug" in pos_md
    assert "### 5. Finale Entscheidung" in fn_md
    assert "Korrekt" in pos_md
    assert "Inkorrekt" in fn_md
    assert "system_prompt" not in pos_md.lower()
    assert "Instruction:" not in pos_md

    excerpts = clinical_report_excerpt(pos)
    assert 2 <= len(excerpts) <= 4
    assert "Delir" in " ".join(excerpts)

    assert len(llm_interpretation_bullets(pos)) <= 4
    rows = dict(final_decision_rows(fn))
    assert rows["Modellvorhersage"] == "Kein Delir"
    assert rows["Manuelle Referenz"] == "Delir"
    assert rows["Bewertung"] == "Inkorrekt"


def test_export_demo_thesis(tmp_path):
    from src.analysis.demo_delirium_case import export_demo_thesis

    pos = build_curated_snapshot(polarity="positive")
    fn = build_curated_snapshot(polarity="false_negative")
    pos_path = tmp_path / "pos.json"
    fn_path = tmp_path / "fn.json"
    save_snapshot(pos, pos_path)
    save_snapshot(fn, fn_path)
    paths = export_demo_thesis(pos_path, fn_path, output_dir=tmp_path / "out")
    assert len(paths) == 4
    combined = paths[2].read_text(encoding="utf-8")
    assert "Case A" in combined and "Case B" in combined


def test_fn_pick_uses_model_report_prediction_over_stale_klasse():
    preds = pd.DataFrame(
        [
            {
                "validation_report_id": "Patient_0057_Report_0001",
                "validation_patient_id": "Patient_0057",
                "klasse": 1,
                "model_report_prediction": 0,
                "decision_rule_applied": "isolated_indirect_not_positive",
                "has_indirect_delir_evidence": True,
                "llm_called": True,
                "evidence_snippets": "[]",
            }
        ]
    )
    labels = pd.DataFrame(
        {
            "validation_report_id": ["Patient_0057_Report_0001"],
            "manual_report_ground_truth": [1],
        }
    )
    assert (
        autopick_validation_report_id(preds, labels, polarity="false_negative")
        == "Patient_0057_Report_0001"
    )


def test_patient_level_fn_pick_without_report_level_manual_on_same_row(tmp_path, monkeypatch):
    from src.analysis.demo_delirium_snapshot import _load_frozen_patient_manual_gt

    preds = pd.DataFrame(
        [
            {
                "validation_report_id": "Patient_0075_Report_0001",
                "validation_patient_id": "Patient_0075",
                "model_report_prediction": 0,
                "decision_rule_applied": "isolated_indirect_not_positive",
                "has_indirect_delir_evidence": True,
                "llm_called": True,
                "evidence_snippets": "[]",
            },
            {
                "validation_report_id": "Patient_0075_Report_0002",
                "validation_patient_id": "Patient_0075",
                "model_report_prediction": 0,
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "llm_skipped_by_prefilter": True,
                "evidence_snippets": "[]",
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "validation_report_id": [
                "Patient_0075_Report_0001",
                "Patient_0075_Report_0002",
            ],
            "manual_report_ground_truth": [0, 0],
        }
    )
    monkeypatch.setattr(
        "src.analysis.demo_delirium_snapshot._load_frozen_patient_manual_gt",
        lambda: {"Patient_0075": 1},
    )
    picked = autopick_validation_report_id(
        preds, labels, polarity="false_negative", preferred_fn_patient_suffix="0075"
    )
    assert picked == "Patient_0075_Report_0001"


def test_patient_suffix_matches_and_fn_diagnose():
    from src.analysis.demo_delirium_snapshot import (
        diagnose_preferred_fn_patients,
        patient_suffix_matches,
    )

    row_fp = pd.Series(
        {
            "validation_patient_id": "Patient_0057",
            "validation_report_id": "Patient_0057_Report_0002",
            "klasse": 1,
            "manual_report_ground_truth": 0,
        }
    )
    row_fn = pd.Series(
        {
            "validation_patient_id": "Patient_0057",
            "validation_report_id": "Patient_0057_Report_0003",
            "klasse": 0,
            "manual_report_ground_truth": 1,
        }
    )
    assert patient_suffix_matches(row_fp, "0057")
    assert patient_suffix_matches(row_fp, "57")
    preds = pd.DataFrame([row_fp.to_dict(), row_fn.to_dict()])
    labels = pd.DataFrame(
        {
            "validation_report_id": ["Patient_0057_Report_0002", "Patient_0057_Report_0003"],
            "manual_report_ground_truth": [0, 1],
        }
    )
    diag = diagnose_preferred_fn_patients(preds, labels)
    block = next(d for d in diag if d["patient_suffix"] == "0057")
    assert block["pickable_fn_report_id"] == "Patient_0057_Report_0003"
    assert block["report_level_fn_reports"] == ["Patient_0057_Report_0003"]


def test_run_demo_no_pause(capsys, tmp_path):
    from src.analysis.demo_delirium_case import run_demo

    pos = build_curated_snapshot(polarity="positive")
    pos_path = tmp_path / "pos.json"
    save_snapshot(pos, pos_path)
    run_demo(positive=True, pause=False, positive_path=pos_path, negative_path=pos_path)
    captured = capsys.readouterr()
    assert "STEP 1" in captured.out
    assert "Agent 1" in captured.out
    assert "hypoaktives Delir" in captured.out or "Delir" in captured.out
