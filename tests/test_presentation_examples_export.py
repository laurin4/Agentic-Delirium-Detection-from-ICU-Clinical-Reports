"""Tests for presentation examples export."""

import json
from pathlib import Path

import pandas as pd

from src.analysis.export_presentation_examples import (
    build_presentation_examples,
    highlight_text,
    main,
    parse_evidence_snippets,
    select_example_indices,
)


def _fake_predictions() -> pd.DataFrame:
    direct_snippets = [
        {
            "section": "diag",
            "keyword": "hypoaktives delir",
            "evidence_type": "direct_delir",
            "text": "[Diagnosen]\nPatient mit hypoaktives Delir.",
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
                "PatientenID": "p1",
                "bericht": "austritt_p1",
                "bertyp": "Austrittsbericht",
                "klasse": 1,
                "signalstaerke": "hoch",
                "delir_probability_estimate": 90,
                "decision_rule_applied": "direct_delir_positive",
                "manual_review_candidate": "False",
                "has_direct_delir_evidence": "True",
                "has_indirect_delir_evidence": "False",
                "has_prophylaxis_or_risk_only": "False",
                "llm_skipped_by_prefilter": "False",
                "llm_text_reduction_method": "structured_evidence_extraction",
                "evidence_snippets": json.dumps(direct_snippets, ensure_ascii=False),
                "kontext": "Explizites Delir dokumentiert.",
                "begruendung": "Delir in Diagnose",
            },
            {
                "PatientenID": "p2",
                "bericht": "verlauf_p2",
                "bertyp": "Verlaufseintrag",
                "klasse": 1,
                "signalstaerke": "mittel",
                "delir_probability_estimate": 55,
                "decision_rule_applied": "indirect_symptoms_positive_review_needed",
                "manual_review_candidate": "True",
                "has_direct_delir_evidence": "False",
                "has_indirect_delir_evidence": "True",
                "has_prophylaxis_or_risk_only": "False",
                "llm_skipped_by_prefilter": "False",
                "llm_text_reduction_method": "structured_evidence_extraction",
                "evidence_snippets": json.dumps(indirect_snippets, ensure_ascii=False),
                "kontext": "Indirekte Symptome.",
                "begruendung": "Agitation",
            },
            {
                "PatientenID": "p3",
                "bericht": "verlauf_p3",
                "bertyp": "Verlaufseintrag",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 5,
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "manual_review_candidate": "False",
                "has_direct_delir_evidence": "False",
                "has_indirect_delir_evidence": "False",
                "has_prophylaxis_or_risk_only": "False",
                "llm_skipped_by_prefilter": "True",
                "llm_text_reduction_method": "no_evidence_prefilter_skip",
                "evidence_snippets": "[]",
                "kontext": "Keine Hinweise.",
                "begruendung": "",
            },
            {
                "PatientenID": "p4",
                "bericht": "verlauf_p4",
                "bertyp": "Verlaufseintrag",
                "klasse": 1,
                "signalstaerke": "mittel",
                "delir_probability_estimate": 40,
                "decision_rule_applied": "llm_classification",
                "manual_review_candidate": "False",
                "has_direct_delir_evidence": "False",
                "has_indirect_delir_evidence": "True",
                "has_prophylaxis_or_risk_only": "False",
                "llm_skipped_by_prefilter": "False",
                "llm_text_reduction_method": "structured_evidence_extraction",
                "evidence_snippets": json.dumps(indirect_snippets, ensure_ascii=False),
                "kontext": "FP case",
                "begruendung": "",
            },
            {
                "PatientenID": "p5",
                "bericht": "prophy_p5",
                "bertyp": "Verlaufseintrag",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 10,
                "decision_rule_applied": "prophylaxis_only_not_positive",
                "manual_review_candidate": "False",
                "has_direct_delir_evidence": "False",
                "has_indirect_delir_evidence": "False",
                "has_prophylaxis_or_risk_only": "True",
                "llm_skipped_by_prefilter": "False",
                "llm_text_reduction_method": "structured_evidence_extraction",
                "evidence_snippets": json.dumps(
                    [
                        {
                            "section": "prozedere",
                            "keyword": "delirprophylaxe",
                            "evidence_type": "prophylaxis_or_risk",
                            "text": "Delirprophylaxe empfohlen.",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "kontext": "Nur Prophylaxe.",
                "begruendung": "",
            },
        ]
    )


def _fake_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": ["p1", "p2", "p3", "p4", "p5"],
            "baseline_icd10": [1, 0, 0, 0, 0],
            "max_icdsc": [5, 2, 0, 0, 0],
            "baseline_icdsc_ge_4": [1, 0, 0, 0, 0],
            "baseline_composite": [1, 0, 0, 0, 0],
        }
    )


def test_highlight_markers():
    snippets = [{"keyword": "delir", "evidence_type": "direct_delir"}]
    out = highlight_text("Patient mit Delir.", snippets)
    assert "==Delir==" in out or "==delir==" in out.lower()


def test_select_works_with_few_rows():
    preds = _fake_predictions().iloc[:2]
    df = preds.copy()
    picks = select_example_indices(df)
    assert len(picks) >= 1


def test_export_with_fake_data(tmp_path):
    lookup = {
        ("p1", "austritt_p1"): "[Diagnosen]\nPatient mit hypoaktives Delir im Verlauf.",
    }
    csv_df, md, report = build_presentation_examples(
        _fake_predictions(),
        baseline=_fake_baseline(),
        comparison=None,
        report_lookup=lookup,
    )
    assert len(csv_df) >= 1
    assert "presentation examples" in md.lower()
    assert "example_01" in report
    assert "==" in csv_df.iloc[0]["original_report_excerpt"] or "**" in csv_df.iloc[0]["original_report_excerpt"]


def test_missing_baseline_does_not_crash():
    csv_df, _, _ = build_presentation_examples(
        _fake_predictions().iloc[:1],
        baseline=None,
        comparison=None,
        report_lookup={},
    )
    assert len(csv_df) == 1


def test_main_writes_files(tmp_path):
    pred = tmp_path / "pred.csv"
    out = tmp_path / "out"
    _fake_predictions().iloc[:2].to_csv(pred, index=False)
    main(
        predictions_path=pred,
        baseline_path=tmp_path / "missing_baseline.csv",
        comparison_path=tmp_path / "missing_comp.csv",
        output_dir=out,
        berichte_path=tmp_path / "missing_berichte.csv",
    )
    assert (out / "presentation_examples.csv").exists()
    assert (out / "presentation_examples.md").exists()
    assert (out / "presentation_examples_report.txt").exists()
    md = (out / "presentation_examples.md").read_text(encoding="utf-8")
    assert "## Example" in md


def test_parse_evidence_snippets_empty():
    assert parse_evidence_snippets("[]") == []
    assert parse_evidence_snippets(None) == []
