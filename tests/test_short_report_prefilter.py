"""Short-report fallback when no evidence snippets (env-gated)."""

import json

import pytest

from src.preprocessing.evidence_extraction import (
    METHOD_NO_EVIDENCE,
    METHOD_SHORT_REPORT_FULLTEXT,
    apply_short_report_fulltext_to_evidence,
    extract_delirium_evidence,
    should_send_short_report_without_evidence,
)


def test_short_report_not_sent_when_env_disabled(monkeypatch):
    monkeypatch.delenv("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM", raising=False)
    text = "[Diagnosen]\nRoutinekontrolle unauffällig.\n"
    ev = extract_delirium_evidence(text)
    assert not should_send_short_report_without_evidence(
        text, "Verlaufseintrag", ev["evidence_snippets"], original_length=len(text)
    )


def test_short_report_sent_when_enabled(monkeypatch):
    monkeypatch.setenv("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM", "true")
    monkeypatch.setenv("SHORT_REPORT_CHAR_THRESHOLD", "1000")
    text = "[Diagnosen]\nKurzer Verlauf, mobilisation wie geplant.\n"
    ev = extract_delirium_evidence(text)
    assert ev["llm_text_reduction_method"] == METHOD_NO_EVIDENCE
    assert should_send_short_report_without_evidence(
        text, "Verlaufseintrag", ev["evidence_snippets"], original_length=len(text)
    )
    updated = apply_short_report_fulltext_to_evidence(ev, text)
    assert updated["llm_text_reduction_method"] == METHOD_SHORT_REPORT_FULLTEXT
    assert updated["llm_report_text_length"] > 0


def test_short_report_not_sent_for_long_report(monkeypatch):
    monkeypatch.setenv("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM", "true")
    monkeypatch.setenv("SHORT_REPORT_CHAR_THRESHOLD", "50")
    text = "[Diagnosen]\n" + ("x" * 200)
    ev = extract_delirium_evidence(text)
    assert not should_send_short_report_without_evidence(
        text, "Verlaufseintrag", ev["evidence_snippets"], original_length=len(text)
    )


def test_short_report_not_sent_for_dokumentationsblatt_type(monkeypatch):
    monkeypatch.setenv("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM", "true")
    text = "[Diagnosen]\nKurz.\n"
    ev = extract_delirium_evidence(text)
    assert not should_send_short_report_without_evidence(
        text, "Dokumentationsblatt", ev["evidence_snippets"], original_length=len(text)
    )


def test_pipeline_short_report_path(monkeypatch, tmp_path):
    import src.agents.extraction as extraction
    import src.agents.interpretation_llm as interpretation_llm
    from src.pipeline import run_pipeline

    monkeypatch.setenv("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM", "true")
    monkeypatch.setenv("SHORT_REPORT_CHAR_THRESHOLD", "2000")

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)

    calls = []

    def fake_call_llm(messages):
        calls.append(messages)
        user = str(messages[-1].get("content", ""))
        if "Extrahierte Signale" in user or "Agent 1" in user:
            return json.dumps(
                {
                    "signalstaerke": "niedrig",
                    "kontext": "kein Delir",
                    "alternative_erklaerung": False,
                    "alternative_erklaerung_keywords": [],
                    "begruendung": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "desorientierung": [],
                "delir_explizit": [],
                "hyperaktivitaet_agitation": [],
                "vigilanz": [],
                "delir_therapie": [],
                "delir_prophylaxe": [],
            }
        )

    monkeypatch.setattr(extraction, "call_llm", fake_call_llm)
    monkeypatch.setattr(interpretation_llm, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        run_pipeline,
        "_get_report_records",
        lambda: [
            {
                "PatientenID": "p_short",
                "bericht": "v1",
                "bertyp": "Verlaufseintrag",
                "report_text": "[Diagnosen]\nRoutinekontrolle, Patient stabil.\n",
            }
        ],
    )

    run_pipeline.main()
    import pandas as pd

    df = pd.read_csv(pred_dir / "agent1_agent2_agent3_results_prompt.csv")
    assert len(calls) >= 2
    assert df.iloc[0]["llm_text_reduction_method"] == METHOD_SHORT_REPORT_FULLTEXT
    assert str(df.iloc[0]["llm_skipped_by_prefilter"]).lower() in ("false", "0")
