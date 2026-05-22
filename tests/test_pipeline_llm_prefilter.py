"""Tests for rule-based LLM skip prefilter before Agent 1 / interpretation LLM."""

import json

import pandas as pd

import src.agents.extraction as extraction
import src.agents.interpretation_llm as interpretation_llm
from src.pipeline import run_pipeline
from src.preprocessing.delirium_hint_keywords import haystack_contains_delirium_hint


def _recording_call_llm(calls):
    def _fake_call_llm(messages):
        calls.append(messages)
        return "{}"

    return _fake_call_llm


def test_prefilter_skips_llm_when_no_hints(monkeypatch, tmp_path):
    calls = []
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)

    fake = _recording_call_llm(calls)
    monkeypatch.setattr(extraction, "call_llm", fake)
    monkeypatch.setattr(interpretation_llm, "call_llm", fake)
    monkeypatch.setattr(
        run_pipeline,
        "_get_report_records",
        lambda: [
            {
                "PatientenID": "p_no_hint",
                "bericht": "x.txt",
                "report_text": "Routinekontrolle. Labor unauffällig. Mobilisation wie geplant.",
            }
        ],
    )

    run_pipeline.main()

    assert len(calls) == 0
    df = pd.read_csv(pred_dir / "agent1_agent2_agent3_results_prompt.csv")
    assert df["llm_skipped_by_prefilter"].astype(str).str.lower().iloc[0] in ("true", "1")
    assert int(df["klasse"].iloc[0]) == 0
    assert df["klassifikation"].iloc[0] == "kein_delir"
    assert int(df["anzahl_treffer"].iloc[0]) == 0
    assert str(df["delir_signale"].iloc[0]) in ("nan", "")
    assert run_pipeline.NO_EVIDENCE_BE in str(df["begruendung"].iloc[0])
    assert run_pipeline.NO_EVIDENCE_KONTEXT in str(df["kontext"].iloc[0])
    assert json.loads(str(df["evidence_snippets"].iloc[0])) == []


def test_prefilter_calls_llm_when_delirium_present(monkeypatch, tmp_path):
    calls = []
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)

    def fake(messages):
        calls.append(messages)
        user = str(messages[-1].get("content", "")) if messages else ""
        if "Extrahierte Signale (JSON):" in user:
            return json.dumps(
                {
                    "signalstaerke": "mittel",
                    "kontext": "stub",
                    "alternative_erklaerung": False,
                    "alternative_erklaerung_keywords": [],
                    "begruendung": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "desorientierung": [],
                "delir_explizit": ["Delirium"],
                "hyperaktivitaet_agitation": [],
                "vigilanz": [],
                "delir_therapie": [],
                "delir_prophylaxe": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(extraction, "call_llm", fake)
    monkeypatch.setattr(interpretation_llm, "call_llm", fake)
    monkeypatch.setattr(
        run_pipeline,
        "_get_report_records",
        lambda: [
            {
                "PatientenID": "p_hit",
                "bericht": "y.txt",
                "report_text": "Verdacht auf Delirium bei Desorientierung.",
            }
        ],
    )

    run_pipeline.main()

    assert len(calls) >= 2
    extr_messages = calls[0]
    user_ex = str(extr_messages[-1].get("content", "")) if extr_messages else ""
    assert "Evidenz-Bündel" in user_ex
    df = pd.read_csv(pred_dir / "agent1_agent2_agent3_results_prompt.csv")
    assert str(df["llm_skipped_by_prefilter"].iloc[0]).lower() in ("false", "0")
    assert int(df["klasse"].iloc[0]) in (0, 1)
    raw_snip = str(df["evidence_snippets"].iloc[0])
    data = json.loads(raw_snip)
    assert isinstance(data, list) and len(data) >= 1
    assert "section" in data[0] and "evidence_type" in data[0]


_EXPECTED_COLUMNS = (
    "PatientenID",
    "bericht",
    "bertyp",
    "berdat",
    "source_report_row_id",
    "original_report_text_length",
    "llm_report_text_length",
    "llm_text_reduction_method",
    "delir_keyword_hits_count",
    "has_direct_delir_evidence",
    "has_indirect_delir_evidence",
    "has_negated_delir_evidence",
    "has_prophylaxis_or_risk_only",
    "has_alternative_explanation",
    "manual_review_candidate",
    "decision_rule_applied",
    "status",
    "llm_called",
    "skipped_reason",
    "llm_skipped_by_prefilter",
    "anzahl_treffer",
    "delir_signale",
    "evidence_snippets",
    "signalstaerke",
    "delir_probability_estimate",
    "kontext",
    "alternative_erklaerung",
    "alternative_erklaerung_keywords",
    "begruendung",
    "klasse",
    "klassifikation",
    "klassifikation_begruendung",
)


def test_prefilter_csv_schema_stable(monkeypatch, tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_pipeline, "PREDICTIONS_DIR", pred_dir)
    fak = _recording_call_llm([])
    monkeypatch.setattr(extraction, "call_llm", fak)
    monkeypatch.setattr(interpretation_llm, "call_llm", fak)

    monkeypatch.setattr(
        run_pipeline,
        "_get_report_records",
        lambda: [
            {
                "PatientenID": "p_s",
                "bericht": "z.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "source_report_row_id": "berichte_row_0",
                "report_text": "Unauffälliger Verlauf.",
            }
        ],
    )
    run_pipeline.main()
    df = pd.read_csv(pred_dir / "agent1_agent2_agent3_results_prompt.csv")
    assert tuple(df.columns) == _EXPECTED_COLUMNS
    assert "bertyp" in df.columns
    assert df["bertyp"].iloc[0] == "Verlaufseintrag"


def test_haystack_hint_case_insensitive_delirium():
    assert haystack_contains_delirium_hint("Nach Delirium-Screening unauffällig.")
    assert not haystack_contains_delirium_hint("Nur Routine.")
