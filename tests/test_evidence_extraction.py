"""Structured rule-based delirium evidence extraction (pre-LLM)."""

import json
import os

import pytest

from src.preprocessing.evidence_extraction import (
    METHOD_NO_EVIDENCE,
    METHOD_STRUCTURED,
    extract_delirium_evidence,
    llm_should_receive_evidence,
)


def test_direct_delir_in_diagnosen_prioritized():
    text = "[Diagnosen]\nPatient mit hyperaktives Delir.\n\n[Epikrise]\nStabil."
    ev = extract_delirium_evidence(text)
    assert ev["has_direct_delir_evidence"] is True
    assert ev["llm_text_reduction_method"] == METHOD_STRUCTURED
    assert llm_should_receive_evidence(ev["evidence_snippets"])
    types = [s["evidence_type"] for s in ev["evidence_snippets"]]
    assert "direct_delir" in types
    secs = [s["section"] for s in ev["evidence_snippets"] if s["evidence_type"] == "direct_delir"]
    assert "diag" in secs


def test_negation_not_positive_direct():
    text = "[Diagnosen]\nKein Delir nach Screening.\n"
    ev = extract_delirium_evidence(text)
    assert ev["has_negated_delir_evidence"] is True
    assert not ev["has_direct_delir_evidence"]
    assert not llm_should_receive_evidence(ev["evidence_snippets"])
    assert ev["llm_text_reduction_method"] == METHOD_NO_EVIDENCE


def test_delirprophylaxe_is_prophylaxis_not_direct():
    text = "[Prozedere]\nDelirprophylaxe mit Mobilisation.\n"
    ev = extract_delirium_evidence(text)
    assert ev["has_prophylaxis_or_risk_only"] is True
    assert not ev["has_direct_delir_evidence"]
    assert llm_should_receive_evidence(ev["evidence_snippets"])
    assert any(s["evidence_type"] == "prophylaxis_or_risk" for s in ev["evidence_snippets"])


def test_indirect_symptoms_extracted():
    text = "[Jetziges Leiden]\nPatient stark desorientiert und verwirrt.\n"
    ev = extract_delirium_evidence(text)
    assert ev["has_indirect_delir_evidence"] is True
    ets = {s["evidence_type"] for s in ev["evidence_snippets"]}
    assert "indirect_symptom" in ets


def test_no_evidence_method():
    text = "Routinekontrolle. Labor unauffällig.\n"
    ev = extract_delirium_evidence(text)
    assert ev["evidence_snippets"] == []
    assert ev["llm_text_reduction_method"] == METHOD_NO_EVIDENCE
    assert not llm_should_receive_evidence(ev["evidence_snippets"])


def test_max_snippets_respected(monkeypatch):
    monkeypatch.setenv("EVIDENCE_MAX_SNIPPETS", "2")
    text = "[Diagnosen]\nDelir. Verwirrt. Agitiert. Somnolent.\n"
    ev = extract_delirium_evidence(text)
    assert len(ev["evidence_snippets"]) <= 2


def test_max_llm_chars_respected(monkeypatch):
    monkeypatch.setenv("EVIDENCE_MAX_LLM_CHARS", "400")
    text = "[Diagnosen]\n" + "Delir. " * 40 + "\n[Epikrise]\n" + "Verwirrt. " * 40
    ev = extract_delirium_evidence(text)
    assert ev["llm_text_reduction_method"] == METHOD_STRUCTURED
    assert len(ev["llm_report_text"]) <= 400 + 50


def test_section_labels_preserved_in_llm_text():
    text = "[Diagnosen]\nHyperaktives Delir dokumentiert.\n"
    ev = extract_delirium_evidence(text)
    assert "Diagnosen" in ev["llm_report_text"]
    assert "direct_delir" in ev["llm_report_text"]


def test_evidence_snippets_json_roundtrip():
    ev = extract_delirium_evidence("[Diagnosen]\nDelir.\n")
    raw = json.dumps(ev["evidence_snippets"], ensure_ascii=False)
    data = json.loads(raw)
    assert isinstance(data, list)
    assert data[0]["section"] == "diag"
