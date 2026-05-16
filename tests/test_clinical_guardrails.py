"""Clinical decision guardrails (less aggressive: preserve uncertain positives)."""

import pytest

from src.agents.classification import classify_delirium
from src.agents.clinical_guardrails import apply_clinical_decision_guardrails
from src.agents.extraction import normalize_extraction_result
from src.preprocessing.evidence_extraction import extract_delirium_evidence


def _ev(**kwargs):
    base = {
        "has_direct_delir_evidence": False,
        "has_indirect_delir_evidence": False,
        "has_negated_delir_evidence": False,
        "has_prophylaxis_or_risk_only": False,
        "llm_text_reduction_method": "structured_evidence_extraction",
    }
    base.update(kwargs)
    return base


def _interp(signal="mittel", alt=False):
    return {
        "signalstaerke": signal,
        "kontext": "test",
        "alternative_erklaerung": alt,
        "begruendung": [],
    }


def test_hypoaktives_delir_klasse_1_no_manual_review():
    signals = {
        "delir_explizit": ["hypoaktives Delir"],
        "desorientierung": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(_interp("hoch"), signals, _ev(has_direct_delir_evidence=True))
    assert g["klasse"] == 1
    assert g["decision_rule_applied"] == "direct_delir_positive"
    assert g["manual_review_candidate"] is False


def test_delirprophylaxe_only_klasse_0():
    signals = {
        "delir_prophylaxe": ["Delirprophylaxe"],
        "delir_explizit": [],
        "desorientierung": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
    }
    g = apply_clinical_decision_guardrails(_interp("mittel"), signals, _ev(has_prophylaxis_or_risk_only=True))
    assert g["klasse"] == 0
    assert g["decision_rule_applied"] == "prophylaxis_only_not_positive"


def test_kein_delir_klasse_0():
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        {},
        _ev(has_negated_delir_evidence=True),
    )
    assert g["klasse"] == 0
    assert g["decision_rule_applied"] == "negated_delir_not_positive"


def test_agitation_only_llm_positive_keeps_klasse_1_with_review():
    signals = {
        "hyperaktivitaet_agitation": ["agitiert"],
        "delir_explizit": [],
        "desorientierung": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["manual_review_candidate"] is True
    assert g["decision_rule_applied"] == "indirect_symptoms_positive_review_needed"


def test_alternative_explanation_llm_positive_keeps_klasse_1_with_review():
    signals = {
        "hyperaktivitaet_agitation": ["unruhig"],
        "delir_explizit": [],
        "desorientierung": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("hoch", alt=True),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["manual_review_candidate"] is True
    assert g["decision_rule_applied"] == "positive_with_alternative_explanation_review_needed"


def test_signalstaerke_mittel_positive_manual_review():
    signals = {
        "desorientierung": ["desorientiert"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["manual_review_candidate"] is True


def test_no_evidence_klasse_0():
    g = apply_clinical_decision_guardrails(_interp(), {}, _ev(), llm_skipped=True)
    assert g["klasse"] == 0
    assert g["decision_rule_applied"] == "no_evidence_prefilter_skip"


def test_classify_medium_preliminary_is_one():
    c = classify_delirium(_interp("mittel"))
    assert c["klasse"] == 1


def test_extraction_dedupe_and_cap():
    raw = {
        "delir_explizit": ["Delir", "delir", "Delir"],
        "delir_prophylaxe": [f"p{i}" for i in range(20)],
    }
    out = normalize_extraction_result(raw)
    assert len(out["delir_explizit"]) == 1
    assert len(out["delir_prophylaxe"]) == 10


def test_evidence_snippets_bounded_and_deduped(monkeypatch):
    monkeypatch.setenv("EVIDENCE_MAX_HITS_PROPHYLAXIS", "1")
    text = "[Prozedere]\nDelirprophylaxe empfohlen. Delirprophylaxe weiter. Delirprophylaxe mobilisation.\n"
    ev = extract_delirium_evidence(text)
    prophy = [s for s in ev["evidence_snippets"] if s["evidence_type"] == "prophylaxis_or_risk"]
    assert len(prophy) <= 1
    for s in ev["evidence_snippets"]:
        assert len(s["text"]) <= 400
