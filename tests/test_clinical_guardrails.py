"""Clinical decision guardrails (symptom clusters + isolated weak symptoms)."""

import pytest

from src.agents.classification import classify_delirium
from src.agents.clinical_guardrails import (
    _has_symptom_cluster,
    _is_isolated_indirect_only,
    apply_clinical_decision_guardrails,
)
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


def _interp(signal="mittel", alt=False, kontext="test"):
    return {
        "signalstaerke": signal,
        "kontext": kontext,
        "alternative_erklaerung": alt,
        "begruendung": [],
    }


def _agitation_only_signals():
    return {
        "hyperaktivitaet_agitation": ["agitiert"],
        "delir_explizit": [],
        "desorientierung": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
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


def test_direct_delir_with_alternative_still_klasse_1():
    signals = {
        "delir_explizit": ["Delir"],
        "desorientierung": [],
        "hyperaktivitaet_agitation": ["unruhig"],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("hoch", alt=True),
        signals,
        _ev(has_direct_delir_evidence=True, has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["decision_rule_applied"] == "direct_delir_positive"


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


def test_bei_delir_conditional_only_klasse_0():
    signals = {
        "delir_prophylaxe": ["Bei Delir Massnahmen"],
        "delir_explizit": [],
        "desorientierung": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        signals,
        _ev(has_prophylaxis_or_risk_only=True),
    )
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


def test_isolated_agitation_without_alternative_klasse_0_with_review():
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        _agitation_only_signals(),
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 0
    assert g["manual_review_candidate"] is True
    assert g["decision_rule_applied"] == "isolated_indirect_not_positive"


@pytest.mark.parametrize(
    "kontext",
    [
        "Agitation im Kontext von Suizidalität und Borderline-Persönlichkeitsstörung.",
        "Unruhe nach Sedierung und Intoxikation.",
    ],
)
def test_isolated_agitation_with_psychiatric_alt_downgrades(kontext):
    g = apply_clinical_decision_guardrails(
        _interp("hoch", alt=True, kontext=kontext),
        _agitation_only_signals(),
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 0
    assert g["manual_review_candidate"] is True
    assert g["decision_rule_applied"] in (
        "alternative_explanation_downgrade",
        "isolated_indirect_not_positive",
    )


def test_isolated_gcs_14_klasse_0():
    signals = {
        "vigilanz": ["GCS 14"],
        "delir_explizit": [],
        "desorientierung": [],
        "hyperaktivitaet_agitation": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    assert _is_isolated_indirect_only(signals, _ev(has_indirect_delir_evidence=True))
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 0
    assert g["decision_rule_applied"] == "isolated_indirect_not_positive"


def test_desorientation_plus_vigilance_cluster_klasse_1():
    signals = {
        "desorientierung": ["desorientiert"],
        "vigilanz": ["Vigilanzminderung"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    assert _has_symptom_cluster(signals, _ev(has_indirect_delir_evidence=True))
    g = apply_clinical_decision_guardrails(
        _interp("hoch"),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["decision_rule_applied"] == "symptom_cluster_positive_review_needed"
    assert g["manual_review_candidate"] is True


def test_desorientation_plus_fluctuating_course_cluster_klasse_1():
    signals = {
        "desorientierung": ["desorientiert", "fluktuierender Verlauf"],
        "hyperaktivitaet_agitation": ["wechselhaft"],
        "delir_explizit": [],
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
    assert g["decision_rule_applied"] in (
        "symptom_cluster_positive_review_needed",
        "indirect_symptoms_positive_review_needed",
    )


def test_delir_therapy_with_compatible_symptoms_klasse_1():
    signals = {
        "delir_therapie": ["Haloperidol bei Delir"],
        "desorientierung": ["desorientiert"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("mittel"),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["decision_rule_applied"] == "delir_therapy_with_compatible_symptoms"


def test_cluster_with_alternative_keeps_klasse_1_with_review():
    signals = {
        "desorientierung": ["desorientiert"],
        "vigilanz": ["somnolent"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    g = apply_clinical_decision_guardrails(
        _interp("mittel", alt=True),
        signals,
        _ev(has_indirect_delir_evidence=True),
    )
    assert g["klasse"] == 1
    assert g["decision_rule_applied"] == "symptom_cluster_with_alternative_review_needed"


def test_no_evidence_klasse_0():
    g = apply_clinical_decision_guardrails(_interp(), {}, _ev(), llm_skipped=True)
    assert g["klasse"] == 0
    assert g["decision_rule_applied"] == "no_evidence_prefilter_skip"


def test_classify_medium_preliminary_is_one():
    c = classify_delirium(_interp("mittel"))
    assert c["klasse"] == 1


def test_binary_output_only_zero_or_one():
    for signal in ("niedrig", "mittel", "hoch"):
        g = apply_clinical_decision_guardrails(
            _interp(signal),
            {"desorientierung": ["x"], "delir_explizit": [], "hyperaktivitaet_agitation": [], "vigilanz": [], "delir_therapie": [], "delir_prophylaxe": []},
            _ev(has_indirect_delir_evidence=True),
        )
        assert g["klasse"] in (0, 1)


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


def test_interpretation_prompt_german_and_clusters():
    from pathlib import Path

    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "agent_interpretation.txt").read_text(
        encoding="utf-8"
    )
    assert "Du bist ein klinisches Bewertungssystem" in prompt
    assert "ISOLIERTE SCHWACHE SYMPTOME" in prompt
    assert "Desorientierung + Vigilanzminderung" in prompt
    assert '"signalstaerke"' in prompt
    assert "«bei delir»" in prompt.lower() or "bei delir" in prompt.lower()


def test_extraction_prompt_german_json_schema():
    from pathlib import Path

    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "agent_extraction.txt").read_text(
        encoding="utf-8"
    )
    assert "Du bist ein klinisches Informationsextraktionssystem" in prompt
    assert '"delir_explizit"' in prompt
