"""
Build hemorrhage-style pipeline traces for the delirium presentation demo.

A trace holds: report → rule extraction → Agent 1 → Agent 2 → guardrails → klasse,
including prompts and (when captured with --live) raw LLM JSON responses.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.agents.classification import classify_delirium
from src.agents.clinical_guardrails import apply_clinical_decision_guardrails
from src.agents.extraction import EXPECTED_KEYS, empty_result, load_prompt as load_agent1_prompt
from src.agents.extraction import normalize_extraction_result
from src.models.json_parsing import parse_llm_json_output
from src.models.llm_interface import call_llm
from src.pipeline.prompt_selector import load_interpretation_prompt
from src.preprocessing.evidence_extraction import (
    apply_short_report_fulltext_to_evidence,
    extract_delirium_evidence,
    llm_should_receive_evidence,
    should_send_short_report_without_evidence,
)

TRACE_VERSION = 2

TEXT_BLOCK_CHARS = 1400
SYSTEM_PROMPT_EXCERPT_CHARS = 1100


def parse_delir_signale(raw: object) -> Dict[str, List[str]]:
    empty = {
        "desorientierung": [],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return empty
    if isinstance(raw, dict):
        out = dict(empty)
        for key in empty:
            val = raw.get(key, [])
            out[key] = [str(v) for v in val] if isinstance(val, list) else []
        return out
    text = str(raw).strip()
    if not text:
        return empty
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parse_delir_signale(parsed)
    except json.JSONDecodeError:
        pass
    return empty


def _agent1_messages(llm_text: str) -> Tuple[str, str, List[Dict[str, str]]]:
    system_prompt = load_agent1_prompt()
    user_prompt = f"""Evidenz-Bündel (regelbasiert aus dem Bericht; ggf. gekürzter Kurzbericht-Volltext ohne Snippet-Treffer):
{llm_text}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return system_prompt, user_prompt, messages


def _agent2_messages(llm_text: str, signals: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
    system_prompt = load_interpretation_prompt()
    signals_json = json.dumps(signals, ensure_ascii=False, indent=2)
    user_prompt = f"""Der folgende Block ist ein Evidenz-Bündel für die Delir-Beurteilung (regelbasiert extrahierte Snippets oder bei kurzen Berichten ohne Treffer der gekürzte Volltext).

Evidenz / Text:
{llm_text}

Extrahierte Signale (JSON) von Agent 1 zum gleichen Bündel:
{signals_json}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return system_prompt, user_prompt, messages


def _stage_skipped(reason: str) -> Dict[str, Any]:
    return {"ran": False, "skip_reason": reason}


def _agent1_parsed_from_row(row: pd.Series) -> Dict[str, Any]:
    parsed = parse_delir_signale(row.get("delir_signale"))
    if any(parsed.values()):
        return parsed
    raw = str(row.get("delir_signale") or "").strip()
    if not raw:
        return empty_result()
    terms = [t.strip() for t in raw.replace("|", " ").split() if t.strip()]
    out = empty_result()
    out["delir_explizit"] = terms[:10]
    return out


def _agent2_parsed_from_row(row: pd.Series) -> Dict[str, Any]:
    begr = row.get("begruendung", "")
    if isinstance(begr, list):
        begr_list = [str(x) for x in begr]
    else:
        begr_list = [s.strip() for s in str(begr or "").split("|") if s.strip()]
    kw = row.get("alternative_erklaerung_keywords", "")
    if isinstance(kw, list):
        kw_list = kw
    else:
        kw_list = [s.strip() for s in str(kw or "").split("|") if s.strip()]
    return {
        "signalstaerke": str(row.get("signalstaerke") or "niedrig"),
        "kontext": str(row.get("kontext") or ""),
        "alternative_erklaerung": str(row.get("alternative_erklaerung", "")).strip().lower() in ("1", "true", "yes"),
        "alternative_erklaerung_keywords": kw_list,
        "begruendung": begr_list,
    }


def _synthetic_raw_from_parsed(parsed: Dict[str, Any]) -> str:
    """Reconstructed JSON when raw response was not stored (replay from CSV)."""
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _run_agent1(llm_text: str, *, live: bool, replay_row: Optional[pd.Series]) -> Dict[str, Any]:
    system_prompt, user_prompt, messages = _agent1_messages(llm_text)
    if live:
        raw_output = call_llm(messages)
        parsed = normalize_extraction_result(parse_llm_json_output(raw_output, "demo_agent1"))
        return {
            "ran": True,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_output,
            "parsed": parsed,
        }
    parsed = _agent1_parsed_from_row(replay_row) if replay_row is not None else empty_result()
    return {
        "ran": True,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": _synthetic_raw_from_parsed(parsed),
        "parsed": parsed,
        "replay_note": "Reconstructed from validation CSV — use --live for real LLM JSON.",
    }


def _run_agent2(
    llm_text: str,
    signals: Dict[str, Any],
    *,
    live: bool,
    replay_row: Optional[pd.Series],
) -> Dict[str, Any]:
    system_prompt, user_prompt, messages = _agent2_messages(llm_text, signals)
    if live:
        raw_output = call_llm(messages)
        parsed = parse_llm_json_output(raw_output, "demo_agent2")
        if parsed.get("signalstaerke") not in ("hoch", "mittel", "niedrig"):
            parsed["signalstaerke"] = "niedrig"
        return {
            "ran": True,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_output,
            "parsed": parsed,
        }
    parsed = _agent2_parsed_from_row(replay_row) if replay_row is not None else {}
    return {
        "ran": True,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": _synthetic_raw_from_parsed(parsed),
        "parsed": parsed,
        "replay_note": "Reconstructed from validation CSV — use --live for real LLM JSON.",
    }


def _resolve_llm_path(
    full_report_text: str,
    bertyp: str,
) -> Tuple[Dict[str, Any], str, bool]:
    """
    Returns (evidence_dict, llm_text, llm_skipped_by_prefilter).
    Mirrors run_pipeline prefilter / short-report fallback.
    """
    ev = extract_delirium_evidence(full_report_text)
    snippets = ev.get("evidence_snippets") or []
    if llm_should_receive_evidence(snippets):
        return ev, str(ev.get("llm_report_text") or ""), False
    if should_send_short_report_without_evidence(
        full_report_text,
        bertyp,
        snippets,
        original_length=int(ev.get("original_report_text_length") or 0),
    ):
        ev = apply_short_report_fulltext_to_evidence(ev, full_report_text)
        return ev, str(ev.get("llm_report_text") or ""), False
    return ev, "", True


def build_delirium_trace(
    *,
    report_text: str,
    bertyp: str = "",
    manual_gt: Optional[int] = None,
    live: bool = False,
    replay_row: Optional[pd.Series] = None,
    source: str = "validation_cohort",
    polarity: Optional[str] = None,
    case_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a self-contained hemorrhage-style trace dict."""
    ev, llm_text, prefilter_skip = _resolve_llm_path(report_text, bertyp)

    if prefilter_skip:
        guard = apply_clinical_decision_guardrails(
            {"signalstaerke": "niedrig", "kontext": "", "alternative_erklaerung": False, "begruendung": []},
            {},
            ev,
            llm_skipped=True,
        )
        klasse = int(guard["klasse"])
        mode = "live" if live else "replay_csv"
        pol = polarity or ("negative" if klasse == 0 else "positive")
        return {
            "version": TRACE_VERSION,
            "mode": mode,
            "source": source,
            "polarity": pol,
            "case": dict(case_meta or {}),
            "report_text": report_text,
            "extraction": {
                "original_report_text_length": ev.get("original_report_text_length", len(report_text)),
                "llm_report_text": "",
                "llm_report_text_length": 0,
                "llm_text_reduction_method": ev.get("llm_text_reduction_method", ""),
                "evidence_snippets": ev.get("evidence_snippets") or [],
                "has_direct_delir_evidence": bool(ev.get("has_direct_delir_evidence")),
                "has_indirect_delir_evidence": bool(ev.get("has_indirect_delir_evidence")),
            },
            "llm_input_text": "",
            "agent1": _stage_skipped(str(guard.get("decision_rule_applied") or "no_evidence_prefilter_skip")),
            "agent2": _stage_skipped(str(guard.get("decision_rule_applied") or "no_evidence_prefilter_skip")),
            "guardrails": {
                "signalstaerke": guard["signalstaerke"],
                "klasse": klasse,
                "kontext": guard.get("kontext", ""),
                "begruendung": guard.get("begruendung", []),
                "decision_rule_applied": guard.get("decision_rule_applied", ""),
                "manual_review_candidate": bool(guard.get("manual_review_candidate")),
                "klassifikation": guard.get("klassifikation", ""),
            },
            "final": {
                "klasse": klasse,
                "signalstaerke": guard["signalstaerke"],
                "decision_rule_applied": guard.get("decision_rule_applied", ""),
                "llm_called": False,
                "llm_skipped_by_prefilter": True,
                "manual_review_candidate": bool(guard.get("manual_review_candidate")),
                "status": "skipped",
            },
            "verification": {
                "manual_report_ground_truth": manual_gt,
                "model_correct_vs_manual": manual_gt is not None and klasse == manual_gt,
            },
        }

    agent1 = _run_agent1(llm_text, live=live, replay_row=replay_row)
    signals = agent1.get("parsed") or empty_result()
    agent2 = _run_agent2(llm_text, signals, live=live, replay_row=replay_row)
    interpretation = dict(agent2.get("parsed") or {})
    classification = classify_delirium(interpretation)
    guard = apply_clinical_decision_guardrails(interpretation, signals, ev, llm_skipped=False)
    klasse = int(guard["klasse"])
    mode = "live" if live else "replay_csv"
    pol = polarity or ("positive" if klasse == 1 else "negative")

    return {
        "version": TRACE_VERSION,
        "mode": mode,
        "source": source,
        "polarity": pol,
        "case": dict(case_meta or {}),
        "report_text": report_text,
        "extraction": {
            "original_report_text_length": ev.get("original_report_text_length", len(report_text)),
            "llm_report_text": llm_text,
            "llm_report_text_length": len(llm_text),
            "llm_text_reduction_method": ev.get("llm_text_reduction_method", ""),
            "evidence_snippets": ev.get("evidence_snippets") or [],
            "has_direct_delir_evidence": bool(ev.get("has_direct_delir_evidence")),
            "has_indirect_delir_evidence": bool(ev.get("has_indirect_delir_evidence")),
        },
        "llm_input_text": llm_text,
        "agent1": agent1,
        "agent2": agent2,
        "guardrails": {
            "signalstaerke": guard["signalstaerke"],
            "klasse": klasse,
            "kontext": guard.get("kontext", ""),
            "begruendung": guard.get("begruendung", []),
            "decision_rule_applied": guard.get("decision_rule_applied", ""),
            "manual_review_candidate": bool(guard.get("manual_review_candidate")),
            "klassifikation": guard.get("klassifikation", ""),
        },
        "final": {
            "klasse": klasse,
            "signalstaerke": guard["signalstaerke"],
            "decision_rule_applied": guard.get("decision_rule_applied", ""),
            "llm_called": True,
            "llm_skipped_by_prefilter": False,
            "manual_review_candidate": bool(guard.get("manual_review_candidate")),
            "status": "success",
        },
        "verification": {
            "manual_report_ground_truth": manual_gt,
            "model_correct_vs_manual": manual_gt is not None and klasse == manual_gt,
        },
    }


def trace_is_v2(snapshot: Dict[str, Any]) -> bool:
    return int(snapshot.get("version") or 0) >= 2 and "agent1" in snapshot
