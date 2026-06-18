"""V3 cascade adjudicator for V1+/V2- disagreement reports."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.models.json_parsing import parse_llm_json_output
from src.models.llm_debug import write_llm_debug
from src.models.llm_interface import call_llm
from src.pipeline.paths import PROJECT_ROOT

V3_PROMPT_PATH = PROJECT_ROOT / "prompts" / "delirium_cascade_adjudicator_v3.txt"


def load_v3_adjudicator_prompt() -> str:
    if not V3_PROMPT_PATH.exists():
        raise FileNotFoundError(f"V3 adjudicator prompt missing: {V3_PROMPT_PATH}")
    return V3_PROMPT_PATH.read_text(encoding="utf-8")


def empty_v3_result() -> Dict[str, Any]:
    return {
        "klasse": 0,
        "signalstaerke": "niedrig",
        "kontext": "keine verwertbare V3-Adjudikation",
        "begruendung": ["V3-Adjudikation fehlgeschlagen"],
    }


def _stage_summary(stage_output: Dict[str, Any]) -> Dict[str, Any]:
    begr = stage_output.get("begruendung", "")
    if isinstance(begr, list):
        begr_list = [str(x) for x in begr]
    else:
        begr_list = [str(begr)] if str(begr).strip() else []
    return {
        "klasse": int(stage_output.get("klasse") or 0),
        "signalstaerke": str(stage_output.get("signalstaerke") or "niedrig"),
        "kontext": str(stage_output.get("kontext") or ""),
        "begruendung": begr_list,
        "decision_rule_applied": str(stage_output.get("decision_rule_applied") or ""),
        "evidence_snippets": str(stage_output.get("evidence_snippets") or ""),
        "delir_signale": str(stage_output.get("delir_signale") or ""),
    }


def adjudicate_cascade_v3(
    report_text: str,
    *,
    v1_output: Dict[str, Any],
    v2_output: Dict[str, Any],
    patient_id: str = "",
    report_name: str = "",
) -> Dict[str, Any]:
    """
    Adjudicate a V1+/V2- report. Must not receive ground truth or baseline labels.
    """
    system_prompt = load_v3_adjudicator_prompt()
    payload = {
        "original_report_text": report_text,
        "extracted_evidence": {
            "evidence_snippets": v1_output.get("evidence_snippets") or v2_output.get("evidence_snippets"),
            "delir_signale": v1_output.get("delir_signale") or v2_output.get("delir_signale"),
            "has_direct_delir_evidence": v1_output.get("has_direct_delir_evidence"),
            "has_indirect_delir_evidence": v1_output.get("has_indirect_delir_evidence"),
            "has_negated_delir_evidence": v1_output.get("has_negated_delir_evidence"),
        },
        "v1_review": _stage_summary(v1_output),
        "v2_review": _stage_summary(v2_output),
    }
    user_prompt = (
        "Beurteile den folgenden Bericht als V3-Adjudikator. "
        "V1 und V2 haben unterschiedliche Einschätzungen (V1 positiv, V2 negativ).\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = ""
    try:
        raw_output = call_llm(messages)
        result = parse_llm_json_output(raw_output, "V3 / Cascade Adjudicator")
        klasse = int(result.get("klasse") or 0)
        if klasse not in (0, 1):
            klasse = 0
        signal = str(result.get("signalstaerke") or "niedrig")
        if signal not in ("hoch", "mittel", "niedrig"):
            signal = "niedrig"
        kontext = str(result.get("kontext") or "")
        begr = result.get("begruendung", [])
        if not isinstance(begr, list):
            begr = [str(begr)] if str(begr).strip() else []
        return {
            "klasse": klasse,
            "signalstaerke": signal,
            "kontext": kontext,
            "begruendung": [str(x) for x in begr],
        }
    except Exception as exc:
        debug_path = write_llm_debug(
            agent_name="V3_Cascade_Adjudicator",
            patient_id=patient_id,
            report_name=report_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            error_message=str(exc),
        )
        print(f"Fehler beim JSON-Parsing in V3: {exc}")
        print(f"LLM-Debug gespeichert in: {debug_path}")
        return empty_v3_result()
