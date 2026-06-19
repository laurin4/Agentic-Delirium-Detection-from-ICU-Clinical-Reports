"""Cascade stage-2 reviewer (confirm/reject V1-positive reports)."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.models.json_parsing import parse_llm_json_output
from src.models.llm_debug import write_llm_debug
from src.models.llm_interface import call_llm
from src.pipeline.paths import PROJECT_ROOT

CASCADE_REVIEWER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "delirium_cascade_reviewer.txt"


def load_cascade_reviewer_prompt() -> str:
    if not CASCADE_REVIEWER_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Cascade reviewer prompt missing: {CASCADE_REVIEWER_PROMPT_PATH}")
    return CASCADE_REVIEWER_PROMPT_PATH.read_text(encoding="utf-8")


def empty_cascade_reviewer_result() -> Dict[str, Any]:
    return {
        "signalstaerke": "niedrig",
        "kontext": "keine verwertbare Cascade-Reviewer-Interpretation",
        "alternative_erklaerung": False,
        "alternative_erklaerung_keywords": [],
        "begruendung": ["Cascade-Reviewer-Interpretation fehlgeschlagen"],
    }


def interpret_cascade_reviewer(
    report_text: str,
    signals: Dict[str, Any],
    patient_id: str = "",
    report_name: str = "",
) -> Dict[str, Any]:
    """
    Agent-2-compatible interpretation for cascade stage 2 (reviewer mode).

    niedrig → reject V1 positive; mittel/hoch → confirm V1 positive.
    Downstream guardrails still apply.
    """
    system_prompt = load_cascade_reviewer_prompt()
    signals_json = json.dumps(signals, ensure_ascii=False, indent=2)
    user_prompt = f"""Der folgende Block ist ein Evidenz-Bündel für die Delir-Beurteilung (regelbasierte Snippets oder bei kurzen Berichten ohne Treffer der gekürzte Volltext).

Hinweis: V1 (Screening) hat diesen Bericht bereits als positiv eingestuft. Bestätige oder weise das V1-Ergebnis zurück.

Evidenz / Text:
{report_text}

Extrahierte Signale (JSON) von Agent 1 zum gleichen Bündel:
{signals_json}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = ""
    try:
        raw_output = call_llm(messages)
        result = parse_llm_json_output(raw_output, "Cascade Reviewer / Interpretation")

        if result.get("signalstaerke") not in ["hoch", "mittel", "niedrig"]:
            result["signalstaerke"] = "niedrig"

        if not isinstance(result.get("kontext"), str):
            result["kontext"] = "keine verwertbare Cascade-Reviewer-Interpretation"

        if not isinstance(result.get("alternative_erklaerung"), bool):
            result["alternative_erklaerung"] = False

        if not isinstance(result.get("alternative_erklaerung_keywords"), list):
            result["alternative_erklaerung_keywords"] = []

        if not isinstance(result.get("begruendung"), list):
            result["begruendung"] = []

        return result

    except Exception as exc:
        debug_path = write_llm_debug(
            agent_name="Cascade_Reviewer",
            patient_id=patient_id,
            report_name=report_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            error_message=str(exc),
        )
        print(f"Fehler beim JSON-Parsing im Cascade-Reviewer: {exc}")
        print(f"LLM-Debug gespeichert in: {debug_path}")
        return empty_cascade_reviewer_result()
