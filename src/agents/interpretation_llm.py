import json
from typing import Dict, Any

from src.models.llm_interface import call_llm
from src.models.json_parsing import parse_llm_json_output
from src.models.llm_debug import write_llm_debug


def load_prompt() -> str:
    with open("prompts/agent_interpretation.txt", "r", encoding="utf-8") as f:
        return f.read()


def empty_result() -> Dict[str, Any]:
    return {
        "signalstaerke": "niedrig",
        "kontext": "keine verwertbare LLM-Interpretation",
        "alternative_erklaerung": False,
        "alternative_erklaerung_keywords": [],
        "begruendung": ["LLM-Interpretation fehlgeschlagen"],
    }


def interpret_signals_llm(
    report_text: str,
    signals: Dict[str, Any],
    patient_id: str = "",
    report_name: str = "",
) -> Dict[str, Any]:

    system_prompt = load_prompt()

    signals_json = json.dumps(signals, ensure_ascii=False, indent=2)

    user_prompt = f"""Klinischer Bericht:
{report_text}

Extrahierte Signale (JSON):
{signals_json}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = ""

    try:
        raw_output = call_llm(messages)
        result = parse_llm_json_output(raw_output, "Agent 2 / Interpretation")

        if result.get("signalstaerke") not in ["hoch", "mittel", "niedrig"]:
            result["signalstaerke"] = "niedrig"

        if not isinstance(result.get("kontext"), str):
            result["kontext"] = "keine verwertbare LLM-Interpretation"

        if not isinstance(result.get("alternative_erklaerung"), bool):
            result["alternative_erklaerung"] = False

        if not isinstance(result.get("alternative_erklaerung_keywords"), list):
            result["alternative_erklaerung_keywords"] = []

        if not isinstance(result.get("begruendung"), list):
            result["begruendung"] = []

        return result

    except Exception as exc:
        debug_path = write_llm_debug(
            agent_name="Agent_2_Interpretation",
            patient_id=patient_id,
            report_name=report_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            error_message=str(exc),
        )
        print(f"Fehler beim JSON-Parsing in Agent 2: {exc}")
        print(f"LLM-Debug gespeichert in: {debug_path}")
        return empty_result()