import json
from typing import Dict, Any

from src.models.llm_interface import call_llm


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


def interpret_signals_llm(report_text: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 2 (LLM-basiert):
    interpretiert die von Agent 1 extrahierten Signale mit einem Prompt.
    Gibt KEINE finale Klasse 0/1/2 zurück, sondern nur:
    - signalstaerke
    - kontext
    - alternative_erklaerung
    - alternative_erklaerung_keywords
    - begruendung
    """
    system_prompt = load_prompt()

    signals_json = json.dumps(signals, ensure_ascii=False, indent=2)

    user_prompt = f"""Klinischer Bericht:
{report_text}

Extrahierte Signale (JSON):
{signals_json}
"""

    raw_output = call_llm(system_prompt, user_prompt)

    print("=== RAW INTERPRETATION LLM START ===")
    print(raw_output)
    print("=== RAW INTERPRETATION LLM END ===")

    try:
        result = json.loads(raw_output)

        if "signalstaerke" not in result or result["signalstaerke"] not in ["hoch", "mittel", "niedrig"]:
            result["signalstaerke"] = "niedrig"

        if "kontext" not in result or not isinstance(result["kontext"], str):
            result["kontext"] = "keine verwertbare LLM-Interpretation"

        if "alternative_erklaerung" not in result or not isinstance(result["alternative_erklaerung"], bool):
            result["alternative_erklaerung"] = False

        if "alternative_erklaerung_keywords" not in result or not isinstance(result["alternative_erklaerung_keywords"], list):
            result["alternative_erklaerung_keywords"] = []

        if "begruendung" not in result or not isinstance(result["begruendung"], list):
            result["begruendung"] = []

        return result

    except Exception:
        print("Fehler beim JSON parsing in interpretation_llm.py")
        return empty_result()