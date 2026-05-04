from src.models.llm_interface import call_llm
from src.models.json_parsing import parse_llm_json_output
from src.models.llm_debug import write_llm_debug


EXPECTED_KEYS = [
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
]


def load_prompt():
    with open("prompts/agent_extraction.txt", "r", encoding="utf-8") as f:
        return f.read()


def empty_result():
    return {
        "desorientierung": [],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }


def extract_passages(text: str, patient_id: str = "", report_name: str = ""):
    system_prompt = load_prompt()
    user_prompt = f"""Bericht:
{text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = ""

    try:
        raw_output = call_llm(messages)
        result = parse_llm_json_output(raw_output, "Agent 1 / Extraction")

        for key in EXPECTED_KEYS:
            if key not in result or not isinstance(result[key], list):
                result[key] = []

        return result

    except Exception as exc:
        debug_path = write_llm_debug(
            agent_name="Agent_1_Extraction",
            patient_id=patient_id,
            report_name=report_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            error_message=str(exc),
        )
        print(f"Fehler beim JSON-Parsing in Agent 1: {exc}")
        print(f"LLM-Debug gespeichert in: {debug_path}")
        return empty_result()