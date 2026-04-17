from src.models.llm_interface import call_llm
from src.models.json_parsing import parse_llm_json_output

EXPECTED_KEY = "desorientierung"

def load_prompt():
    with open("prompts/agent_extraction.txt", "r", encoding="utf-8") as f:
        return f.read()

def empty_result():
    return {EXPECTED_KEY: []}

def extract_passages(text: str):
    system_prompt = load_prompt()
    user_prompt = f"""Bericht:
{text}
"""

    raw_output = call_llm(system_prompt, user_prompt)

    try:
        result = parse_llm_json_output(raw_output, "Agent 1 / Extraction")

        if EXPECTED_KEY not in result or not isinstance(result[EXPECTED_KEY], list):
            result = empty_result()

    except Exception as exc:
        print(f"Fehler beim JSON-Parsing in Agent 1: {exc}")
        result = empty_result()

    return result