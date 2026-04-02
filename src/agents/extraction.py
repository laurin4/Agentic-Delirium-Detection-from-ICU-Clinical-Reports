import json
from src.models.llm_interface import call_llm

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
    print("=== RAW OUTPUT START ===")
    print(raw_output)
    print("=== RAW OUTPUT END ===")
    

    try:
        result = json.loads(raw_output)

        if EXPECTED_KEY not in result or not isinstance(result[EXPECTED_KEY], list):
            result = empty_result()

    except Exception:
        print("Fehler beim JSON parsing")
        print(raw_output)
        result = empty_result()

    return result