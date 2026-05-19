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

MAX_HITS_PER_CATEGORY = 10


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


def normalize_extraction_result(result: dict) -> dict:
    """De-duplicate and cap Agent 1 hit lists (verbatim terms preserved)."""
    out = empty_result()
    for key in EXPECTED_KEYS:
        raw = result.get(key, [])
        if not isinstance(raw, list):
            raw = []
        seen: set = set()
        deduped = []
        for item in raw:
            s = str(item).strip()
            if not s:
                continue
            key_lower = s.lower()
            if key_lower in seen:
                continue
            seen.add(key_lower)
            deduped.append(s)
            if len(deduped) >= MAX_HITS_PER_CATEGORY:
                break
        out[key] = deduped
    return out


def extract_passages(text: str, patient_id: str = "", report_name: str = ""):
    system_prompt = load_prompt()
    user_prompt = f"""Evidenz-Bündel (regelbasiert aus dem Bericht; ggf. gekürzter Kurzbericht-Volltext ohne Snippet-Treffer):
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

        return normalize_extraction_result(result)

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