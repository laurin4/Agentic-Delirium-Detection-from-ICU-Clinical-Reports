import json
import re
from typing import Any, Dict


def parse_llm_json_output(raw_output: str, context_name: str) -> Dict[str, Any]:
    print(f"=== ROHE LLM-AUSGABE ({context_name}) START ===")
    print(raw_output)
    print(f"=== ROHE LLM-AUSGABE ({context_name}) ENDE ===")

    cleaned = (raw_output or "").strip()
    if not cleaned:
        raise ValueError(f"Leere LLM-Antwort in {context_name}: Kein JSON-Inhalt vorhanden.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(
            f"Keine JSON-Struktur in der LLM-Antwort gefunden ({context_name})."
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON konnte in {context_name} nicht robust geparst werden: {exc}"
        ) from exc
