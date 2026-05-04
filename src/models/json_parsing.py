import json
import re
from typing import Any, Dict


def parse_llm_json_output(raw_output: str, context_name: str) -> Dict[str, Any]:
    print(f"=== ROHE LLM-AUSGABE ({context_name}) START ===")
    print(raw_output)
    print(f"=== ROHE LLM-AUSGABE ({context_name}) ENDE ===")

    if not raw_output or not raw_output.strip():
        raise ValueError(f"Leere LLM-Antwort in {context_name}: Kein JSON-Inhalt vorhanden.")

    text = raw_output.strip()

    # Markdown-Codeblock entfernen
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 1) Direktes Parsing versuchen
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Erstes JSON-Objekt extrahieren
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as exc:
            raise ValueError(
                f"JSON konnte in [{context_name}] nicht robust geparst werden: {exc}"
            )

    raise ValueError(f"Keine JSON-Struktur in der LLM-Antwort gefunden [{context_name}].")