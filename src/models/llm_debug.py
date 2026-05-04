from pathlib import Path
from datetime import datetime
import json

from src.pipeline.paths import LLM_DEBUG_DIR


def _safe_name(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "unknown"
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)[:80]


def write_llm_debug(
    agent_name: str,
    patient_id: str,
    report_name: str,
    system_prompt: str,
    user_prompt: str,
    raw_output: str,
    error_message: str,
) -> Path:
    LLM_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_stem = f"{timestamp}_{_safe_name(agent_name)}_{_safe_name(patient_id)}_{_safe_name(report_name)}"
    out_path = LLM_DEBUG_DIR / f"{file_stem}.json"

    payload = {
        "agent_name": agent_name,
        "patient_id": patient_id,
        "report_name": report_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_output": raw_output,
        "error_message": error_message,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path