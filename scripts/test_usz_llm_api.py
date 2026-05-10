"""
Smoke test for the local USZ/Gemma generate API.

Sends a small JSON-only prompt and prints status code, raw response,
normalized final text, and json.loads parse result.

Usage:
    python scripts/test_usz_llm_api.py

Environment:
    USZ_LLM_URL    default: http://localhost:8100/generate
    USZ_TIMEOUT    default: 60 (seconds)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.llm_interface import extract_first_json_object


DEFAULT_URL = "http://localhost:8100/generate"


def _build_payload() -> dict:
    system_prompt = (
        "You answer ONLY in valid JSON. No prose, no markdown fences, no extra keys."
    )
    user_prompt = (
        "Return a JSON object with exactly these keys and types: "
        '"ok" (boolean true), "provider" (string "usz_api"), "n" (integer 42).'
    )
    return {
        "prompt": user_prompt.strip(),
        "system_prompt": system_prompt,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 1000,
        "disable_think": False,
    }


def main() -> int:
    url = os.getenv("USZ_LLM_URL", DEFAULT_URL)
    timeout = int(os.getenv("USZ_TIMEOUT", "60"))

    payload = _build_payload()
    print(f"POST {url}")
    print("payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except Exception as exc:
        print(f"NETWORK ERROR: {exc}")
        return 2

    print(f"status_code: {resp.status_code}")
    raw = resp.text or ""
    print("raw response (first 1000 chars):")
    print(raw[:1000])

    if resp.status_code != 200:
        print("HTTP status indicates failure; aborting.")
        return 1

    try:
        body = resp.json()
    except Exception as exc:
        print(f"Response body is not JSON: {exc}")
        return 1

    result = body.get("response", "")
    if isinstance(result, list):
        raw_text = "\n".join(str(x) for x in result)
    else:
        raw_text = str(result)
    final_text = extract_first_json_object(raw_text.strip())

    print("normalized final_text (after extract_first_json_object):")
    print(final_text)

    try:
        parsed = json.loads(final_text)
        print("json.loads(final_text): SUCCESS")
        print("parsed:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"json.loads(final_text): FAILED ({exc})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
