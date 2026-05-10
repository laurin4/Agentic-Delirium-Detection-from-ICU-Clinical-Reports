"""
Provider-agnostic LLM interface.

Primary: USZ local generate API (`LLM_PROVIDER=usz_api`).
Optional comparison: Ollama chat API (`LLM_PROVIDER=ollama`).

Exposes:
- call_llm(messages) -> str       (existing public API; routes by LLM_PROVIDER)
- call_usz_api(system_prompt, user_prompt) -> str

Long report handling may need future chunking/summarization depending on real
model context limits; we only log warnings today (no truncation).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import requests

from src.models.model_config import (
    LLM_DISABLE_THINK,
    LLM_LONG_INPUT_WARNING_CHARS,
    LLM_MAX_TOKENS,
    LLM_MODEL_LABEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    MODEL_NAME,
    OLLAMA_NUM_CTX,
    OLLAMA_URL,
    SUPPORTED_PROVIDERS,
    TIMEOUT,
    USZ_LLM_URL,
)

LOGGER = logging.getLogger(__name__)

# USZ/Gemma template tokens sometimes appended after the model JSON payload.
_USZ_TOKENS_TO_STRIP = (
    "<start_of_turn>user",
    "<start_of_turn>model",
    "<start_of_turn>",
    "<end_of_turn>",
)


def _strip_usz_template_tokens(text: str) -> str:
    """Remove known Gemma/USZ turn markers so JSON extraction is reliable."""
    out = text
    for tok in _USZ_TOKENS_TO_STRIP:
        out = out.replace(tok, "")
    return out


def extract_first_json_object(text: str) -> str:
    """
    Return the first complete top-level JSON object substring, or stripped original text.

    Uses brace matching with awareness of double-quoted strings and backslash escapes.
    """
    if text is None:
        return ""
    original_stripped = text.strip()
    if not original_stripped:
        return ""

    cleaned = _strip_usz_template_tokens(original_stripped).strip()

    start = cleaned.find("{")
    if start < 0:
        return original_stripped

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()

    return original_stripped


def _build_chat_url(base_url: str) -> str:
    clean = base_url.rstrip("/")

    if clean.endswith("/api/chat"):
        return clean
    if clean.endswith("/api/generate"):
        return f"{clean[:-len('/api/generate')]}/api/chat"
    if clean.endswith("/api"):
        return f"{clean}/chat"

    return f"{clean}/api/chat"


def _extract_system_user(messages: list) -> Tuple[str, str]:
    """Collapse a chat-style messages list into (system_prompt, user_prompt)."""
    sys_parts: List[str] = []
    user_parts: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            sys_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    return "\n\n".join(p for p in sys_parts if p), "\n\n".join(p for p in user_parts if p)


def _log_input(provider: str, model_label: str, system_prompt: str, user_prompt: str) -> None:
    sys_len = len(system_prompt or "")
    user_len = len(user_prompt or "")
    total = sys_len + user_len
    print(
        "[LLM] provider={p} model={m} system_len={s} user_len={u} total_len={t}".format(
            p=provider, m=model_label, s=sys_len, u=user_len, t=total
        )
    )
    LOGGER.info(
        "LLM call provider=%s model=%s system_len=%d user_len=%d total_chars=%d",
        provider,
        model_label,
        sys_len,
        user_len,
        total,
    )
    if total > LLM_LONG_INPUT_WARNING_CHARS:
        warn_msg = (
            "Long LLM input detected ({t} chars > {th}); "
            "output may depend on backend context length.".format(
                t=total, th=LLM_LONG_INPUT_WARNING_CHARS
            )
        )
        print(f"[LLM][WARN] {warn_msg}")
        LOGGER.warning(warn_msg)


def _call_ollama_messages(messages: list) -> str:
    try:
        chat_url = _build_chat_url(OLLAMA_URL)

        response = requests.post(
            chat_url,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "num_predict": LLM_MAX_TOKENS,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError("Antwort von Ollama enthält keinen gültigen Text unter 'message.content'.")

        return content.strip()

    except Exception as e:
        print(f"LLM-Fehler (ollama): {e}")
        return ""


def call_usz_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call the local USZ/Gemma generate API and return the normalized text.

    Endpoint: USZ_LLM_URL (default http://localhost:8100/generate).

    Raises RuntimeError on non-200 HTTP status or non-JSON response.
    """
    payload = {
        "prompt": (user_prompt or "").strip(),
        "system_prompt": system_prompt or "",
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
        "disable_think": LLM_DISABLE_THINK,
    }

    try:
        response = requests.post(USZ_LLM_URL, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        raise RuntimeError(f"USZ LLM API request failed ({USZ_LLM_URL}): {exc}") from exc

    if response.status_code != 200:
        snippet = (response.text or "")[:500]
        raise RuntimeError(
            f"USZ LLM API returned HTTP {response.status_code} from {USZ_LLM_URL}: {snippet}"
        )

    try:
        body = response.json()
    except Exception as exc:
        snippet = (response.text or "")[:500]
        raise RuntimeError(
            f"USZ LLM API returned non-JSON response from {USZ_LLM_URL}: {exc}; body[:500]={snippet}"
        ) from exc

    result = body.get("response", "")
    if isinstance(result, list):
        final_text = "\n".join(str(x) for x in result)
    else:
        final_text = str(result)
    final_text = extract_first_json_object(final_text.strip())
    return final_text


def call_llm(messages: list) -> str:
    """
    Provider-agnostic entry point used by the agents.

    Existing callers pass a chat-style messages list; for the USZ API we
    internally collapse it to (system_prompt, user_prompt).
    """
    provider = LLM_PROVIDER
    system_prompt, user_prompt = _extract_system_user(messages)
    _log_input(provider, LLM_MODEL_LABEL, system_prompt, user_prompt)

    if provider == "ollama":
        return _call_ollama_messages(messages)

    if provider == "usz_api":
        try:
            return call_usz_api(system_prompt, user_prompt)
        except Exception as exc:
            print(f"LLM-Fehler (usz_api): {exc}")
            return ""

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Allowed providers: {SUPPORTED_PROVIDERS}"
    )
