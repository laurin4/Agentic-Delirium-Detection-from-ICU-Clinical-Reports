"""
LLM provider configuration.

Primary production backend: USZ/Gemma local HTTP API (`usz_api`).
Optional comparison backend: Ollama (`ollama`).

Values are read from environment variables at import time.
"""

import os
from typing import Tuple

SUPPORTED_PROVIDERS: Tuple[str, ...] = ("usz_api", "ollama")


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# === Provider selection (default: USZ API for primary production runs) ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "usz_api").strip().lower()

# === USZ/Gemma API (primary) ===
USZ_LLM_URL = os.getenv("USZ_LLM_URL", "http://localhost:8100/generate")

# === Ollama (optional comparison / legacy backend — not required for primary run) ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11500")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# === Shared generation parameters (both providers consume these where applicable) ===
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_DISABLE_THINK = _parse_bool_env("LLM_DISABLE_THINK", False)

# Timeout: LLM_TIMEOUT wins, then legacy OLLAMA_TIMEOUT, then 120s.
TIMEOUT = int(os.getenv("LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "120")))

# Ollama-only context window (chat options)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Warn when system+user prompt characters exceed this (no truncation).
LLM_LONG_INPUT_WARNING_CHARS = int(os.getenv("LLM_LONG_INPUT_WARNING_CHARS", "12000"))

# === Model label for logging and prediction filename suffixes ===
# - Explicit LLM_MODEL_LABEL always wins (recommended for USZ runs).
# - If unset: `usz_api` → gemma4_26b_usz; `ollama` → OLLAMA_MODEL.
# OLLAMA_MODEL is never required when using usz_api unless you run Ollama.
_llm_model_label_env = os.getenv("LLM_MODEL_LABEL")
if _llm_model_label_env and _llm_model_label_env.strip():
    LLM_MODEL_LABEL = _llm_model_label_env.strip()
elif LLM_PROVIDER == "ollama":
    LLM_MODEL_LABEL = OLLAMA_MODEL
else:
    LLM_MODEL_LABEL = "gemma4_26b_usz"

# Request body model name for Ollama only.
MODEL_NAME = OLLAMA_MODEL
