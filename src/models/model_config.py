import os

# === LLM CONFIG ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11500")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Rückwärtskompatibler Alias
MODEL_NAME = OLLAMA_MODEL