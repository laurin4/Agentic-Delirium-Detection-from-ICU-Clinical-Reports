import requests

from src.models.model_config import OLLAMA_URL, MODEL_NAME, TIMEOUT


def _build_chat_url(base_url: str) -> str:
    clean = base_url.rstrip("/")

    if clean.endswith("/api/chat"):
        return clean
    if clean.endswith("/api/generate"):
        return f"{clean[:-len('/api/generate')]}/api/chat"
    if clean.endswith("/api"):
        return f"{clean}/chat"

    return f"{clean}/api/chat"


def call_llm(messages: list) -> str:
    try:
        chat_url = _build_chat_url(OLLAMA_URL)

        response = requests.post(
            chat_url,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1024,
                    "num_ctx": 4096,
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
        print(f"LLM-Fehler: {e}")
        return ""