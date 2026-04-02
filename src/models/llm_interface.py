import requests
from src.pipeline.config import OLLAMA_URL, MODEL_NAME

def call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        # Wichtig: Falls deine OLLAMA_URL auf ".../api/generate" endet, 
        # biegen wir das hier sicherheitshalber auf die Chat-API um.
        chat_url = OLLAMA_URL.replace("/api/generate", "/api/chat")
        if not chat_url.endswith("/api/chat"):
            chat_url = "http://localhost:11434/api/chat" # Fallback

        response = requests.post(
            chat_url,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0}
            },
            timeout=120
        )
        response.raise_for_status()
        
        # Bei der Chat-API liegt die Antwort verschachtelt in 'message' -> 'content'
        return response.json()["message"]["content"]
        
    except Exception as e:
        print(f"LLM Fehler: {e}")
        return ""