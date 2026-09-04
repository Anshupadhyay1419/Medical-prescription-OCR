"""
The one place the project talks to Ollama.

Every LLM stage (reranker, restructurer, corrector) used to carry its own copy
of this request code. They now share this function, so changing the endpoint,
the model, or the retry behaviour is a single edit.
"""
import requests

from prescription_ocr.config import OLLAMA_MODEL, OLLAMA_URL


def generate(prompt, temperature=0.1, num_predict=256, timeout=30, model=None, **options):
    """
    Send `prompt` to the local Ollama server and return the response text.

    Returns "" on any failure (server down, timeout, malformed reply). Every
    caller must treat an empty result as "keep the original text" — a failed
    LLM call is never allowed to lose OCR output.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, **options},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return response.json().get("response", "").strip()
