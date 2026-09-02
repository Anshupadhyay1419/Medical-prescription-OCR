import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

RERANKER_PROMPT = """You are given multiple OCR transcription candidates for the SAME handwritten line from a medical prescription. Pick the most plausible one.

Candidates:
{candidates}

Selection criteria (in order):
1. Prefer candidates with valid dosage patterns (X-X-X, X mg, X/X)
2. Prefer candidates with valid medical abbreviations (BD, TDS, HS, mg, ml)
3. Prefer candidates with real English/medical words over gibberish
4. Prefer shorter candidates if longer ones look repetitive

Reply with ONLY the number of the best candidate (1, 2, 3, 4, or 5).
Nothing else. No explanation.

Best:"""


def rerank_candidates(candidates, timeout=15):
    """Pick best candidate using LLM. Fallback to first on error."""
    # dedupe while preserving order
    seen, unique = set(), []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            unique.append(c_clean)
    
    if len(unique) <= 1:
        return unique[0] if unique else ""
    
    formatted = "\n".join(f"{i+1}. {c}" for i, c in enumerate(unique))
    prompt = RERANKER_PROMPT.format(candidates=formatted)
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 10}
            },
            timeout=timeout
        )
        result = response.json().get('response', '').strip()
        match = re.search(r'\d+', result)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(unique):
                return unique[idx]
    except Exception as e:
        print(f"  [Reranker] Error: {e}")
    
    return unique[0]  # fallback to first (greedy-equivalent)