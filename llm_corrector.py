"""
LLM-based contextual corrector for OCR output.
Uses local Qwen2.5-7B via Ollama.
"""
import requests
import re
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

CORRECTOR_PROMPT = """You are correcting OCR errors from a handwritten Indian medical prescription.

FIX these SPECIFIC OCR errors ONLY when they appear as standalone words:
   - "wit" or "but" or "bit" -> "Wt"  (ONLY when at start of line followed by " - <number> kg")
   - "Rix" or "Rex" -> "Rx" (ONLY when line contains only these characters)
   - "HLAIC" or "HLA1C" -> "HbA1c"
   - "Sir." -> "Sr." (ONLY when followed by a lab name like Creatinine)
   - "Creative" -> "Creatinine" (ONLY when preceded by "Sr.")
   - "Urine RIE" -> "Urine R/E"
   - "mmmy" -> "mmHg"

CRITICAL: Do NOT apply corrections inside the middle of a line where they don't fit context.
- "Reduce weight" should stay "Reduce weight" - do NOT change to "Reduce Wt"
- "Avoid smoking & Alcohol" should NOT become "Avoid smoking & HbA1c"
- Dr. name "Mukhopadhyay" should NOT get "Sr." prefix

PRESERVE UNCHANGED:
   - All dose schedules: 1-0-1, 0-0-1, 0-1-0
   - All dose strengths and units
   - All Indian drug brand names
   - All frequency codes: BD, TDS, HS, PC, SOS
   - Person names (proper nouns)
   - Any word in the middle of a sentence that already looks correct

If input is fine or you are unsure, return input EXACTLY as-is.

Respond with ONLY the corrected text on a single line. No prefixes. No explanations.

Input: {text}"""


def correct_line(text, verbose=False):
    original = text.strip()
    if len(original) < 3:
        return original
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": CORRECTOR_PROMPT.format(text=original),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            },
            timeout=30
        )
        result = response.json().get('response', '').strip()
        
        if not result:
            return original
        
        # ---- Strip LLM prompt leaks ----
        leak_prefixes = [
            'corrected line:', 'corrected:', 'output:', 'answer:',
            'result:', 'text:', 'response:', 'the corrected line is:',
            'the correction is:', 'unchanged:', 'input:',
        ]
        result_lower = result.lower()
        for prefix in leak_prefixes:
            if result_lower.startswith(prefix):
                result = result[len(prefix):].strip()
                result_lower = result.lower()
        
        # ---- Reject meta-words ----
        meta_words = ['unchanged', 'same', 'no change', 'no correction', 'n/a', 'none']
        if result.lower().strip('.,!') in meta_words:
            return original
        
        # ---- Length sanity ----
        if len(result) < len(original) * 0.4 or len(result) > len(original) * 2.5:
            return original
        
        # ---- Number protection ----
        original_nums = re.findall(r'\d+', original)
        result_nums = re.findall(r'\d+', result)
        original_num_set = set(original_nums)
        result_num_set = set(result_nums)
        
        # allow adding numbers (HLAIC → HbA1c gaining "1"), 
        # but reject if existing numbers were removed/changed
        removed_nums = original_num_set - result_num_set
        if removed_nums:
            if verbose:
                print(f"  [LLM] Removed numbers {removed_nums}, rejecting")
            return original
        
        # ---- Dose pattern protection ----
        dose_patterns = re.findall(r'\d+[-/]\d+(?:[-/]\d+)?', original)
        for pattern in dose_patterns:
            if pattern not in result:
                if verbose:
                    print(f"  [LLM] Dose pattern '{pattern}' lost, rejecting")
                return original
        
        # ---- Final cleanup ----
        result = result.strip('"\'`').split('\n')[0].strip()
        
        if verbose and result != original:
            print(f"  '{original}' -> '{result}'")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return original


def correct_lines_batch(lines, verbose=False):
    """Apply corrector to a list of lines."""
    corrected = []
    total = len(lines)
    start = time.time()
    
    for i, line in enumerate(lines):
        c = correct_line(line, verbose=verbose)
        corrected.append(c)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start
            eta = (elapsed / (i + 1)) * (total - i - 1)
            print(f"  Progress: {i + 1}/{total} ({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")
    
    return corrected


if __name__ == "__main__":
    test_lines = [
        "wit - 74 kg",
        "but - 74 kg",
        "Rix",
        "Sir. Creative",
        "HLAIC",
        "Tab. Telma-H 40/12.5",
        "1-0-1",
        "Urine RIE",
    ]
    
    print("\nTesting corrector:\n")
    for line in test_lines:
        corrected = correct_line(line, verbose=True)
        print(f"  IN:  {line}")
        print(f"  OUT: {corrected}\n")