"""
LLM-based document restructurer for medical prescriptions.
"""
import requests
import re

from postprocess import is_commentary, normalize_doses

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ============================================================
# BETTER MEDICATION DETECTION
# ============================================================

# common Indian medication name patterns (partial match)
MEDICATION_KEYWORDS = [
    # prefixes
    'tab.', 'tab ', 'cap.', 'cap ', 'syp.', 'syp ',
    'inj.', 'inj ', 'inhaler', 'oint', 'susp', 'drops',
    # common drug name components that appear frequently
    'metformin', 'telma', 'atorva', 'rantac', 'clopilet', 'omega',
    'gliminox', 'deriphyllin', 'ecosprin', 'amlokind', 'monti',
    'vit d', 'vit b', 'vitamin',
]


def count_medications(lines_data):
    """
    Count lines that look like medication entries.
    Handles both text-only and (text, coords) tuples.
    """
    count = 0
    for item in lines_data:
        text = item if isinstance(item, str) else item[0]
        text_lower = text.lower()
        if any(kw in text_lower for kw in MEDICATION_KEYWORDS):
            count += 1
    return count


def extract_medication_key(text):
    """
    Extract the distinctive medication name from a line.
    Returns lowercase key or None.
    """
    text_lower = text.lower()
    
    # try to find drug name after Tab./Cap./Syp./Inj. prefix
    match = re.search(
        r'(?:tab|cap|syp|inj|inhaler|susp)\.?\s+([a-z][\w\-]+)',
        text_lower
    )
    if match:
        return match.group(1).strip('.,')
    
    # if no prefix, check if line itself starts with known drug name
    for kw in MEDICATION_KEYWORDS:
        if kw in text_lower and len(kw) > 3:  # skip short keywords like "tab."
            # extract the word containing this keyword
            words = text_lower.split()
            for w in words:
                if kw in w:
                    return w.strip('.,')
    
    return None


# ============================================================
# RESTRUCTURE PROMPT (with explicit examples)
# ============================================================

RESTRUCTURE_PROMPT = """You are reorganizing OCR output from a medical prescription into proper reading order.

INPUT: A list of OCR text fragments with bounding box coordinates in format [x=X,y=Y,w=W,h=H].

CRITICAL RULES:

1. MERGE fragments that share SIMILAR y-coordinates (within 40 pixels - doctors write dosages slightly above/below drug names).
   - Sort fragments left-to-right by x-coordinate
   - Join them with spaces into ONE output line

2. Different y-coordinates (>40px apart) = different output lines. Keep them SEPARATE.

3. DOSAGE ATTACHMENT RULES (CRITICAL - patient safety):
   - Dosages (X-X-X patterns) belong to the CLOSEST medication with similar y-coordinate
   - Never move a dosage to a different medication row
   - If unsure which medication a dosage belongs to, keep it with the CLOSEST y-coordinate medication
   - "HS" means bedtime, "BD" twice daily, "TDS" thrice daily, "PC" after food — these belong with their same-row medication

4. Preserve original text EXACTLY. No spelling corrections. No additions.

5. PRESERVE prefixes like "Tab.", "Cap.", "Syp." - do NOT strip them.

6. Every input fragment must appear in output exactly ONCE.

EXAMPLES:

Input:
    [x=100,y=720] O. Tab. Atorva 20
    [x=550,y=700] 0-0-1 HS         ← y=700, close to Atorva y=720 (20px apart)
    [x=700,y=725] x 1 month

Correct output (merge these - similar y-coordinates):
    O. Tab. Atorva 20 0-0-1 HS x 1 month

Input:
    [x=100,y=980] Q Tab. Rantac 150
    [x=500,y=980] 0-1-0            ← same y=980
    [x=700,y=1010] (Before dinner) ← y=1010, close to 980 (30px apart)

Correct output (all similar y):
    Q Tab. Rantac 150 0-1-0 (Before dinner)

WRONG output (would swap dosages):
    O. Tab. Atorva 20 0-1-0           ← WRONG! 0-1-0 belongs to Rantac
    Q Tab. Rantac 150 0-0-1 HS        ← WRONG! 0-0-1 HS belongs to Atorva

COLUMN HANDLING (for bottom sections):
- Detect columns by clusters of similar x-coordinates
- Process one column FULLY top-to-bottom before moving to next column
- Do NOT interleave text across columns

INPUT LINES TO PROCESS:
{lines_with_coords}

OUTPUT (merged by similar y-coordinate, all medications preserved with CORRECT dosages):"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_llm_output(raw_text):
    """Extract clean lines from LLM output."""
    raw_lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    cleaned = []

    skip_prefixes = (
        'output:', 'output ', 'reconstructed:', 'here is', 'here are',
        'the following', 'note:', 'explanation:', '```', '---', '===',
        'input:', 'result:', 'response:', 'answer:', 'correct output:',
        'correct output', 'example', 'expected:',
    )

    for line in raw_lines:
        if line.lower().startswith(skip_prefixes):
            continue
        # the model frequently appends a sentence describing what it just did
        # ("This output preserves the original text exactly...") — that is not OCR
        if is_commentary(line):
            continue
        if len(line) < 3:
            continue

        # strip formatting prefixes
        line = re.sub(r'^\s*[-*•●○▪]\s*', '', line)
        line = re.sub(r'^\s*\d+[.)\]]\s*', '', line)
        line = re.sub(r'^\s*Line\s*\d+\s*:\s*', '', line, flags=re.IGNORECASE)
        line = line.strip('`"\'')

        if line.strip():
            cleaned.append(normalize_doses(line.strip()))

    return cleaned


def _count_visual_rows(lines_with_boxes, tol=40):
    """
    How many distinct text rows the page actually has, by clustering box
    y-centres. Used to detect when the LLM has collapsed everything onto one line.
    """
    centres = sorted((y1 + y2) / 2 for _, _, y1, _, y2 in lines_with_boxes)
    if not centres:
        return 0
    rows = 1
    last = centres[0]
    for c in centres[1:]:
        if c - last > tol:
            rows += 1
        last = c
    return rows


def deduplicate_lines(lines):
    """Remove exact duplicate consecutive lines."""
    if not lines:
        return lines
    result = [lines[0]]
    for line in lines[1:]:
        if line.strip().lower() != result[-1].strip().lower():
            result.append(line)
    return result


# ============================================================
# MAIN RESTRUCTURE FUNCTION
# ============================================================

def restructure_document(lines_with_boxes, timeout=180, verbose=False):
    """
    Reorganize OCR lines into proper reading order with safety checks.
    
    Args:
        lines_with_boxes: list of tuples (text, x1, y1, x2, y2)
        timeout: max seconds for LLM response
        verbose: print debug info
    
    Returns:
        list of restructured text lines
    """
    if not lines_with_boxes:
        return []
    
    # ---- Count medications in input ----
    input_med_count = count_medications(lines_with_boxes)
    input_med_keys = set()
    for text, *_ in lines_with_boxes:
        key = extract_medication_key(text)
        if key:
            input_med_keys.add(key)
    
    if verbose:
        print(f"  [Restructurer] Input: {len(lines_with_boxes)} lines")
        print(f"  [Restructurer] Detected {input_med_count} medication rows")
        print(f"  [Restructurer] Medication keys: {sorted(input_med_keys)}")
    
    # ---- Format input for LLM ----
    formatted = []
    for text, x1, y1, x2, y2 in lines_with_boxes:
        formatted.append(
            f"[x={x1:4d},y={y1:4d},w={x2-x1:3d},h={y2-y1:3d}] {text}"
        )
    input_text = "\n".join(formatted)
    
    # use .replace() for safety (avoids KeyError from OCR text with braces)
    prompt = RESTRUCTURE_PROMPT.replace("{lines_with_coords}", input_text)
    
    if verbose:
        print(f"  [Restructurer] Prompt size: ~{len(prompt)} chars, calling LLM...")
    
    # ---- Call LLM ----
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 3000,
                    "num_ctx": 8192,
                    "top_p": 0.9,
                }
            },
            timeout=timeout
        )
        result = response.json().get('response', '').strip()
        
        if not result:
            if verbose:
                print(f"  [Restructurer] Empty LLM response, using original")
            return [line[0] for line in lines_with_boxes]
        
        if verbose:
            print(f"  [Restructurer] LLM returned {len(result)} chars")
        
        # ---- Parse and clean output ----
        cleaned = parse_llm_output(result)
        cleaned = deduplicate_lines(cleaned)
        
        if verbose:
            print(f"  [Restructurer] Parsed {len(cleaned)} clean lines")

        # ---- SAFETY CHECK 0: Structural collapse ----
        # If the model answered as one unbroken paragraph instead of separate
        # lines, every field/medication ends up on a single line. That destroys
        # the document structure (and wrecks line-level scoring), so reject it.
        expected_rows = _count_visual_rows(lines_with_boxes)
        if cleaned and len(cleaned) < max(2, expected_rows * 0.4):
            print(f"  [Restructurer] WARNING: output collapsed to {len(cleaned)} line(s) "
                  f"but the page has ~{expected_rows} visual rows")
            print(f"  [Restructurer] Falling back to original")
            return [line[0] for line in lines_with_boxes]

        # ---- SAFETY CHECK 1: Character bounds ----
        input_char_count = sum(len(t) for t, *_ in lines_with_boxes)
        output_char_count = sum(len(l) for l in cleaned)
        
        if output_char_count < input_char_count * 0.4:
            print(f"  [Restructurer] WARNING: output too short "
                  f"({output_char_count} vs {input_char_count} chars)")
            print(f"  [Restructurer] Falling back to original")
            return [line[0] for line in lines_with_boxes]
        
        if output_char_count > input_char_count * 1.5:
            print(f"  [Restructurer] WARNING: output too long, likely hallucination")
            return [line[0] for line in lines_with_boxes]
        
        # ---- SAFETY CHECK 2: Medication preservation (by KEY, not just count) ----
        output_text_combined = ' '.join(cleaned).lower()
        missing_meds = []
        
        for text, x1, y1, x2, y2 in lines_with_boxes:
            key = extract_medication_key(text)
            if key and len(key) > 2:
                if key not in output_text_combined:
                    missing_meds.append(text)
        
        if missing_meds:
            print(f"  [Restructurer] ⚠️  MEDICATION SAFETY WARNING")
            print(f"  [Restructurer]    {len(missing_meds)} medication(s) missing:")
            for m in missing_meds:
                print(f"  [Restructurer]      - {m}")
            print(f"  [Restructurer]    Appending back to output...")
            
            # append missing meds at the end
            for m in missing_meds:
                cleaned.append(m)
                output_text_combined += ' ' + m.lower()
        
        output_med_count = count_medications(cleaned)
        if verbose:
            print(f"  [Restructurer] Output: {len(cleaned)} lines, "
                  f"{output_med_count} medications")
        
        return cleaned
    
    except requests.exceptions.Timeout:
        print(f"  [Restructurer] Timeout after {timeout}s, using original")
        return [line[0] for line in lines_with_boxes]
    except Exception as e:
        print(f"  [Restructurer] Error: {e}")
        import traceback
        traceback.print_exc()
        return [line[0] for line in lines_with_boxes]


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    test_input = [
        ("Rx", 60, 380, 100, 420),
        ("O. Tab. Telma-H 40/12.5", 100, 500, 400, 530),
        ("1-0-1", 500, 500, 600, 530),
        ("x 1 month", 700, 500, 850, 530),
        ("O. Tab. Metformin SR 500", 100, 550, 420, 580),
        ("1-0-1", 500, 550, 600, 580),
        ("x 1 month", 700, 550, 850, 580),
        ("O. Tab. Atorva 20", 100, 750, 320, 780),
        ("0-0-1 HS", 500, 750, 640, 780),
        ("x 1 month", 700, 750, 850, 780),
        ("O. Tab. Rantac 150", 100, 900, 340, 930),
        ("0-1-0", 500, 900, 600, 930),
        ("(Before dinner)", 620, 900, 800, 930),
        ("x 15 days", 820, 900, 950, 930),
        ("Advice -", 60, 1100, 150, 1130),
        ("Low salt diet", 80, 1140, 250, 1170),
        ("Avoid oily food", 80, 1180, 260, 1210),
        ("Investigations -", 400, 1100, 550, 1130),
        ("FBS, PPBS", 420, 1140, 550, 1170),
        ("HbA1c", 420, 1180, 500, 1210),
    ]
    
    print("=" * 60)
    print("INPUT:")
    print("=" * 60)
    for text, x1, y1, x2, y2 in test_input:
        print(f"  [x={x1:4d},y={y1:4d}] {text}")
    
    print(f"\nDetected medications: {count_medications(test_input)}")
    print(f"Detected keys: {sorted({extract_medication_key(t) for t, *_ in test_input if extract_medication_key(t)})}")
    
    print("\n" + "=" * 60)
    print("OUTPUT:")
    print("=" * 60)
    result = restructure_document(test_input, verbose=True)
    print()
    for i, line in enumerate(result):
        print(f"  {i}: {line}")
    
    print(f"\nOutput medications: {count_medications(result)}")
    print(f"Expected: 4 (Telma-H, Metformin, Atorva, Rantac)")