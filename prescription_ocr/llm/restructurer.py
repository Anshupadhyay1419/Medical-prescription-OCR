import re

import requests

from prescription_ocr.llm import prompts
from prescription_ocr.llm.client import generate
from prescription_ocr.postprocess import is_commentary, normalize_doses

# --------------------------------------------------------------------------
# Medication detection
# --------------------------------------------------------------------------

# Dose forms only. An earlier version also listed the specific drug names that
# happened to appear in this corpus (telma, atorva, rantac, ...), which made the
# safety check silently useless on any prescription outside it. Dose-form
# prefixes generalise; brand names do not.
MEDICATION_KEYWORDS = [
    'tab.', 'tab ', 'cap.', 'cap ', 'syp.', 'syp ', 'susp', 'sus.',
    'inj.', 'inj ', 'inhaler', 'oint', 'drops', 'sol.', 'lotion',
    'cream', 'gel ', 'powder', 'sachet', 'vial', 'amp.', 'neb ',
    'vit ', 'vitamin', 'tablet', 'capsule', 'syrup', 'injection',
]

# Rows are considered the same visual line within this many pixels.
ROW_TOLERANCE_PX = 40

# Output must stay within these multiples of the input character count.
MIN_CHAR_RATIO = 0.4
MAX_CHAR_RATIO = 1.5

# Below this fraction of the page's visual rows, the model has collapsed the
# document into a paragraph and the output is unusable.
MIN_ROW_RATIO = 0.4


def count_medications(lines_data):
    """How many lines look like medication entries. Accepts text or tuples."""
    count = 0
    for item in lines_data:
        text = item if isinstance(item, str) else item[0]
        if any(kw in text.lower() for kw in MEDICATION_KEYWORDS):
            count += 1
    return count


def extract_medication_key(text):
    """The distinctive drug name in a line, lowercased, or None."""
    text_lower = text.lower()

    match = re.search(r'(?:tab|cap|syp|inj|inhaler|susp)\.?\s+([a-z][\w\-]+)', text_lower)
    if match:
        return match.group(1).strip('.,')

    # No dose-form prefix — fall back to a known drug name anywhere in the line.
    for keyword in MEDICATION_KEYWORDS:
        if keyword in text_lower and len(keyword) > 3:
            for word in text_lower.split():
                if keyword in word:
                    return word.strip('.,')

    return None


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

SKIP_PREFIXES = (
    'output:', 'output ', 'reconstructed:', 'here is', 'here are',
    'the following', 'note:', 'explanation:', '```', '---', '===',
    'input:', 'result:', 'response:', 'answer:', 'correct output:',
    'correct output', 'example', 'expected:',
)


def parse_llm_output(raw_text):
    """Turn the model's reply into clean transcription lines."""
    cleaned = []

    for line in (l.strip() for l in raw_text.split('\n')):
        if not line or len(line) < 3:
            continue
        if line.lower().startswith(SKIP_PREFIXES):
            continue
        # The model often appends a sentence describing what it just did
        # ("This output preserves the original text...") — that is not OCR.
        if is_commentary(line):
            continue

        line = re.sub(r'^\s*[-*•●○▪]\s*', '', line)
        line = re.sub(r'^\s*\d+[.)\]]\s*', '', line)
        line = re.sub(r'^\s*Line\s*\d+\s*:\s*', '', line, flags=re.IGNORECASE)
        line = line.strip('`"\'')

        if line.strip():
            cleaned.append(normalize_doses(line.strip()))

    return cleaned


def deduplicate_lines(lines):
    """Drop consecutive duplicate lines."""
    if not lines:
        return lines
    result = [lines[0]]
    for line in lines[1:]:
        if line.strip().lower() != result[-1].strip().lower():
            result.append(line)
    return result


def count_visual_rows(lines_with_boxes, tol=ROW_TOLERANCE_PX):
    """How many distinct text rows the page has, by clustering box y-centres."""
    centres = sorted((y1 + y2) / 2 for _, _, y1, _, y2 in lines_with_boxes)
    if not centres:
        return 0
    rows, last = 1, centres[0]
    for centre in centres[1:]:
        if centre - last > tol:
            rows += 1
        last = centre
    return rows


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def restructure_document(lines_with_boxes, timeout=180, verbose=False):
    """
    Reflow [(text, x1, y1, x2, y2), ...] into reading-order lines.

    Always returns usable text: if the LLM fails any safety check, the original
    per-box text is returned unchanged.
    """
    if not lines_with_boxes:
        return []

    original_lines = [line[0] for line in lines_with_boxes]

    input_med_keys = {key for text, *_ in lines_with_boxes
                      if (key := extract_medication_key(text))}

    if verbose:
        print(f"  [Restructurer] Input: {len(lines_with_boxes)} lines")
        print(f"  [Restructurer] Detected {count_medications(lines_with_boxes)} medication rows")
        print(f"  [Restructurer] Medication keys: {sorted(input_med_keys)}")

    formatted = "\n".join(
        f"[x={x1:4d},y={y1:4d},w={x2 - x1:3d},h={y2 - y1:3d}] {text}"
        for text, x1, y1, x2, y2 in lines_with_boxes
    )
    prompt = prompts.render("restructurer", lines_with_coords=formatted)

    if verbose:
        print(f"  [Restructurer] Prompt size: ~{len(prompt)} chars, calling LLM...")

    try:
        result = generate(prompt, temperature=0.1, num_predict=3000,
                          timeout=timeout, num_ctx=8192, top_p=0.9)

        if not result:
            if verbose:
                print("  [Restructurer] Empty LLM response, using original")
            return original_lines

        if verbose:
            print(f"  [Restructurer] LLM returned {len(result)} chars")

        cleaned = deduplicate_lines(parse_llm_output(result))

        if verbose:
            print(f"  [Restructurer] Parsed {len(cleaned)} clean lines")

        # -- Check 1: structural collapse -----------------------------------
        # One unbroken paragraph puts every field on a single line, destroying
        # the document structure and wrecking line-level scoring.
        expected_rows = count_visual_rows(lines_with_boxes)
        if cleaned and len(cleaned) < max(2, expected_rows * MIN_ROW_RATIO):
            print(f"  [Restructurer] WARNING: output collapsed to {len(cleaned)} line(s) "
                  f"but the page has ~{expected_rows} visual rows")
            print("  [Restructurer] Falling back to original")
            return original_lines

        # -- Check 2: character bounds --------------------------------------
        input_chars = sum(len(t) for t, *_ in lines_with_boxes)
        output_chars = sum(len(l) for l in cleaned)

        if output_chars < input_chars * MIN_CHAR_RATIO:
            print(f"  [Restructurer] WARNING: output too short "
                  f"({output_chars} vs {input_chars} chars)")
            print("  [Restructurer] Falling back to original")
            return original_lines

        if output_chars > input_chars * MAX_CHAR_RATIO:
            print("  [Restructurer] WARNING: output too long, likely hallucination")
            return original_lines

        # -- Check 3: medication preservation -------------------------------
        # Matched by drug name, not by count, so a substitution is caught too.
        combined = ' '.join(cleaned).lower()
        missing = [text for text, *_ in lines_with_boxes
                   if (key := extract_medication_key(text))
                   and len(key) > 2 and key not in combined]

        if missing:
            print("  [Restructurer] ⚠️  MEDICATION SAFETY WARNING")
            print(f"  [Restructurer]    {len(missing)} medication(s) missing:")
            for line in missing:
                print(f"  [Restructurer]      - {line}")
            print("  [Restructurer]    Appending back to output...")
            cleaned.extend(missing)

        if verbose:
            print(f"  [Restructurer] Output: {len(cleaned)} lines, "
                  f"{count_medications(cleaned)} medications")

        return cleaned

    except requests.exceptions.Timeout:
        print(f"  [Restructurer] Timeout after {timeout}s, using original")
        return original_lines
    except Exception as e:
        print(f"  [Restructurer] Error: {type(e).__name__}: {e}")
        return original_lines
