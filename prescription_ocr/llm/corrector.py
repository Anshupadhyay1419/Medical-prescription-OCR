"""
Point B — line-level LLM correction.

Fixes the specific OCR confusions that recur on Indian prescriptions ("Rix" for
"Rx", "HLAIC" for "HbA1c"). Edit prompts/corrector.txt to change what it fixes.

Every correction must survive a set of guard rails before it is accepted. An LLM
that rewrites a dose schedule is more dangerous than one that leaves a typo, so
anything suspicious falls back to the original text.
"""
import re
import time

from prescription_ocr.llm import prompts
from prescription_ocr.llm.client import generate

# Prefixes the model bolts onto its answer despite being told not to.
LEAK_PREFIXES = (
    'corrected line:', 'corrected:', 'output:', 'answer:',
    'result:', 'text:', 'response:', 'the corrected line is:',
    'the correction is:', 'unchanged:', 'input:',
)

# Replies that mean "no change" rather than being the corrected text itself.
META_WORDS = ('unchanged', 'same', 'no change', 'no correction', 'n/a', 'none')

MIN_LENGTH_RATIO = 0.4    # a much shorter reply means text was dropped
MAX_LENGTH_RATIO = 2.5    # a much longer reply means the model started talking


def _strip_leaked_prefixes(result):
    lowered = result.lower()
    for prefix in LEAK_PREFIXES:
        if lowered.startswith(prefix):
            result = result[len(prefix):].strip()
            lowered = result.lower()
    return result


def _is_safe(original, result, verbose=False):
    """
    True only if `result` is a plausible correction of `original`.

    The checks exist because a rejected correction costs a typo, while an
    accepted hallucination costs a wrong dose on a prescription.
    """
    if result.lower().strip('.,!') in META_WORDS:
        return False

    if not (len(original) * MIN_LENGTH_RATIO
            <= len(result)
            <= len(original) * MAX_LENGTH_RATIO):
        return False

    # Numbers may be gained (HLAIC -> HbA1c gains a "1") but never lost.
    removed = set(re.findall(r'\d+', original)) - set(re.findall(r'\d+', result))
    if removed:
        if verbose:
            print(f"  [Corrector] removed numbers {removed}, rejecting")
        return False

    # Dose schedules must come through untouched.
    for pattern in re.findall(r'\d+[-/]\d+(?:[-/]\d+)?', original):
        if pattern not in result:
            if verbose:
                print(f"  [Corrector] dose pattern '{pattern}' lost, rejecting")
            return False

    return True


def correct_line(text, verbose=False):
    """Correct a single line, returning it unchanged if anything looks wrong."""
    original = text.strip()
    if len(original) < 3:
        return original

    try:
        result = generate(
            prompts.render("corrector", text=original),
            temperature=0.1,
            num_predict=100,
            timeout=30,
        )
        if not result:
            return original

        result = _strip_leaked_prefixes(result)
        if not _is_safe(original, result, verbose=verbose):
            return original

        result = result.strip('"\'`').split('\n')[0].strip()
        if verbose and result != original:
            print(f"  '{original}' -> '{result}'")
        return result

    except Exception as e:
        if verbose:
            print(f"  [Corrector] error: {e}")
        return original


def correct_lines_batch(lines, verbose=False):
    """Apply the corrector to a list of lines, printing progress with an ETA."""
    corrected = []
    total = len(lines)
    start = time.time()

    for i, line in enumerate(lines):
        corrected.append(correct_line(line, verbose=verbose))
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start
            eta = (elapsed / (i + 1)) * (total - i - 1)
            print(f"  Progress: {i + 1}/{total} "
                  f"({elapsed:.0f}s elapsed, {eta:.0f}s remaining)")

    return corrected
