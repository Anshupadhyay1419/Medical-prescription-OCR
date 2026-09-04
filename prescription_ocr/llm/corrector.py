import difflib
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

# A corrected word must still resemble the word it came from. Below this
# similarity the model has substituted a different word, not repaired a
# misread one — which is how "Levocetirizine" turns into something fluent
# and wrong.
MIN_WORD_SIMILARITY = 0.6

_WORD = re.compile(r"[A-Za-z]+")


def _strip_leaked_prefixes(result):
    lowered = result.lower()
    for prefix in LEAK_PREFIXES:
        if lowered.startswith(prefix):
            result = result[len(prefix):].strip()
            lowered = result.lower()
    return result


def _introduces_new_words(original, result, verbose=False):
    """
    True if `result` contains vocabulary that is not a repair of the input.

    This is the check that actually stops hallucination. A character-level fix
    leaves every word recognisably derived from the word it replaced; an
    invented answer does not. Anything that fails is rejected wholesale.
    """
    source = _WORD.findall(original.lower())
    produced = _WORD.findall(result.lower())

    if not source:
        return bool(produced)
    if abs(len(produced) - len(source)) > 1:
        if verbose:
            print(f"  [Corrector] word count {len(source)} -> {len(produced)}, rejecting")
        return True

    for word in produced:
        if word in source:
            continue
        best = max(difflib.SequenceMatcher(None, word, s).ratio() for s in source)
        if best < MIN_WORD_SIMILARITY:
            if verbose:
                print(f"  [Corrector] invented word {word!r}, rejecting")
            return True
    return False


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

    if _introduces_new_words(original, result, verbose=verbose):
        return False

    # Numbers may be gained (HLAIC -> HbA1c gains a "1") but never lost.
    removed = set(re.findall(r'\d+', original)) - set(re.findall(r'\d+', result))
    if removed:
        if verbose:
            print(f"  [Corrector] removed numbers {removed}, rejecting")
        return False

    # Dose schedules must come through untouched, spacing included: turning
    # "1-0-1" into "1 - 0 - 1" reads the same to a human but stops being a
    # recognisable dose code.
    for pattern in re.findall(r'\d+\s*[-/]\s*\d+(?:\s*[-/]\s*\d+)?', original):
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
