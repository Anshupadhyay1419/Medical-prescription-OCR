"""
Deterministic post-processing shared by the TrOCR and Qwen-VL pipelines.

Two jobs, both of which were previously leaking errors into final output:
  1. Strip LLM meta-commentary ("This output preserves the original text...").
     Any LLM asked to restructure text will occasionally narrate what it did;
     that narration was being saved as if it were OCR output.
  2. Normalise dose schedules. Handwritten 1 and 0 are routinely read as I and O,
     so "I-O-1" must become "1-0-1". These patterns are the highest-value
     characters on a prescription, so we fix them with regex rather than hoping
     an LLM does it.
"""
import re

# ---------------------------------------------------------------
# 1. LLM commentary detection
# ---------------------------------------------------------------

# Lines that open with one of these are the model talking about its own answer.
COMMENTARY_PREFIXES = (
    'this output', 'the output', 'the text is', 'the text has', 'the following',
    'here is', 'here are', 'note that', 'note:', 'explanation:', 'summary:',
    'in this output', 'each fragment', 'medications and', 'dosage attachment',
    'all fragments', 'the medications', 'the dosages', 'as per the rules',
    'this preserves', 'this transcription', 'the transcription',
    'i have', "i've", 'please note', 'output:', 'result:', 'answer:',
    'signature:', 'registration no:',
)

# Phrases that mark a line as commentary no matter where they appear in it.
COMMENTARY_PHRASES = (
    'preserves the original', 'preserve the original',
    'follows the rules', 'following the rules', 'rules provided',
    'correctly grouped', 'patient safety', 'y-coordinate', 'y coordinate',
    'bounding box', 'reading order', 'exactly as provided',
    'is divided into sections', 'organized into logical',
    'based on the coordinates', 'each medication and its',
)


def is_commentary(line):
    """True if the line is the LLM describing its own output rather than OCR text."""
    low = line.strip().lower()
    if not low:
        return False
    if low.startswith(COMMENTARY_PREFIXES):
        return True
    return any(p in low for p in COMMENTARY_PHRASES)


# ---------------------------------------------------------------
# 2. Dose-schedule normalisation
# ---------------------------------------------------------------

_DIGIT_MAP = {'O': '0', 'o': '0', 'Q': '0', 'D': '0',
              'I': '1', 'i': '1', 'l': '1', '|': '1'}

# three slots separated by - or – , each slot a 0/1 or a lookalike
_DOSE_3 = re.compile(r'(?<![\w-])([01OoQDIil|])\s*[-–—]\s*([01OoQDIil|])\s*[-–—]\s*([01OoQDIil|])(?![\w-])')


def _fix_dose(m):
    return '-'.join(_DIGIT_MAP.get(g, g) for g in m.groups())


def normalize_doses(text):
    """I-O-1 -> 1-0-1, O-O-I -> 0-0-1, etc."""
    return _DOSE_3.sub(_fix_dose, text)


# ---------------------------------------------------------------
# 3. Formatting cleanup
# ---------------------------------------------------------------

def clean_line(line):
    """Strip markdown artefacts and normalise whitespace/dose codes."""
    s = line.strip()
    s = re.sub(r'^\s*[-*•●○▪]\s+', '', s)          # bullet markers
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)          # **bold**
    s = re.sub(r'^\s*#+\s*', '', s)                 # markdown headings
    s = re.sub(r'^\s*Line\s*\d+\s*:\s*', '', s, flags=re.IGNORECASE)
    s = s.strip('`')
    s = re.sub(r'\s+', ' ', s)                      # collapse whitespace
    s = normalize_doses(s)
    return s.strip()


def clean_ocr_output(lines):
    """Full cleanup pass over a list of OCR lines. Drops commentary, keeps order."""
    out = []
    for raw in lines:
        for part in str(raw).split('\n'):
            if is_commentary(part):
                continue
            s = clean_line(part)
            if s and s not in ('---', '===', '```'):
                out.append(s)
    return out


# ---------------------------------------------------------------
# 4. Text normalisation used for scoring
# ---------------------------------------------------------------

def normalize_for_scoring(text):
    """
    Case/spacing normalisation so that cosmetic differences (TDS vs tds,
    5ml vs 5 ml, Rx vs RX) are not counted as recognition errors.
    Reported alongside — never instead of — the raw metric.
    """
    t = text.lower()
    t = re.sub(r'[“”]', '"', t)
    t = re.sub(r'[‘’]', "'", t)
    t = re.sub(r'[–—]', '-', t)
    # split number/unit joins: 5ml -> 5 ml, 40mg -> 40 mg
    t = re.sub(r'(\d)\s*(ml|mg|kg|mcg|gm|g|d|days|day)\b', r'\1 \2', t)
    t = re.sub(r'[^\w\s/\-.:]', ' ', t)   # drop stray punctuation, keep / - . :
    # Detokenise: recognisers vary in whether they put a space around punctuation
    # ("MBBS , DNB ( Medicine )" vs "MBBS, DNB (Medicine)"). That is a spacing
    # convention, not a recognition error, so it is normalised away on BOTH the
    # reference and the hypothesis before scoring.
    t = re.sub(r'\s+([,.;:!?)\]])', r'\1', t)
    t = re.sub(r'([(\[])\s+', r'\1', t)
    t = re.sub(r'\s*-\s*', '-', t)        # Dolo - 650 -> Dolo-650
    t = re.sub(r'\s*/\s*', '/', t)        # 23 / 11 / 19 -> 23/11/19
    t = re.sub(r'\s+', ' ', t)
    return t.strip()
