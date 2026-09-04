"""
Scoring OCR output against the references in data/ground_truth/.

Headline metrics are document-level CER/WER: the whole page is joined into one
string before comparison. That is the standard measure for full-page OCR and is
insensitive to how the text happened to be split into lines.

  raw         exact comparison, nothing normalised.
  normalized  lowercased, whitespace and punctuation normalised, unit joins
              split (5ml -> 5 ml). Cosmetic differences stop counting as
              recognition errors, isolating genuine misreads.

Corpus totals aggregate as (total edits / total reference length) rather than
averaging per-file rates, so one short file cannot dominate the result.

For a prescription system the clinical metrics matter more than CER: getting
the drug name and the dose right is the entire point.

NOTE: an earlier version reported the average of a per-line "best match" CER.
That divides an edit distance by the length of a possibly very short reference
line, so one collapsed-paragraph prediction could score a CER of 29. Those
numbers were measurement artefacts. It survives below only as a diagnostic,
clamped to [0, 1].
"""
import os
import re
from collections import Counter
from difflib import SequenceMatcher

import jiwer

from prescription_ocr.config import GROUND_TRUTH_DIR, OUTPUT_PREFIX
from prescription_ocr.io_utils import ground_truth_path, load_lines, output_path
from prescription_ocr.postprocess import normalize_for_scoring

COUNT_KEYS = ('c_err', 'c_ref', 'w_err', 'w_ref')


# --------------------------------------------------------------------------
# Edit-distance counts
# --------------------------------------------------------------------------

def counts(reference, hypothesis):
    """Character- and word-level edit counts for one document."""
    if not reference.strip():
        return None
    c = jiwer.process_characters([reference], [hypothesis])
    w = jiwer.process_words([reference], [hypothesis])
    return {
        'c_err': c.substitutions + c.deletions + c.insertions,
        'c_ref': c.substitutions + c.deletions + c.hits,
        'w_err': w.substitutions + w.deletions + w.insertions,
        'w_ref': w.substitutions + w.deletions + w.hits,
    }


def rate(err, ref):
    return err / ref if ref else 0.0


# --------------------------------------------------------------------------
# Clinical metrics
# --------------------------------------------------------------------------

DRUG_RE = re.compile(r'\b(?:Tab|Cap|Syp|Inj|Inhaler|Susp|T)\.?\s+([A-Za-z][\w\-]{2,})',
                     re.IGNORECASE)
DOSE_RE = re.compile(r'\b\d-\d-\d\b')


def extract_drugs(lines):
    """The set of drug names named on a page."""
    return {m.group(1).lower().strip('.,') for l in lines for m in DRUG_RE.finditer(l)}


def extract_doses(lines):
    """Dose schedules as a multiset — the same schedule can appear many times."""
    return Counter(d for l in lines for d in DOSE_RE.findall(l))


def recall(truth_set, pred_set):
    if not truth_set:
        return None
    return len(truth_set & pred_set) / len(truth_set)


def counter_recall(truth_counter, pred_counter):
    total = sum(truth_counter.values())
    if not total:
        return None
    return sum((truth_counter & pred_counter).values()) / total


def best_match_cer(truth_lines, pred_lines):
    """
    Per-truth-line CER against its best-matching prediction, clamped to [0, 1].
    Diagnostic for line-level alignment only; never a headline number.
    """
    if not truth_lines or not pred_lines:
        return None
    total = 0.0
    for t in truth_lines:
        best, best_ratio = None, -1.0
        for p in pred_lines:
            ratio = SequenceMatcher(None, t.lower(), p.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, p
        total += min(1.0, jiwer.cer(t, best)) if t.strip() else 0.0
    return total / len(truth_lines)


# --------------------------------------------------------------------------
# Locating and scoring files
# --------------------------------------------------------------------------

def available_images(results_dir):
    """Image numbers an arm produced non-empty output for."""
    results_dir = str(results_dir)
    if not os.path.isdir(results_dir):
        return set()
    nums = set()
    for f in os.listdir(results_dir):
        if f.startswith(OUTPUT_PREFIX) and f.endswith(".txt"):
            digits = re.findall(r'\d+', f)
            if digits and os.path.getsize(os.path.join(results_dir, f)) > 0:
                nums.add(int(digits[-1]))
    return nums


def with_ground_truth(image_numbers, gt_dir=GROUND_TRUTH_DIR):
    """Filter to the images that actually have a reference transcription."""
    return {n for n in image_numbers if os.path.exists(ground_truth_path(gt_dir, n))}


def score_image(results_dir, image_number, gt_dir=GROUND_TRUTH_DIR):
    """
    Score one image for one arm.

    Returns (raw counts, normalized counts, drug recall, dose recall,
    truth lines, predicted lines), or None if the pair cannot be scored.
    """
    truth = load_lines(ground_truth_path(gt_dir, image_number))
    pred = load_lines(output_path(results_dir, image_number))

    truth_text, pred_text = " ".join(truth), " ".join(pred)
    raw = counts(truth_text, pred_text)
    norm = counts(normalize_for_scoring(truth_text), normalize_for_scoring(pred_text))
    if raw is None or norm is None:
        return None

    return {
        'raw': raw,
        'norm': norm,
        'drug': recall(extract_drugs(truth), extract_drugs(pred)),
        'dose': counter_recall(extract_doses(truth), extract_doses(pred)),
        'truth': truth,
        'pred': pred,
    }


def score_arm(results_dir, image_numbers, gt_dir=GROUND_TRUTH_DIR):
    """Corpus totals for one arm over a fixed set of image numbers."""
    agg = {k: 0 for k in COUNT_KEYS}
    agg_norm = {k: 0 for k in COUNT_KEYS}
    drug_recalls, dose_recalls = [], []

    for n in sorted(image_numbers):
        result = score_image(results_dir, n, gt_dir=gt_dir)
        if result is None:
            continue
        for k in COUNT_KEYS:
            agg[k] += result['raw'][k]
            agg_norm[k] += result['norm'][k]
        if result['drug'] is not None:
            drug_recalls.append(result['drug'])
        if result['dose'] is not None:
            dose_recalls.append(result['dose'])

    return {
        'n': len(image_numbers),
        'CER': rate(agg['c_err'], agg['c_ref']),
        'WER': rate(agg['w_err'], agg['w_ref']),
        'nCER': rate(agg_norm['c_err'], agg_norm['c_ref']),
        'nWER': rate(agg_norm['w_err'], agg_norm['w_ref']),
        'drug': sum(drug_recalls) / len(drug_recalls) if drug_recalls else None,
        'dose': sum(dose_recalls) / len(dose_recalls) if dose_recalls else None,
    }
