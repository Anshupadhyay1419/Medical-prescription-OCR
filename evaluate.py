"""
Evaluate OCR output against the per-image ground truths in gts/.

Usage:
    python evaluate.py [outputs_dir]        # default: outputs

Reads <outputs_dir>/final_clean<N>.txt and compares against gts/image<N>.txt.

Metrics
-------
Document-level CER/WER (the headline numbers): the whole page is joined into a
single string and compared. This is the standard metric for full-page OCR and
is insensitive to how the text happened to be split into lines.

  raw         - exact comparison, nothing normalised.
  normalized  - lowercased, whitespace/punctuation normalised, unit joins split
                (5ml -> 5 ml). Cosmetic differences stop counting as recognition
                errors, so this isolates genuine misreads.

Corpus totals are aggregated as (total edits / total reference length) rather
than by averaging per-file rates, so one short file cannot dominate the result.

Clinical metrics (drug recall / dose recall) matter more than CER for a
prescription system: getting the drug name right is the point.

NOTE: an earlier version of this script reported an average of a per-line
"best match" CER. That statistic divides an edit distance by the length of a
possibly very short reference line, so a single collapsed-paragraph prediction
could score a CER of 29. Those numbers were measurement artefacts, not OCR
quality. It is kept below only as a diagnostic, clamped to [0, 1].
"""
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

import jiwer

from postprocess import normalize_for_scoring

GTS_DIR = "gts"
DEFAULT_OUTPUT_DIR = "outputs"


def load_lines(path):
    """Load lines from file, stripping a 'Line N:' prefix if present."""
    with open(path, encoding="utf-8") as f:
        out = []
        for l in f:
            l = l.strip()
            if not l:
                continue
            out.append(re.sub(r'^Line\s*\d+\s*:\s*', '', l, flags=re.IGNORECASE))
        return out


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


# ---------------------------------------------------------------
# Clinical metrics
# ---------------------------------------------------------------

DRUG_RE = re.compile(r'\b(?:Tab|Cap|Syp|Inj|Inhaler|Susp|T)\.?\s+([A-Za-z][\w\-]{2,})',
                     re.IGNORECASE)
DOSE_RE = re.compile(r'\b\d-\d-\d\b')


def extract_drugs(lines):
    return {m.group(1).lower().strip('.,') for l in lines for m in DRUG_RE.finditer(l)}


def extract_doses(lines):
    return Counter(d for l in lines for d in DOSE_RE.findall(l))


def recall(truth_set, pred_set):
    if not truth_set:
        return None
    return len(truth_set & pred_set) / len(truth_set)


def counter_recall(truth_c, pred_c):
    total = sum(truth_c.values())
    if not total:
        return None
    return sum((truth_c & pred_c).values()) / total


# ---------------------------------------------------------------
# Diagnostic only — clamped so it cannot exceed 1.0
# ---------------------------------------------------------------

def best_match_cer(truth_lines, pred_lines):
    """
    Per-truth-line CER against its best-matching prediction line, clamped to
    [0,1]. Diagnostic for line-level alignment quality; not a headline metric.
    """
    if not truth_lines or not pred_lines:
        return None
    total = 0.0
    for t in truth_lines:
        best, best_r = None, -1.0
        for p in pred_lines:
            r = SequenceMatcher(None, t.lower(), p.lower()).ratio()
            if r > best_r:
                best_r, best = r, p
        total += min(1.0, jiwer.cer(t, best)) if t.strip() else 0.0
    return total / len(truth_lines)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR

    files = sorted(
        (f for f in os.listdir(output_dir)
         if f.startswith("final_clean") and f.endswith(".txt")),
        key=lambda f: int(re.findall(r'\d+', f)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No final_clean*.txt files found in {output_dir}")

    agg = {k: 0 for k in ('c_err', 'c_ref', 'w_err', 'w_ref')}
    agg_n = {k: 0 for k in ('c_err', 'c_ref', 'w_err', 'w_ref')}
    rows, drug_recalls, dose_recalls, bm_cers = [], [], [], []

    print("=" * 88)
    print(f"EVALUATION — {output_dir}/ vs {GTS_DIR}/")
    print("=" * 88)
    print(f"{'file':>5} {'GT ln':>6} {'OCR ln':>7} "
          f"{'CER':>7} {'WER':>7} {'nCER':>7} {'nWER':>7} {'drug':>6} {'dose':>6}")
    print("-" * 88)

    for fn in files:
        idx = re.findall(r'\d+', fn)[-1]
        truth_path = os.path.join(GTS_DIR, f"image{idx}.txt")
        if not os.path.exists(truth_path):
            print(f"{idx:>5}  (no ground truth at {truth_path})")
            continue

        truth = load_lines(truth_path)
        pred = load_lines(os.path.join(output_dir, fn))

        t_raw, p_raw = " ".join(truth), " ".join(pred)
        c = counts(t_raw, p_raw)
        cn = counts(normalize_for_scoring(t_raw), normalize_for_scoring(p_raw))
        if c is None or cn is None:
            continue

        for k in agg:
            agg[k] += c[k]
            agg_n[k] += cn[k]

        dr = recall(extract_drugs(truth), extract_drugs(pred))
        dsr = counter_recall(extract_doses(truth), extract_doses(pred))
        bm = best_match_cer(truth, pred)
        if dr is not None:
            drug_recalls.append(dr)
        if dsr is not None:
            dose_recalls.append(dsr)
        if bm is not None:
            bm_cers.append(bm)

        rows.append((idx, rate(cn['c_err'], cn['c_ref'])))
        print(f"{idx:>5} {len(truth):>6} {len(pred):>7} "
              f"{rate(c['c_err'], c['c_ref']):>7.3f} {rate(c['w_err'], c['w_ref']):>7.3f} "
              f"{rate(cn['c_err'], cn['c_ref']):>7.3f} {rate(cn['w_err'], cn['w_ref']):>7.3f} "
              f"{(f'{dr:.2f}' if dr is not None else '  -  '):>6} "
              f"{(f'{dsr:.2f}' if dsr is not None else '  -  '):>6}")

    if not rows:
        raise RuntimeError(f"No matching ground truths found in {GTS_DIR}")

    print("=" * 88)
    print(f"Files evaluated: {len(rows)}")
    print()
    print("CORPUS TOTALS  (total edits / total reference length)")
    print(f"  raw         CER {rate(agg['c_err'], agg['c_ref']):.3f}    "
          f"WER {rate(agg['w_err'], agg['w_ref']):.3f}")
    print(f"  normalized  CER {rate(agg_n['c_err'], agg_n['c_ref']):.3f}    "
          f"WER {rate(agg_n['w_err'], agg_n['w_ref']):.3f}")
    print()
    if drug_recalls:
        print(f"  drug-name recall   {sum(drug_recalls)/len(drug_recalls):.3f}")
    if dose_recalls:
        print(f"  dose-code recall   {sum(dose_recalls)/len(dose_recalls):.3f}")
    if bm_cers:
        print(f"  line-align CER     {sum(bm_cers)/len(bm_cers):.3f}  (diagnostic, clamped)")

    worst = sorted(rows, key=lambda r: -r[1])[:5]
    print(f"\n  worst files (normalized CER): "
          + ", ".join(f"image{i} {v:.2f}" for i, v in worst))


if __name__ == "__main__":
    main()
