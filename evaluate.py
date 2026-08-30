"""
Evaluate OCR output against per-image ground truths in the gts folder.
The script reads all files named final_clean*.txt from outputs and matches
against gts/<image_number>.txt for the same index.
"""
from jiwer import cer, wer
from difflib import SequenceMatcher
from collections import Counter
import os
import re


def load_lines(path):
    """Load lines from file, stripping 'Line N:' prefix if present."""
    with open(path) as f:
        return [
            l.strip().split(": ", 1)[-1] if ": " in l else l.strip()
            for l in f if l.strip()
        ]


def load_truth(path):
    """Load ground truth (no prefix stripping needed)."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def compute_metrics(truth_lines, pred_lines):
    """Compute both CER and WER, joined and best-match versions."""
    truth_text = " ".join(truth_lines)
    pred_text = " ".join(pred_lines)
    
    return {
        'cer_joined': cer(truth_text, pred_text),
        'wer_joined': wer(truth_text, pred_text),
        'char_recall': character_recall(truth_lines, pred_lines),
        'best_match_cer': best_match_metric(pred_lines, truth_lines, cer),
        'best_match_wer': best_match_metric(pred_lines, truth_lines, wer),
    }


def character_recall(truth_lines, pred_lines):
    """% of ground truth characters present anywhere in output."""
    truth_chars = Counter(c for l in truth_lines for c in l if c.isalnum())
    pred_chars = Counter(c for l in pred_lines for c in l if c.isalnum())
    overlap = sum((truth_chars & pred_chars).values())
    total = sum(truth_chars.values())
    return overlap / total if total > 0 else 0


def best_match_metric(pred_lines, truth_lines, metric_fn):
    """For each pred line, find best matching truth line, avg the metric."""
    total = 0
    count = 0
    for p in pred_lines:
        if not p.strip():
            continue
        best_ratio = 0
        best_truth = ""
        for t in truth_lines:
            r = SequenceMatcher(None, p.lower(), t.lower()).ratio()
            if r > best_ratio:
                best_ratio = r
                best_truth = t
        if best_truth:
            total += metric_fn(best_truth, p)
            count += 1
    return total / count if count > 0 else 0


def extract_drugs(lines):
    """Extract medication names."""
    drugs = set()
    pattern = re.compile(
        r'\b(?:Tab|Cap|Syp|Inj|Inhaler|Susp)\.?\s+([A-Za-z][\w\-]+)',
        re.IGNORECASE
    )
    for line in lines:
        for match in pattern.finditer(line):
            drugs.add(match.group(1).lower().strip('.,'))
    return drugs


def extract_dosages(lines):
    """Extract dose schedules like X-X-X."""
    doses = []
    pattern = re.compile(r'\d+-\d+-\d+')
    for line in lines:
        doses.extend(pattern.findall(line))
    return doses


# ============================================================
# MAIN
# ============================================================

OUTPUT_DIR = "outputs"
GTS_DIR = "gts"

final_files = sorted(
    os.path.join(OUTPUT_DIR, f)
    for f in os.listdir(OUTPUT_DIR)
    if f.startswith("final_clean") and f.endswith(".txt")
)

if not final_files:
    raise FileNotFoundError(f"No final_clean*.txt files found in {OUTPUT_DIR}")

all_final_metrics = []
print("=" * 70)
print("EVALUATION RESULTS FOR ALL FINAL_CLEAN FILES")
print("=" * 70)

for final_path in final_files:
    image_index = os.path.splitext(os.path.basename(final_path))[0].replace("final_clean", "")
    truth_path = os.path.join(GTS_DIR, f"image{image_index}.txt")

    if not os.path.exists(truth_path):
        print(f"Skipping {final_path}: matching truth not found at {truth_path}")
        continue

    truth = load_truth(truth_path)
    final = load_lines(final_path)

    final_metrics = compute_metrics(truth, final)
    all_final_metrics.append((final_path, final_metrics, len(truth), len(final)))

    print(f"\nFile: {final_path}")
    print(f"  Ground truth: {len(truth)} lines")
    print(f"  Final OCR:    {len(final)} lines")
    print(f"  CER (joined): {final_metrics['cer_joined']:.3f}")
    print(f"  WER (joined): {final_metrics['wer_joined']:.3f}")
    print(f"  Best-match CER: {final_metrics['best_match_cer']:.3f}")
    print(f"  Best-match WER: {final_metrics['best_match_wer']:.3f}")

if not all_final_metrics:
    raise RuntimeError(f"No matching ground truths found in {GTS_DIR} for final_clean files")

average_cer = sum(m['best_match_cer'] for _, m, _, _ in all_final_metrics) / len(all_final_metrics)
average_wer = sum(m['best_match_wer'] for _, m, _, _ in all_final_metrics) / len(all_final_metrics)

print(f"\n{'='*70}")
print("AVERAGE RESULTS")
print("=" * 70)
print(f"Files evaluated: {len(all_final_metrics)}")
print(f"Average best-match CER: {average_cer:.3f}")
print(f"Average best-match WER: {average_wer:.3f}")