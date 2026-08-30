"""
Complete evaluation script.
Compares raw TrOCR output vs final pipeline output against ground truth.
"""
from jiwer import cer, wer
from difflib import SequenceMatcher
from collections import Counter
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

TRUTH_FILE = "prescription_02_ground_truth.txt"
RAW_FILE = "outputs/output_1_raw.txt"
FINAL_FILE = "outputs/output_final.txt"

# load all
truth = load_truth(TRUTH_FILE)
raw = load_lines(RAW_FILE)
final = load_lines(FINAL_FILE)

# clean raw (has coordinates prefix like "[123,456] text")
raw = [re.sub(r'^\[\d+,\d+\]\s*', '', l) for l in raw]

# compute metrics
raw_metrics = compute_metrics(truth, raw)
final_metrics = compute_metrics(truth, final)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"\nLine counts:")
print(f"  Ground truth: {len(truth)}")
print(f"  Raw TrOCR:    {len(raw)}  (diff: {abs(len(raw)-len(truth))})")
print(f"  Final:        {len(final)}  (diff: {abs(len(final)-len(truth))})")

print(f"\n{'='*70}")
print("CHARACTER ERROR RATE (CER)")
print("=" * 70)
print(f"{'Metric':<30} {'Raw':<12} {'Final':<12} {'Improvement':<15}")
print("-" * 70)

def print_row(label, raw_val, final_val, lower_better=True):
    if lower_better:
        improvement = (raw_val - final_val) * 100
        marker = "✓" if final_val < raw_val else "✗"
    else:
        improvement = (final_val - raw_val) * 100
        marker = "✓" if final_val > raw_val else "✗"
    print(f"{label:<30} {raw_val:<12.3f} {final_val:<12.3f} {improvement:+.1f}% {marker}")

print_row("CER (joined text)", raw_metrics['cer_joined'], final_metrics['cer_joined'])
print_row("CER (best-match)", raw_metrics['best_match_cer'], final_metrics['best_match_cer'])

print(f"\n{'='*70}")
print("WORD ERROR RATE (WER)")
print("=" * 70)
print(f"{'Metric':<30} {'Raw':<12} {'Final':<12} {'Improvement':<15}")
print("-" * 70)

print_row("WER (joined text)", raw_metrics['wer_joined'], final_metrics['wer_joined'])
print_row("WER (best-match)", raw_metrics['best_match_wer'], final_metrics['best_match_wer'])

print(f"\n{'='*70}")
print("RECALL METRICS (higher = better)")
print("=" * 70)
print(f"{'Metric':<30} {'Raw':<12} {'Final':<12} {'Change':<15}")
print("-" * 70)

print_row("Character recall", raw_metrics['char_recall'], final_metrics['char_recall'], lower_better=False)

print(f"\n{'='*70}")
print("MEDICAL FIELD EXTRACTION")
print("=" * 70)

truth_drugs = extract_drugs(truth)
raw_drugs = extract_drugs(raw)
final_drugs = extract_drugs(final)

print(f"\nDrug names (truth has {len(truth_drugs)}):")
print(f"  Truth drugs:  {sorted(truth_drugs)}")
print(f"  Raw found:    {len(raw_drugs & truth_drugs)}/{len(truth_drugs)} correct, "
      f"{len(raw_drugs - truth_drugs)} spurious")
print(f"  Final found:  {len(final_drugs & truth_drugs)}/{len(truth_drugs)} correct, "
      f"{len(final_drugs - truth_drugs)} spurious")

if final_drugs - truth_drugs:
    print(f"  Spurious in final: {sorted(final_drugs - truth_drugs)}")
if truth_drugs - final_drugs:
    print(f"  Missing from final: {sorted(truth_drugs - final_drugs)}")

truth_doses = Counter(extract_dosages(truth))
raw_doses = Counter(extract_dosages(raw))
final_doses = Counter(extract_dosages(final))

print(f"\nDose schedules (like X-X-X):")
print(f"  Truth doses: {sorted(truth_doses.items())}")
print(f"  Raw doses:   {sorted(raw_doses.items())}")
print(f"  Final doses: {sorted(final_doses.items())}")

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)

improvements = [
    ("CER (best-match)", raw_metrics['best_match_cer'], final_metrics['best_match_cer']),
    ("WER (best-match)", raw_metrics['best_match_wer'], final_metrics['best_match_wer']),
    ("Line count accuracy", abs(len(raw)-len(truth)), abs(len(final)-len(truth))),
    ("Drug recall", len(raw_drugs & truth_drugs), len(final_drugs & truth_drugs)),
]

print()
for label, raw_v, final_v in improvements:
    if isinstance(raw_v, float):
        change = f"{(raw_v - final_v) * 100:+.1f}% absolute"
    else:
        change = f"{raw_v} → {final_v}"
    print(f"  {label:<25} {change}")

print(f"\n  Final pipeline improved best-match CER by "
      f"{(raw_metrics['best_match_cer'] - final_metrics['best_match_cer']) * 100:.1f}%")