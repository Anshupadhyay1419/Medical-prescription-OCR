import os
import re
import sys

from evaluate import (
    GTS_DIR, load_lines, counts, rate,
    extract_drugs, extract_doses, recall, counter_recall,
)

ARMS = [
    ("Raw TrOCR",      "outputs_raw"),
    ("TrOCR + LLM",    "outputs"),
    ("Hybrid PP+TrOCR", "outputs_hybrid"),
    ("Qwen2.5-VL",     "outputs_vl"),
]

# Not prescriptions — clinical narrative text, different task.
NON_PRESCRIPTION = {11, 12, 13, 14, 15}


def available(output_dir):
    """Image numbers this arm has produced output for."""
    if not os.path.isdir(output_dir):
        return set()
    nums = set()
    for f in os.listdir(output_dir):
        if f.startswith("final_clean") and f.endswith(".txt"):
            m = re.findall(r'\d+', f)
            if m and os.path.getsize(os.path.join(output_dir, f)) > 0:
                nums.add(int(m[-1]))
    return nums


def with_ground_truth(nums):
    return {n for n in nums if os.path.exists(os.path.join(GTS_DIR, f"image{n}.txt"))}


def score(output_dir, image_nums):
    """Corpus totals for one arm over a fixed set of image numbers."""
    agg = {k: 0 for k in ('c_err', 'c_ref', 'w_err', 'w_ref')}
    agg_n = dict(agg)
    drug_recalls, dose_recalls = [], []

    from postprocess import normalize_for_scoring

    for n in sorted(image_nums):
        truth = load_lines(os.path.join(GTS_DIR, f"image{n}.txt"))
        pred = load_lines(os.path.join(output_dir, f"final_clean{n}.txt"))

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
        if dr is not None:
            drug_recalls.append(dr)
        if dsr is not None:
            dose_recalls.append(dsr)

    return {
        'n': len(image_nums),
        'CER': rate(agg['c_err'], agg['c_ref']),
        'WER': rate(agg['w_err'], agg['w_ref']),
        'nCER': rate(agg_n['c_err'], agg_n['c_ref']),
        'nWER': rate(agg_n['w_err'], agg_n['w_ref']),
        'drug': sum(drug_recalls) / len(drug_recalls) if drug_recalls else None,
        'dose': sum(dose_recalls) / len(dose_recalls) if dose_recalls else None,
    }


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, float) else "  -  "


def table(title, arms, image_nums):
    print(f"\n{title}  (n = {len(image_nums)} images)")
    print(f"| {'Arm':<14} | {'CER':>6} | {'WER':>6} | {'nCER':>6} | {'nWER':>6} "
          f"| {'drug rec':>8} | {'dose rec':>8} |")
    print(f"|{'-'*16}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*10}|{'-'*10}|")
    for name, d in arms:
        s = score(d, image_nums)
        print(f"| {name:<14} | {fmt(s['CER']):>6} | {fmt(s['WER']):>6} "
              f"| {fmt(s['nCER']):>6} | {fmt(s['nWER']):>6} "
              f"| {fmt(s['drug'], 3):>8} | {fmt(s['dose'], 3):>8} |")


def main():
    present = {name: with_ground_truth(available(d)) for name, d in ARMS}

    print("=" * 78)
    print("RECOGNISER COMPARISON")
    print("=" * 78)
    for (name, d) in ARMS:
        got = present[name]
        print(f"  {name:<14} {d:<14} {len(got):>3} scoreable outputs")

    missing = [n for n, _ in ARMS if not present[n]]
    if missing:
        print(f"\n  MISSING ENTIRELY: {', '.join(missing)} — run those arms first.")
        return

    common = set.intersection(*present.values())
    for name, _ in ARMS:
        gap = sorted(common.symmetric_difference(present[name]) & present[name])
        if gap:
            print(f"  note: {name} also has image{gap} which another arm lacks "
                  f"— excluded to keep arms comparable")
    print(f"\n  Scored on the {len(common)} images all arms produced.")

    scripts = sorted(common - NON_PRESCRIPTION)
    narrative = sorted(common & NON_PRESCRIPTION)

    if scripts:
        table("PRESCRIPTIONS (headline)", ARMS, scripts)
    if narrative:
        table("CLINICAL NARRATIVE image11-15 (different task, reported separately)",
              ARMS, narrative)

    print("\nLower is better for CER/WER; higher is better for recall.")


if __name__ == "__main__":
    main()
