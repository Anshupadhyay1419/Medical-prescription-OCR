"""
Per-image evaluation table for one arm.

    python -m prescription_ocr.cli.evaluate                    # results/trocr_llm
    python -m prescription_ocr.cli.evaluate results/trocr_raw  # any other arm
"""
import argparse

from prescription_ocr.config import GROUND_TRUTH_DIR, TROCR_LLM_DIR
from prescription_ocr.evaluation.metrics import (
    COUNT_KEYS, available_images, best_match_cer, rate, score_image, with_ground_truth,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(TROCR_LLM_DIR),
                   help="directory of final_clean<N>.txt files to score")
    p.add_argument("--gt-dir", default=str(GROUND_TRUTH_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    image_numbers = sorted(with_ground_truth(available_images(args.results_dir),
                                             gt_dir=args.gt_dir))
    if not image_numbers:
        raise FileNotFoundError(
            f"No scoreable final_clean*.txt files in {args.results_dir} "
            f"with a matching reference in {args.gt_dir}")

    agg = {k: 0 for k in COUNT_KEYS}
    agg_norm = {k: 0 for k in COUNT_KEYS}
    rows, drug_recalls, dose_recalls, align_cers = [], [], [], []

    print("=" * 88)
    print(f"EVALUATION — {args.results_dir} vs {args.gt_dir}")
    print("=" * 88)
    print(f"{'file':>5} {'GT ln':>6} {'OCR ln':>7} "
          f"{'CER':>7} {'WER':>7} {'nCER':>7} {'nWER':>7} {'drug':>6} {'dose':>6}")
    print("-" * 88)

    for n in image_numbers:
        result = score_image(args.results_dir, n, gt_dir=args.gt_dir)
        if result is None:
            continue

        raw, norm = result['raw'], result['norm']
        for k in COUNT_KEYS:
            agg[k] += raw[k]
            agg_norm[k] += norm[k]

        if result['drug'] is not None:
            drug_recalls.append(result['drug'])
        if result['dose'] is not None:
            dose_recalls.append(result['dose'])
        align = best_match_cer(result['truth'], result['pred'])
        if align is not None:
            align_cers.append(align)

        rows.append((n, rate(norm['c_err'], norm['c_ref'])))
        drug = f"{result['drug']:.2f}" if result['drug'] is not None else '  -  '
        dose = f"{result['dose']:.2f}" if result['dose'] is not None else '  -  '
        print(f"{n:>5} {len(result['truth']):>6} {len(result['pred']):>7} "
              f"{rate(raw['c_err'], raw['c_ref']):>7.3f} {rate(raw['w_err'], raw['w_ref']):>7.3f} "
              f"{rate(norm['c_err'], norm['c_ref']):>7.3f} {rate(norm['w_err'], norm['w_ref']):>7.3f} "
              f"{drug:>6} {dose:>6}")

    if not rows:
        raise RuntimeError(f"No matching references found in {args.gt_dir}")

    print("=" * 88)
    print(f"Files evaluated: {len(rows)}\n")
    print("CORPUS TOTALS  (total edits / total reference length)")
    print(f"  raw         CER {rate(agg['c_err'], agg['c_ref']):.3f}    "
          f"WER {rate(agg['w_err'], agg['w_ref']):.3f}")
    print(f"  normalized  CER {rate(agg_norm['c_err'], agg_norm['c_ref']):.3f}    "
          f"WER {rate(agg_norm['w_err'], agg_norm['w_ref']):.3f}\n")

    if drug_recalls:
        print(f"  drug-name recall   {sum(drug_recalls) / len(drug_recalls):.3f}")
    if dose_recalls:
        print(f"  dose-code recall   {sum(dose_recalls) / len(dose_recalls):.3f}")
    if align_cers:
        print(f"  line-align CER     {sum(align_cers) / len(align_cers):.3f}  "
              f"(diagnostic, clamped)")

    worst = sorted(rows, key=lambda r: -r[1])[:5]
    print("\n  worst files (normalized CER): "
          + ", ".join(f"image{i} {v:.2f}" for i, v in worst))


if __name__ == "__main__":
    main()
