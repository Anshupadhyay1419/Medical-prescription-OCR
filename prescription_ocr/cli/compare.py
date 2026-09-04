"""
Side-by-side comparison of every recogniser arm.

    python -m prescription_ocr.cli.compare

Arms are scored only on the images they ALL produced output for, so the numbers
in a row are always comparable. Edit COMPARISON_ARMS in config.py to add or
remove an arm.
"""
import os

from prescription_ocr.config import (
    COMPARISON_ARMS, NON_PRESCRIPTION_IMAGES, PROJECT_ROOT,
)
from prescription_ocr.evaluation.metrics import (
    available_images, score_arm, with_ground_truth,
)


def fmt(value, decimals=3):
    return f"{value:.{decimals}f}" if isinstance(value, float) else "  -  "


def table(title, arms, image_numbers):
    print(f"\n{title}  (n = {len(image_numbers)} images)")
    print(f"| {'Arm':<14} | {'CER':>6} | {'WER':>6} | {'nCER':>6} | {'nWER':>6} "
          f"| {'drug rec':>8} | {'dose rec':>8} |")
    print(f"|{'-' * 16}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 10}|")
    for name, results_dir in arms:
        s = score_arm(results_dir, image_numbers)
        print(f"| {name:<14} | {fmt(s['CER']):>6} | {fmt(s['WER']):>6} "
              f"| {fmt(s['nCER']):>6} | {fmt(s['nWER']):>6} "
              f"| {fmt(s['drug']):>8} | {fmt(s['dose']):>8} |")


def main():
    present = {name: with_ground_truth(available_images(d)) for name, d in COMPARISON_ARMS}

    print("=" * 78)
    print("RECOGNISER COMPARISON")
    print("=" * 78)
    for name, results_dir in COMPARISON_ARMS:
        shown = os.path.relpath(results_dir, PROJECT_ROOT)
        print(f"  {name:<16} {shown:<24} {len(present[name]):>3} scoreable outputs")

    missing = [name for name, _ in COMPARISON_ARMS if not present[name]]
    if missing:
        print(f"\n  MISSING ENTIRELY: {', '.join(missing)} — run those arms first.")
        return

    common = set.intersection(*present.values())
    for name, _ in COMPARISON_ARMS:
        extra = sorted(present[name] - common)
        if extra:
            print(f"  note: {name} also has image{extra} which another arm lacks "
                  f"— excluded to keep arms comparable")
    print(f"\n  Scored on the {len(common)} images all arms produced.")

    prescriptions = sorted(common - NON_PRESCRIPTION_IMAGES)
    narrative = sorted(common & NON_PRESCRIPTION_IMAGES)

    if prescriptions:
        table("PRESCRIPTIONS (headline)", COMPARISON_ARMS, prescriptions)
    if narrative:
        table("CLINICAL NARRATIVE image11-15 (different task, reported separately)",
              COMPARISON_ARMS, narrative)

    print("\nLower is better for CER/WER; higher is better for recall.")


if __name__ == "__main__":
    main()
