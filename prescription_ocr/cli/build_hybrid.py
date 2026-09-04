"""
Build the hybrid arm from the dumped candidates.

    python -m prescription_ocr.cli.build_hybrid              # default threshold
    python -m prescription_ocr.cli.build_hybrid --thr 0.9    # sweep it

For each detected box, take PP-OCR's reading when it is confident and fall back
to TrOCR otherwise. Printed header text is PP-OCR's strength; handwriting is
TrOCR's. Because the candidates were dumped once, sweeping the threshold costs
seconds instead of a full GPU pass.
"""
import argparse
import json
import os

from prescription_ocr.config import CANDIDATES_DIR, HYBRID_DIR, HYBRID_PADDLE_THRESHOLD
from prescription_ocr.io_utils import extract_image_number, natural_sort_key, output_path
from prescription_ocr.postprocess import clean_ocr_output


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thr", type=float, default=HYBRID_PADDLE_THRESHOLD,
                   help="PP-OCR confidence at or above which PP-OCR wins the box")
    p.add_argument("--cand-dir", default=str(CANDIDATES_DIR))
    p.add_argument("--out-dir", default=str(HYBRID_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted((f for f in os.listdir(args.cand_dir) if f.endswith(".json")),
                   key=natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No candidate dumps in {args.cand_dir} — "
            f"run `python -m prescription_ocr.cli.dump_candidates` first")

    n_paddle = n_trocr = 0
    for filename in files:
        number = extract_image_number(filename)
        with open(os.path.join(args.cand_dir, filename), encoding="utf-8") as f:
            records = json.load(f)

        lines = []
        for record in records:
            use_paddle = record["paddle_score"] >= args.thr
            text = record["paddle"] if use_paddle else record["trocr"]
            n_paddle += use_paddle
            n_trocr += not use_paddle
            if text.strip():
                lines.append(text)

        out = output_path(args.out_dir, number)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_ocr_output(lines)) + "\n")

    total = n_paddle + n_trocr
    print(f"thr={args.thr}: {len(files)} images, {total} boxes -> "
          f"PP-OCR {n_paddle} ({n_paddle / total:.0%}), TrOCR {n_trocr} ({n_trocr / total:.0%})")
    print(f"written to {args.out_dir}/")


if __name__ == "__main__":
    main()
