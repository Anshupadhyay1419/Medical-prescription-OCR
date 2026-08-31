"""
Build the hybrid recogniser output from the dumped candidates.

    python build_hybrid.py [--thr 0.80]

Per detected box, take the PP-OCR reading when its confidence clears the
threshold, otherwise fall back to TrOCR. PP-OCR is far stronger on the printed
letterhead/label text that trocr-large-handwritten was never trained for, and
its confidence is a good proxy for "this box is the kind of text I handle well".

Threshold 0.80 was chosen by sweeping candidates/ offline against gts/ — see the
sweep in the session notes. It is a flat minimum across CER/WER and the best
point for dose recall, which is the metric that matters medically.

No LLM and no VLM in this path: detection + two allowed recognisers + a
deterministic rule.
"""
import argparse, json, os
from postprocess import clean_ocr_output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.80)
    ap.add_argument("--cand-dir", default="candidates")
    ap.add_argument("--out-dir", default="outputs_hybrid")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(os.listdir(a.cand_dir), key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
    n_pad = n_tr = 0
    for fn in files:
        if not fn.endswith(".json"):
            continue
        num = "".join(filter(str.isdigit, fn))
        recs = json.load(open(os.path.join(a.cand_dir, fn), encoding="utf-8"))
        lines = []
        for r in recs:
            use_paddle = r["paddle_score"] >= a.thr
            s = r["paddle"] if use_paddle else r["trocr"]
            n_pad += use_paddle
            n_tr += not use_paddle
            if s.strip():
                lines.append(s)
        out = os.path.join(a.out_dir, f"final_clean{num}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_ocr_output(lines)) + "\n")
    total = n_pad + n_tr
    print(f"thr={a.thr}: {len(files)} images, {total} boxes -> "
          f"PP-OCR {n_pad} ({n_pad/total:.0%}), TrOCR {n_tr} ({n_tr/total:.0%})")
    print(f"written to {a.out_dir}/")

if __name__ == "__main__":
    main()
