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
