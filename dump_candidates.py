"""
Dump per-box recognition candidates from BOTH recognisers, once.

    python dump_candidates.py

For every detected box on every image it records:
    paddle text + confidence   (PP-OCR recogniser — strong on printed text)
    trocr  text                (trocr-large-handwritten — strong on handwriting)

Written to candidates/image<N>.json.

Why dump instead of deciding inline: the interesting question is *which*
recogniser to trust per box, and the answer is a threshold to be tuned. Dumping
once lets the threshold be swept offline in seconds instead of re-running the
GPU pass for every candidate value.

No VLM anywhere — PP-OCR and TrOCR are both allowed under the project constraint.
"""
import json
import os
import time

import cv2
import torch
from PIL import Image
from paddleocr import TextDetection, TextRecognition
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from trocr_llm import list_images, extract_image_number, sort_reading_order

OUT_DIR = "candidates"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def paddle_rec_result(rec, crop_bgr):
    """Return (text, score) from the PP-OCR recogniser for one crop."""
    out = rec.predict(crop_bgr)
    if not out:
        return "", 0.0
    r = out[0]
    return r.get("rec_text", ""), float(r.get("rec_score", 0.0))


@torch.inference_mode()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    images = list_images("image")

    print("[LOAD] detection + both recognisers...")
    t0 = time.time()
    det = TextDetection()
    rec = TextRecognition()
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    model.to(DEVICE).eval()
    print(f"       loaded in {time.time() - t0:.0f}s (device: {DEVICE})\n")

    t_all = time.time()
    for i, path in enumerate(images, 1):
        n = extract_image_number(path)
        out_path = os.path.join(OUT_DIR, f"image{n}.json")
        if os.path.exists(out_path):
            print(f"[{i:2d}/{len(images)}] image{n} -> skip (exists)")
            continue

        t0 = time.time()
        img = cv2.imread(path)
        boxes = sort_reading_order(det.predict(path)[0]['dt_polys'])

        records = []
        for box in boxes:
            pts = [(int(x), int(y)) for x, y in box]
            x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
            x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            p_text, p_score = paddle_rec_result(rec, crop)

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pv = processor(images=pil, return_tensors="pt").pixel_values.to(DEVICE)
            t_text = processor.batch_decode(
                model.generate(pv, max_new_tokens=64, num_beams=1),
                skip_special_tokens=True)[0]

            records.append({"box": [x1, y1, x2, y2],
                            "paddle": p_text, "paddle_score": p_score,
                            "trocr": t_text})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        print(f"[{i:2d}/{len(images)}] image{n} -> {len(records):3d} boxes ({time.time()-t0:.0f}s)")

    print(f"\nDone in {time.time() - t_all:.0f}s. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
