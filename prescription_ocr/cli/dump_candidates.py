"""
Dump per-box candidates from BOTH recognisers, once.

    python -m prescription_ocr.cli.dump_candidates

For every detected box on every image it records:
    paddle text + confidence   (PP-OCR recogniser — strong on printed text)
    trocr  text                (trocr-large-handwritten — strong on handwriting)

Written to results/candidates/image<N>.json.

Why dump instead of deciding inline: the interesting question is *which*
recogniser to trust per box, and the answer is a threshold to be tuned. Dumping
once lets `build_hybrid` sweep that threshold offline in seconds instead of
re-running the GPU pass for every candidate value.

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

from prescription_ocr.config import CANDIDATES_DIR, IMAGES_DIR, MAX_NEW_TOKENS, TROCR_MODEL
from prescription_ocr.io_utils import extract_image_number, list_images
from prescription_ocr.ocr.models import DEVICE
from prescription_ocr.ocr.recognizer import box_bounds, detect_boxes


def paddle_rec_result(recognizer, crop_bgr):
    """(text, confidence) from the PP-OCR recogniser for one crop."""
    out = recognizer.predict(crop_bgr)
    if not out:
        return "", 0.0
    r = out[0]
    return r.get("rec_text", ""), float(r.get("rec_score", 0.0))


@torch.inference_mode()
def main():
    out_dir = str(CANDIDATES_DIR)
    os.makedirs(out_dir, exist_ok=True)
    images = list_images(IMAGES_DIR)

    print("[LOAD] detection + both recognisers...")
    t0 = time.time()
    detector = TextDetection()
    paddle_rec = TextRecognition()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    model.to(DEVICE).eval()
    print(f"       loaded in {time.time() - t0:.0f}s (device: {DEVICE})\n")

    t_all = time.time()
    for i, path in enumerate(images, 1):
        n = extract_image_number(path)
        out_path = os.path.join(out_dir, f"image{n}.json")
        if os.path.exists(out_path):
            print(f"[{i:2d}/{len(images)}] image{n} -> skip (exists)")
            continue

        t0 = time.time()
        img = cv2.imread(path)
        boxes = detect_boxes(detector, path)

        records = []
        for box in boxes:
            x1, y1, x2, y2 = box_bounds(box)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            paddle_text, paddle_score = paddle_rec_result(paddle_rec, crop)

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(DEVICE)
            trocr_text = processor.batch_decode(
                model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS, num_beams=1),
                skip_special_tokens=True)[0]

            records.append({"box": [x1, y1, x2, y2],
                            "paddle": paddle_text, "paddle_score": paddle_score,
                            "trocr": trocr_text})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        print(f"[{i:2d}/{len(images)}] image{n} -> {len(records):3d} boxes "
              f"({time.time() - t0:.0f}s)")

    print(f"\nDone in {time.time() - t_all:.0f}s. Output in {out_dir}/")


if __name__ == "__main__":
    main()
