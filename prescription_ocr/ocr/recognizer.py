"""
Detection and recognition: image in, per-box text plus coordinates out.

The coordinates travel with the text because the restructuring stage needs them
to work out which fragments belong on the same visual row.
"""
import time

import cv2
import torch
from PIL import Image

from prescription_ocr.config import MAX_NEW_TOKENS, NUM_BEAMS, USE_RERANKER
from prescription_ocr.llm.reranker import rerank_candidates
from prescription_ocr.ocr.models import DEVICE
from prescription_ocr.ocr.reading_order import sort_reading_order


def detect_boxes(detector, img_path):
    """Detected text polygons for one image, already in reading order."""
    result = detector.predict(str(img_path))
    return sort_reading_order(result[0]['dt_polys'])


def box_bounds(box):
    """Polygon -> integer (x1, y1, x2, y2) bounding box."""
    pts = [(int(x), int(y)) for x, y in box]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def read_crop(processor, model, crop_bgr, use_reranker=USE_RERANKER):
    """
    Transcribe one cropped text line.

    Greedy decoding is the default: on handwriting, beam search produces fluent
    text that drifts from what is actually written. Beams are only used when the
    reranker is enabled to choose between them.
    """
    pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(DEVICE)

    if use_reranker:
        outputs = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return rerank_candidates(processor.batch_decode(outputs, skip_special_tokens=True))

    outputs = model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS, num_beams=1)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


@torch.inference_mode()
def recognize(detector, processor, model, img_path,
              use_reranker=USE_RERANKER, verbose=True):
    """Detect, sort and transcribe. Returns [(text, x1, y1, x2, y2), ...]."""
    print("[1/5] Detecting text lines...")
    t0 = time.time()
    boxes = detect_boxes(detector, img_path)
    print(f"      Detected and sorted {len(boxes)} boxes in {time.time() - t0:.0f}s\n")

    img = cv2.imread(str(img_path))

    label = "TrOCR + Point A reranker" if use_reranker else "TrOCR (greedy)"
    print(f"[2/5] Recognition: {label}")
    t0 = time.time()

    lines_with_boxes = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box_bounds(box)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text = read_crop(processor, model, crop, use_reranker=use_reranker)
        lines_with_boxes.append((text, x1, y1, x2, y2))

        if verbose:
            print(f"      Line {i:2d}: [{x1:4d},{y1:4d}] {text}")

    print(f"      Recognition done in {time.time() - t0:.0f}s\n")
    return lines_with_boxes
