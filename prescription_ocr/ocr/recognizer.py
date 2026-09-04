"""
Preprocessing, detection and recognition: image in, per-box text plus
coordinates out.

The scan is put through document image processing first, and everything
downstream — detection, box visualisation and the TrOCR crops — works on that
processed image. This matters because the DIP stage resizes and deskews: box
coordinates are in the processed image's space, not the original scan's, so
mixing the two would crop the wrong regions.

The coordinates travel with the text because the restructuring stage needs them
to work out which fragments belong on the same visual row.
"""
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from prescription_ocr.config import (
    MAX_NEW_TOKENS, NUM_BEAMS, PADDLE_CONFIDENCE_THRESHOLD, RECOGNIZER,
    USE_PREPROCESSING, USE_RERANKER,
)
from prescription_ocr.ocr.layout import order_items
from prescription_ocr.llm.reranker import rerank_candidates
from prescription_ocr.ocr.models import DEVICE
from prescription_ocr.ocr.reading_order import sort_reading_order
from prescription_ocr.ocr.visualize import draw
from prescription_ocr.preprocessing import preprocess_prescription


def load_image(img_path, use_preprocessing=USE_PREPROCESSING, preprocessed_path=None):
    """
    The image every later stage works on.

    With preprocessing on, that is the DIP output (optionally written to
    `preprocessed_path`); with it off, the untouched scan.
    """
    if use_preprocessing:
        return preprocess_prescription(img_path, output_path=preprocessed_path)

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot load {img_path}")
    return img


def detect_boxes(detector, image):
    """
    Detected text polygons, already in reading order.

    `image` may be a path or a BGR array — the pipeline passes the preprocessed
    array so that detection and cropping share one coordinate space.
    """
    result = detector.predict(image if isinstance(image, np.ndarray) else str(image))
    return sort_reading_order(result[0]['dt_polys'])


def box_bounds(box):
    """Polygon -> integer (x1, y1, x2, y2) bounding box."""
    pts = [(int(x), int(y)) for x, y in box]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def read_crop_paddle(paddle_recognizer, crop_bgr):
    """(text, confidence) from the PP-OCR recogniser for one crop."""
    out = paddle_recognizer.predict(crop_bgr)
    if not out:
        return "", 0.0
    return out[0].get("rec_text", ""), float(out[0].get("rec_score", 0.0))


def read_crop_hybrid(processor, model, paddle_recognizer, crop_bgr,
                     recognizer=RECOGNIZER,
                     threshold=PADDLE_CONFIDENCE_THRESHOLD,
                     use_reranker=USE_RERANKER):
    """
    Transcribe one crop with whichever recogniser the configured mode selects.

    PP-OCR reads this corpus better than TrOCR overall, so in "hybrid" mode it
    wins any box it is confident about and TrOCR picks up the rest.
    """
    if recognizer == "trocr":
        return read_crop(processor, model, crop_bgr, use_reranker=use_reranker)

    text, score = read_crop_paddle(paddle_recognizer, crop_bgr)
    if recognizer == "paddle" or score >= threshold:
        return text
    return read_crop(processor, model, crop_bgr, use_reranker=use_reranker)


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
              paddle_recognizer=None,
              recognizer=RECOGNIZER,
              use_reranker=USE_RERANKER, verbose=True,
              use_preprocessing=USE_PREPROCESSING,
              preprocessed_path=None, boxes_path=None):
    """
    Preprocess, detect, transcribe and put into reading order.

    Returns (lines_with_boxes, page_shape), where lines_with_boxes is
    [(text, x1, y1, x2, y2), ...] in document reading order, in the coordinate
    space of the processed image. `preprocessed_path` and `boxes_path`, when
    given, receive the DIP output and the box visualisation.
    """
    print("[1/6] Document image processing...")
    t0 = time.time()
    if use_preprocessing:
        img = load_image(img_path, use_preprocessing=True,
                         preprocessed_path=preprocessed_path)
        print(f"      Preprocessed to {img.shape[1]}x{img.shape[0]} "
              f"in {time.time() - t0:.0f}s")
        if preprocessed_path:
            print(f"      Saved DIP image: {preprocessed_path}")
        print()
    else:
        img = load_image(img_path, use_preprocessing=False)
        print("      Skipped (USE_PREPROCESSING is off)\n")

    print("[2/6] Detecting text lines...")
    t0 = time.time()
    boxes = detect_boxes(detector, img)
    print(f"      Detected and sorted {len(boxes)} boxes in {time.time() - t0:.0f}s\n")

    if boxes_path:
        os.makedirs(os.path.dirname(str(boxes_path)) or ".", exist_ok=True)
        cv2.imwrite(str(boxes_path), draw(img, boxes))
        print(f"      Saved box visualisation: {boxes_path}\n")

    label = {"trocr": "TrOCR + reranker" if use_reranker else "TrOCR (greedy)",
             "paddle": "PP-OCR",
             "hybrid": f"PP-OCR (conf >= {PADDLE_CONFIDENCE_THRESHOLD}) else TrOCR"}[recognizer]
    print(f"[3/6] Recognition: {label}")
    t0 = time.time()

    lines_with_boxes = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box_bounds(box)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text = read_crop_hybrid(processor, model, paddle_recognizer, crop,
                                recognizer=recognizer, use_reranker=use_reranker)
        lines_with_boxes.append((text, x1, y1, x2, y2))

    print(f"      Recognition done in {time.time() - t0:.0f}s\n")

    # Reading order is decided here, from the geometry, not by an LLM later.
    lines_with_boxes = order_items(lines_with_boxes, img.shape[:2])
    if verbose:
        for i, (text, x1, y1, *_ ) in enumerate(lines_with_boxes):
            print(f"      Line {i:2d}: [{x1:4d},{y1:4d}] {text}")
        print()

    return lines_with_boxes, img.shape[:2]
