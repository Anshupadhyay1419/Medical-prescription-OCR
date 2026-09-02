import argparse
import os
import re
import time
import cv2
import torch
from PIL import Image
from paddleocr import TextDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# LLM modules
from llm_corrector import correct_lines_batch
from llm_reranker import rerank_candidates
from llm_restructurer import restructure_document
from postprocess import clean_ocr_output


IMG_PATH = "image"
OUTPUT_DIR = "outputs"
RAW_OUTPUT_DIR = "outputs_raw"
FINAL_OUTPUT_PREFIX = "final_clean"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# pipeline toggles
USE_POINT_A = False
USE_POINT_D = True
USE_POINT_B = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sort_reading_order(boxes):
    """Basic sort — grouping boxes into visual rows, then left-to-right."""
    if len(boxes) == 0:
        return []

    box_data = []
    for box in boxes:
        y1 = min(p[1] for p in box)
        y2 = max(p[1] for p in box)
        x1 = min(p[0] for p in box)
        box_data.append((box, y1, x1, y2 - y1))

    box_data.sort(key=lambda x: x[1])

    lines = [[box_data[0]]]
    for item in box_data[1:]:
        avg_y = sum(b[1] for b in lines[-1]) / len(lines[-1])
        avg_h = sum(b[3] for b in lines[-1]) / len(lines[-1])
        if abs(item[1] - avg_y) < avg_h * 0.5:
            lines[-1].append(item)
        else:
            lines[-1].sort(key=lambda x: x[2])
            lines.append([item])
    lines[-1].sort(key=lambda x: x[2])

    return [item[0] for line in lines for item in line]


def load_models():
    """Load detection + recognition once for the whole batch."""
    print("[LOAD] Loading models...")
    t0 = time.time()
    det = TextDetection()
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    model.to(DEVICE)
    model.eval()
    print(f"       Loaded in {time.time() - t0:.0f}s (device: {DEVICE})\n")
    return det, processor, model


@torch.inference_mode()
def recognize(det, processor, model, img_path, use_point_a=USE_POINT_A, verbose=True):
    """Detection + reading-order sort + TrOCR. Returns [(text, x1, y1, x2, y2), ...]."""
    print("[1/5] Detecting text lines...")
    t0 = time.time()
    result = det.predict(img_path)
    raw_boxes = result[0]['dt_polys']
    print(f"      Detected {len(raw_boxes)} boxes in {time.time() - t0:.0f}s\n")

    print("[2/5] Sorting boxes in reading order...")
    sorted_boxes = sort_reading_order(raw_boxes)
    print(f"      Sorted {len(sorted_boxes)} boxes\n")

    img = cv2.imread(img_path)

    label = "TrOCR + Point A reranker" if use_point_a else "TrOCR (greedy)"
    print(f"[3/5] Recognition: {label}")
    t0 = time.time()

    raw_lines_with_boxes = []
    for i, box in enumerate(sorted_boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(DEVICE)

        if use_point_a:
            # beam-5 with reranker
            outputs = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=5,
                num_return_sequences=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            candidates = processor.batch_decode(outputs, skip_special_tokens=True)
            text = rerank_candidates(candidates)
        else:
            # greedy (default — faster, avoids beam hallucinations)
            outputs = model.generate(pixel_values, max_new_tokens=64, num_beams=1)
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        raw_lines_with_boxes.append((text, x1, y1, x2, y2))
        if verbose:
            print(f"      Line {i:2d}: [{x1:4d},{y1:4d}] {text}")

    print(f"      Recognition done in {time.time() - t0:.0f}s\n")
    return raw_lines_with_boxes


def apply_llm_stages(raw_lines_with_boxes, use_point_d=USE_POINT_D, use_point_b=USE_POINT_B):
    """Point D (restructure) then Point B (correct)."""
    if use_point_d:
        print("[4/5] Point D: LLM document restructuring...")
        t0 = time.time()
        restructured_lines = restructure_document(raw_lines_with_boxes, verbose=True)
        print(f"      Restructured {len(raw_lines_with_boxes)} → "
              f"{len(restructured_lines)} lines in {time.time() - t0:.0f}s\n")
    else:
        restructured_lines = [t for t, *_ in raw_lines_with_boxes]

    if use_point_b:
        print("[5/5] Point B: LLM line-level correction...")
        t0 = time.time()
        final_lines = correct_lines_batch(restructured_lines, verbose=True)
        print(f"      Corrected {len(restructured_lines)} lines in {time.time() - t0:.0f}s\n")
    else:
        final_lines = restructured_lines

    return final_lines


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def run_pipeline(
    img_path=IMG_PATH,
    output_path=None,
    raw_output_path=None,
    models=None,
    use_point_a=USE_POINT_A,
    use_point_d=USE_POINT_D,
    use_point_b=USE_POINT_B,
):
    """
    Recognise one image and write the requested arms.

    output_path      full pipeline result (None to skip the LLM stages entirely)
    raw_output_path  raw TrOCR result (None to skip writing it)
    models           (det, processor, model) — loaded here if not supplied
    """
    if output_path is None and raw_output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"{FINAL_OUTPUT_PREFIX}1.txt")

    print("=" * 60)
    print("PRESCRIPTION OCR PIPELINE")
    print("=" * 60)
    print(f"Input:  {img_path}")
    print(f"Output: {output_path or '(skipped)'}   Raw: {raw_output_path or '(skipped)'}")
    print(f"Config: Point A={use_point_a}, Point D={use_point_d}, Point B={use_point_b}")
    print()

    det, processor, model = models if models else load_models()

    raw_lines_with_boxes = recognize(det, processor, model, img_path,
                                     use_point_a=use_point_a)

    # Raw arm: recognised text with the same deterministic cleanup every arm
    # gets (whitespace + dose-code normalisation), but no LLM anywhere.
    if raw_output_path:
        write_lines(raw_output_path, clean_ocr_output([t for t, *_ in raw_lines_with_boxes]))
        print(f"Saved raw TrOCR baseline: {raw_output_path}")

    if output_path:
        final_lines = apply_llm_stages(raw_lines_with_boxes,
                                       use_point_d=use_point_d, use_point_b=use_point_b)
        write_lines(output_path, clean_ocr_output(final_lines))
        print(f"Saved final cleaned output: {output_path}")

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

def natural_sort_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def extract_image_number(filename):
    matches = re.findall(r'(\d+)', os.path.basename(filename))
    return int(matches[-1]) if matches else 0


def list_images(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = []
    for filename in sorted(os.listdir(image_dir), key=natural_sort_key):
        full_path = os.path.join(image_dir, filename)
        if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
            image_files.append(full_path)

    return image_files


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, metavar="N",
                   help="process only image N (smoke-test a change before the batch)")
    p.add_argument("--force", action="store_true",
                   help="redo images whose output already exists")
    p.add_argument("--no-llm", action="store_true",
                   help="write the raw TrOCR baseline only, skip Point D and Point B")
    p.add_argument("--no-raw", action="store_true",
                   help="skip writing the raw TrOCR baseline")
    p.add_argument("--img-dir", default=IMG_PATH)
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--raw-dir", default=RAW_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    image_files = list_images(args.img_dir)
    if not image_files:
        raise FileNotFoundError(f"No supported images found in {args.img_dir}")
    if args.only is not None:
        image_files = [p for p in image_files if extract_image_number(p) == args.only]
        if not image_files:
            raise FileNotFoundError(f"No image numbered {args.only} in {args.img_dir}")

    want_pipeline = not args.no_llm
    want_raw = not args.no_raw

    todo = []
    for image_path in image_files:
        n = extract_image_number(image_path)
        out = os.path.join(args.out_dir, f"{FINAL_OUTPUT_PREFIX}{n}.txt") if want_pipeline else None
        raw = os.path.join(args.raw_dir, f"{FINAL_OUTPUT_PREFIX}{n}.txt") if want_raw else None
        if not args.force:
            # only skip when every arm this run would write is already present
            if out and os.path.exists(out):
                out = None
            if raw and os.path.exists(raw):
                raw = None
        if out or raw:
            todo.append((image_path, out, raw))

    if not todo:
        print("Nothing to do — all requested outputs already exist. Use --force to redo.")
        return
    print(f"{len(image_files) - len(todo)} already done, {len(todo)} to process.\n")

    models = load_models()

    t_all = time.time()
    for i, (image_path, out, raw) in enumerate(todo, 1):
        print(f"\n### [{i}/{len(todo)}] {image_path}")
        try:
            run_pipeline(img_path=image_path, output_path=out, raw_output_path=raw,
                         models=models)
        except Exception as e:
            print(f"!!! FAILED on {image_path}: {type(e).__name__}: {e}")

    print(f"\nBatch done in {time.time() - t_all:.0f}s.")


if __name__ == "__main__":
    main()
