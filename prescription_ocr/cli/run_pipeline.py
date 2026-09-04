"""
Run the OCR pipeline over the image corpus.

    python -m prescription_ocr.cli.run_pipeline              # everything not yet done
    python -m prescription_ocr.cli.run_pipeline --only 17    # one image, to smoke-test
    python -m prescription_ocr.cli.run_pipeline --force      # redo finished images
    python -m prescription_ocr.cli.run_pipeline --no-llm     # raw TrOCR baseline only

By default both arms are written from a single recognition pass: the raw TrOCR
baseline and the full LLM pipeline. Images whose output already exists are
skipped, so an interrupted batch resumes where it stopped.
"""
import argparse
import os
import time

from prescription_ocr.config import IMAGES_DIR, TROCR_LLM_DIR, TROCR_RAW_DIR
from prescription_ocr.io_utils import extract_image_number, list_images, output_path
from prescription_ocr.ocr.models import load_models
from prescription_ocr.ocr.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, metavar="N",
                   help="process only image N (smoke-test a change before the batch)")
    p.add_argument("--force", action="store_true",
                   help="redo images whose output already exists")
    p.add_argument("--no-llm", action="store_true",
                   help="write the raw TrOCR baseline only, skip the LLM stages")
    p.add_argument("--no-raw", action="store_true",
                   help="skip writing the raw TrOCR baseline")
    p.add_argument("--img-dir", default=str(IMAGES_DIR))
    p.add_argument("--out-dir", default=str(TROCR_LLM_DIR))
    p.add_argument("--raw-dir", default=str(TROCR_RAW_DIR))
    return p.parse_args()


def build_worklist(args):
    """[(image path, output path or None, raw path or None), ...] still to do."""
    image_files = list_images(args.img_dir)
    if not image_files:
        raise FileNotFoundError(f"No supported images found in {args.img_dir}")

    if args.only is not None:
        image_files = [p for p in image_files if extract_image_number(p) == args.only]
        if not image_files:
            raise FileNotFoundError(f"No image numbered {args.only} in {args.img_dir}")

    todo = []
    for image_path in image_files:
        n = extract_image_number(image_path)
        out = None if args.no_llm else output_path(args.out_dir, n)
        raw = None if args.no_raw else output_path(args.raw_dir, n)

        # Skip only the arms already on disk, not the whole image.
        if not args.force:
            if out and os.path.exists(out):
                out = None
            if raw and os.path.exists(raw):
                raw = None

        if out or raw:
            todo.append((image_path, out, raw))

    return image_files, todo


def main():
    args = parse_args()
    image_files, todo = build_worklist(args)

    if not todo:
        print("Nothing to do — all requested outputs already exist. Use --force to redo.")
        return
    print(f"{len(image_files) - len(todo)} already done, {len(todo)} to process.\n")

    models = load_models()

    t_all = time.time()
    for i, (image_path, out, raw) in enumerate(todo, 1):
        print(f"\n### [{i}/{len(todo)}] {image_path}")
        try:
            run_pipeline(img_path=image_path, output_path=out,
                         raw_output_path=raw, models=models)
        except Exception as e:
            # One bad scan must not abandon the rest of an hours-long batch.
            print(f"!!! FAILED on {image_path}: {type(e).__name__}: {e}")

    print(f"\nBatch done in {time.time() - t_all:.0f}s.")


if __name__ == "__main__":
    main()
