"""
Save text-detection visualisations for every image.

    python -m prescription_ocr.cli.draw_boxes             # all, skips done ones
    python -m prescription_ocr.cli.draw_boxes --only 17   # single image
    python -m prescription_ocr.cli.draw_boxes --force     # redraw everything

Writes results/detection_boxes/image<N>_boxes.png — the original scan with every
detected region outlined and numbered.

The boxes are drawn on the preprocessed (DIP) image, because that is what the
detector actually sees. Pass --no-dip to draw on the original scan instead.

The numbers are the pipeline's own reading order, not raw detector order. That
makes these directly useful for diagnosing multi-column interleaving: if the
numbers jump between a left and a right column instead of running down one and
then the other, the sort is the problem, not TrOCR.

run_pipeline already writes these on every run; this command is for redrawing
them without redoing recognition.
"""
import argparse
import os
import time

import cv2
from paddleocr import TextDetection

from prescription_ocr.config import DETECTION_BOXES_DIR, IMAGES_DIR
from prescription_ocr.io_utils import extract_image_number, list_images
from prescription_ocr.ocr.layout import order_items
from prescription_ocr.ocr.recognizer import box_bounds, detect_boxes, load_image
from prescription_ocr.ocr.visualize import draw


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, metavar="N", help="process only image N")
    p.add_argument("--force", action="store_true", help="redraw images already done")
    p.add_argument("--no-dip", action="store_true",
                   help="draw on the original scan instead of the preprocessed image")
    p.add_argument("--img-dir", default=str(IMAGES_DIR))
    p.add_argument("--out-dir", default=str(DETECTION_BOXES_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    images = list_images(args.img_dir)
    if args.only is not None:
        images = [p for p in images if extract_image_number(p) == args.only]
        if not images:
            raise FileNotFoundError(f"No image numbered {args.only} in {args.img_dir}")

    todo = []
    for path in images:
        out = os.path.join(args.out_dir, f"image{extract_image_number(path)}_boxes.png")
        if args.force or not os.path.exists(out):
            todo.append((path, out))

    if not todo:
        print(f"All {len(images)} visualisations already in {args.out_dir}/. "
              f"Use --force to redraw.")
        return
    print(f"{len(images) - len(todo)} already done, {len(todo)} to draw.\n")

    detector = TextDetection()
    t_all = time.time()
    for i, (path, out) in enumerate(todo, 1):
        try:
            img = load_image(path, use_preprocessing=not args.no_dip)
        except ValueError:
            print(f"[{i:2d}/{len(todo)}] {os.path.basename(path):18s} -> unreadable, skipped")
            continue
        boxes = detect_boxes(detector, img)
        # Number them in the pipeline's reading order, not raw detector order.
        items = [(None, *box_bounds(b)) for b in boxes]
        ordered = order_items(items, img.shape[:2])
        index = {tuple(box_bounds(b)): b for b in boxes}
        boxes = [index[(i[1], i[2], i[3], i[4])] for i in ordered]
        cv2.imwrite(out, draw(img, boxes))
        print(f"[{i:2d}/{len(todo)}] {os.path.basename(path):18s} -> "
              f"{len(boxes):3d} boxes  {out}")

    print(f"\nDone in {time.time() - t_all:.0f}s. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
