"""
Save PaddleOCR text-detection visualisations for every image.

    python draw_boxes.py                 # all images, skips ones already drawn
    python draw_boxes.py --only 17       # single image
    python draw_boxes.py --force         # redraw everything

Writes <out-dir>/image<N>_boxes.png — the original scan with every detected text
region outlined and numbered.

The numbers are the pipeline's own reading order (the same sort_reading_order
used by trocr_llm), not raw detector order. That makes these directly useful for
diagnosing the multi-column interleaving failure: if the numbers jump between a
left and a right column instead of running down one and then the other, the sort
is the problem, not TrOCR.
"""
import argparse
import os
import time

import cv2
import numpy as np
from paddleocr import TextDetection

from trocr_llm import list_images, extract_image_number, sort_reading_order

OUT_DIR = "detection_boxes"

BOX_COLOR = (0, 200, 0)        # BGR — green outline
LABEL_BG = (0, 0, 220)         # red chip behind the index
LABEL_FG = (255, 255, 255)


def draw(img, boxes):
    """Outline each box and stamp its reading-order index."""
    vis = img.copy()
    # scale line/text with image size so 2.5MP scans and small jpegs both read
    scale = max(1.0, min(vis.shape[:2]) / 900.0)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.5 * scale

    for i, box in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)

        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=True, color=BOX_COLOR,
                      thickness=thickness)

        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # keep the chip on-canvas for boxes that touch the top edge
        ly = max(th + 4, y1)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly), LABEL_BG, -1)
        cv2.putText(vis, label, (x1 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, LABEL_FG, thickness, cv2.LINE_AA)

    return vis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, metavar="N", help="process only image N")
    p.add_argument("--force", action="store_true", help="redraw images already done")
    p.add_argument("--img-dir", default="image")
    p.add_argument("--out-dir", default=OUT_DIR)
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

    det = TextDetection()
    t_all = time.time()
    for i, (path, out) in enumerate(todo, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i:2d}/{len(todo)}] {os.path.basename(path):18s} -> unreadable, skipped")
            continue
        boxes = sort_reading_order(det.predict(path)[0]['dt_polys'])
        cv2.imwrite(out, draw(img, boxes))
        print(f"[{i:2d}/{len(todo)}] {os.path.basename(path):18s} -> "
              f"{len(boxes):3d} boxes  {out}")

    print(f"\nDone in {time.time() - t_all:.0f}s. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
