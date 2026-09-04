"""
Rendering detected boxes onto an image.

Shared by the pipeline (which saves a visualisation on every run) and the
standalone draw_boxes command, so both produce the same picture.
"""
import cv2
import numpy as np

BOX_COLOR = (0, 200, 0)        # BGR — green outline
LABEL_BG = (0, 0, 220)         # red chip behind the index
LABEL_FG = (255, 255, 255)


def draw(img, boxes):
    """Outline each box and stamp its reading-order index."""
    vis = img.copy()
    # Scale strokes with image size so 2.5MP scans and small jpegs both read.
    scale = max(1.0, min(vis.shape[:2]) / 900.0)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.5 * scale

    for i, box in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)

        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=True,
                      color=BOX_COLOR, thickness=thickness)

        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Keep the chip on-canvas for boxes that touch the top edge.
        ly = max(th + 4, y1)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly), LABEL_BG, -1)
        cv2.putText(vis, label, (x1 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, LABEL_FG, thickness, cv2.LINE_AA)

    return vis
