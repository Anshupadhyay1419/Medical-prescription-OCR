"""
Shared filesystem helpers: locating input images and reading/writing the
per-image text files that every stage of the project exchanges.
"""
import os
import re

from prescription_ocr.config import (
    GROUND_TRUTH_PREFIX,
    OUTPUT_PREFIX,
    SUPPORTED_EXTENSIONS,
)


def natural_sort_key(name):
    """Sort key that orders image2 before image10 instead of after it."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(name))]


def extract_image_number(filename):
    """image23.png -> 23. The trailing number is the image's identity."""
    matches = re.findall(r'(\d+)', os.path.basename(str(filename)))
    return int(matches[-1]) if matches else 0


def list_images(image_dir):
    """Every supported image in `image_dir`, in natural order."""
    image_dir = str(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = []
    for filename in sorted(os.listdir(image_dir), key=natural_sort_key):
        full_path = os.path.join(image_dir, filename)
        if (os.path.isfile(full_path)
                and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS):
            image_files.append(full_path)

    return image_files


def output_path(results_dir, image_number):
    """Where an arm's transcription of image <N> lives."""
    return os.path.join(str(results_dir), f"{OUTPUT_PREFIX}{image_number}.txt")


def ground_truth_path(gt_dir, image_number):
    """Where the reference transcription of image <N> lives."""
    return os.path.join(str(gt_dir), f"{GROUND_TRUTH_PREFIX}{image_number}.txt")


def write_lines(path, lines):
    """Write one line per list entry, creating the parent directory if needed."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def load_lines(path):
    """Read non-empty lines, dropping any leftover 'Line N:' prefix."""
    with open(str(path), encoding="utf-8") as f:
        out = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(re.sub(r'^Line\s*\d+\s*:\s*', '', line, flags=re.IGNORECASE))
        return out
