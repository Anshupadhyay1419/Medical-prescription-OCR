"""
Preprocessing pipeline for handwritten prescription images.
Lightweight, no heavy dependencies.
"""
import os

import cv2
import numpy as np


def preprocess_prescription(img_path, output_path=None, debug=False):
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Load and resize (if too large)
    2. Grayscale conversion (preserve strokes)
    3. Illumination correction (handle uneven lighting)
    4. Denoising (bilateral filter — preserves edges)
    5. Contrast enhancement (CLAHE)
    6. Skew detection + correction
    7. Optional: adaptive binarization
    """
    # ---- Load ----
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot load {img_path}")
    
    original = img.copy()
    
    # ---- Step 1: Resize if huge ----
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ---- Step 2: Grayscale ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ---- Step 3: Illumination correction ----
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    illum_corrected = cv2.divide(gray, blurred, scale=255)
    
    # ---- Step 4: Denoise (bilateral preserves edges) ----
    denoised = cv2.bilateralFilter(illum_corrected, d=9, sigmaColor=75, sigmaSpace=75)
    
    # ---- Step 5: CLAHE contrast ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # ---- Step 6: Deskew ----
    deskewed = deskew_image(enhanced)
    
    binary = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=41, C=15
    )
    
    # convert back to 3-channel for compatibility with PaddleOCR
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if output_path:
        os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
        cv2.imwrite(str(output_path), output)

    return output


def deskew_image(gray):
    """
    Detect skew angle using Hough transform on edges, then rotate to correct.
    """
    # detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=100,
        minLineLength=100, maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return gray  # no lines detected, return as-is
    
    # compute angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # focus on near-horizontal lines (text baselines)
        if -45 <= angle <= 45:
            angles.append(angle)
    
    if not angles:
        return gray
    
    # use median angle (robust to outliers)
    skew_angle = np.median(angles)
    
    # only correct if significant skew
    if abs(skew_angle) < 0.5:
        return gray
    
    # rotate image to correct skew
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    
    # calculate new dimensions to fit rotated image
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(
        gray, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255  # white background
    )
    
    return rotated


if __name__ == "__main__":
    # test standalone
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "image/bw.png"
    output_path = "preprocessed.png"
    
    preprocess_prescription(input_path, output_path, debug=True)
    print(f"Preprocessed image saved to {output_path}")
    print(f"Debug intermediates saved to debug_*.png")