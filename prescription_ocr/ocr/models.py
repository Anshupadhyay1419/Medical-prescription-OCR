"""
Loading the detection and recognition models.

Loading is slow (tens of seconds) and the weights are large, so a batch run
loads once and passes the handles to every image. To swap the recogniser, edit
TROCR_MODEL in config.py rather than anything here.
"""
import time

import torch
from paddleocr import TextDetection, TextRecognition
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from prescription_ocr.config import RECOGNIZER, TROCR_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(recognizer=RECOGNIZER):
    """
    Return (detector, processor, model, paddle_recognizer) ready for inference.

    Only the recognisers the configured mode actually needs are loaded, so a
    TrOCR-only or PP-OCR-only run does not pay for the other one.
    """
    print("[LOAD] Loading models...")
    t0 = time.time()

    detector = TextDetection()

    processor = model = None
    if recognizer in ("trocr", "hybrid"):
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
        model.to(DEVICE)
        model.eval()

    paddle_recognizer = None
    if recognizer in ("paddle", "hybrid"):
        paddle_recognizer = TextRecognition()

    print(f"       Loaded '{recognizer}' in {time.time() - t0:.0f}s (device: {DEVICE})\n")
    return detector, processor, model, paddle_recognizer
