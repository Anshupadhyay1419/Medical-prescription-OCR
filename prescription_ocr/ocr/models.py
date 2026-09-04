"""
Loading the detection and recognition models.

Loading is slow (tens of seconds) and the weights are large, so a batch run
loads once and passes the handles to every image. To swap the recogniser, edit
TROCR_MODEL in config.py rather than anything here.
"""
import time

import torch
from paddleocr import TextDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from prescription_ocr.config import TROCR_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Return (detector, processor, model) ready for inference."""
    print("[LOAD] Loading models...")
    t0 = time.time()

    detector = TextDetection()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    model.to(DEVICE)
    model.eval()

    print(f"       Loaded in {time.time() - t0:.0f}s (device: {DEVICE})\n")
    return detector, processor, model
