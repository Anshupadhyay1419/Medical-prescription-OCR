"""
End-to-end orchestration for one image.

    preprocess -> detect -> recognise -> [Point D] -> [Point B] -> clean

Two arms can be written from a single recognition pass: the raw TrOCR baseline
and the full LLM pipeline. Producing both together means the comparison is never
confounded by two different detection runs.
"""
import time

from prescription_ocr.config import (
    USE_CORRECTOR, USE_PREPROCESSING, USE_RERANKER, USE_RESTRUCTURER,
)
from prescription_ocr.io_utils import write_lines
from prescription_ocr.llm.corrector import correct_lines_batch
from prescription_ocr.llm.restructurer import restructure_document
from prescription_ocr.ocr.models import load_models
from prescription_ocr.ocr.recognizer import recognize
from prescription_ocr.postprocess import clean_ocr_output


def apply_llm_stages(lines_with_boxes,
                     use_restructurer=USE_RESTRUCTURER,
                     use_corrector=USE_CORRECTOR):
    """Point D (reading order) then Point B (line corrections)."""
    if use_restructurer:
        print("[4/6] Point D: LLM document restructuring...")
        t0 = time.time()
        lines = restructure_document(lines_with_boxes, verbose=True)
        print(f"      Restructured {len(lines_with_boxes)} -> {len(lines)} lines "
              f"in {time.time() - t0:.0f}s\n")
    else:
        lines = [text for text, *_ in lines_with_boxes]

    if use_corrector:
        print("[5/6] Point B: LLM line-level correction...")
        t0 = time.time()
        lines = correct_lines_batch(lines, verbose=True)
        print(f"      Corrected {len(lines)} lines in {time.time() - t0:.0f}s\n")

    return lines


def run_pipeline(img_path,
                 output_path=None,
                 raw_output_path=None,
                 preprocessed_path=None,
                 boxes_path=None,
                 models=None,
                 use_reranker=USE_RERANKER,
                 use_preprocessing=USE_PREPROCESSING,
                 use_restructurer=USE_RESTRUCTURER,
                 use_corrector=USE_CORRECTOR):
    """
    Recognise one image and write whichever outputs were requested.

    output_path       final transcription (None skips the LLM stages entirely)
    raw_output_path   raw TrOCR baseline (None skips writing it)
    preprocessed_path DIP image handed to PaddleOCR (None skips writing it)
    boxes_path        detected boxes drawn in reading order (None skips it)
    models            (detector, processor, model); loaded here if not supplied
    """
    print("=" * 60)
    print("PRESCRIPTION OCR PIPELINE")
    print("=" * 60)
    print(f"Input:  {img_path}")
    print(f"Output: {output_path or '(skipped)'}   Raw: {raw_output_path or '(skipped)'}")
    print(f"Config: preprocessing={use_preprocessing}, reranker={use_reranker}, "
          f"restructurer={use_restructurer}, corrector={use_corrector}")
    print()

    detector, processor, model = models if models else load_models()

    lines_with_boxes = recognize(detector, processor, model, img_path,
                                 use_reranker=use_reranker,
                                 use_preprocessing=use_preprocessing,
                                 preprocessed_path=preprocessed_path,
                                 boxes_path=boxes_path)

    # The raw arm gets the same deterministic cleanup as every other arm
    # (whitespace + dose-code normalisation) but no LLM anywhere, so the
    # comparison isolates the LLM's contribution.
    if raw_output_path:
        write_lines(raw_output_path,
                    clean_ocr_output([text for text, *_ in lines_with_boxes]))
        print(f"Saved raw TrOCR baseline: {raw_output_path}")

    if output_path:
        lines = apply_llm_stages(lines_with_boxes,
                                 use_restructurer=use_restructurer,
                                 use_corrector=use_corrector)
        print("[6/6] Deterministic cleanup...")
        write_lines(output_path, clean_ocr_output(lines))
        print(f"Saved final cleaned output: {output_path}")

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()
