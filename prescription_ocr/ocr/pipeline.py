"""
End-to-end orchestration for one image.

    preprocess -> detect -> recognise -> [Point D] -> [Point B] -> clean

Two arms can be written from a single recognition pass: the raw TrOCR baseline
and the full LLM pipeline. Producing both together means the comparison is never
confounded by two different detection runs.
"""
import time

from prescription_ocr.config import (
    RECOGNIZER, USE_CORRECTOR, USE_PREPROCESSING, USE_RERANKER, USE_RESTRUCTURER,
)
from prescription_ocr.io_utils import write_lines
from prescription_ocr.llm.corrector import correct_lines_batch
from prescription_ocr.llm.restructurer import restructure_document
from prescription_ocr.ocr.layout import merge_rows
from prescription_ocr.ocr.models import load_models
from prescription_ocr.ocr.recognizer import recognize
from prescription_ocr.postprocess import clean_ocr_output


def assemble_lines(lines_with_boxes, page_shape,
                   use_restructurer=USE_RESTRUCTURER,
                   use_corrector=USE_CORRECTOR):
    """
    Turn per-box results into document lines.

    By default this is pure geometry: boxes sharing a visual row are joined in
    reading order by ocr/layout.py. The LLM stages are opt-in because, measured
    against ground truth, both made the output worse.
    """
    if use_restructurer:
        print("[4/6] Point D: LLM document restructuring...")
        t0 = time.time()
        lines = restructure_document(lines_with_boxes, verbose=True)
        print(f"      Restructured {len(lines_with_boxes)} -> {len(lines)} lines "
              f"in {time.time() - t0:.0f}s\n")
    else:
        print("[4/6] Assembling rows from box geometry...")
        t0 = time.time()
        lines = merge_rows(lines_with_boxes, page_shape)
        print(f"      Merged {len(lines_with_boxes)} boxes -> {len(lines)} rows "
              f"in {time.time() - t0:.1f}s\n")

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
                 recognizer=RECOGNIZER,
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
    print(f"Config: recogniser={recognizer}, preprocessing={use_preprocessing}, "
          f"restructurer={use_restructurer}, corrector={use_corrector}")
    print()

    detector, processor, model, paddle_recognizer = (
        models if models else load_models(recognizer))

    lines_with_boxes, page_shape = recognize(
        detector, processor, model, img_path,
        paddle_recognizer=paddle_recognizer,
        recognizer=recognizer,
        use_reranker=use_reranker,
        use_preprocessing=use_preprocessing,
        preprocessed_path=preprocessed_path,
        boxes_path=boxes_path)

    # The raw arm is one line per detected box, with the same deterministic
    # cleanup every arm gets. It isolates recognition quality from the row
    # assembly that the main arm applies on top.
    if raw_output_path:
        write_lines(raw_output_path,
                    clean_ocr_output([text for text, *_ in lines_with_boxes]))
        print(f"Saved per-box baseline: {raw_output_path}")

    if output_path:
        lines = assemble_lines(lines_with_boxes, page_shape,
                               use_restructurer=use_restructurer,
                               use_corrector=use_corrector)
        print("[6/6] Deterministic cleanup...")
        write_lines(output_path, clean_ocr_output(lines))
        print(f"Saved final cleaned output: {output_path}")

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print()
