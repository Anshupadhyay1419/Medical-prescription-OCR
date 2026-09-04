# Archive

Superseded code, kept for reference. Nothing here is imported by the package or
run by any command.

- **`paddle_trocr_prototype.py`** — the original single-file proof of concept:
  detection, reading-order sort, TrOCR, straight to a text file. No LLM stages,
  no batching, no evaluation, and its hardcoded input path (`images/bw.png`) no
  longer exists. Superseded by `prescription_ocr/ocr/pipeline.py`, which does
  the same thing with batching, resumable runs and the LLM stages.
