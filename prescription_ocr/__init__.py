"""
Handwritten prescription OCR: PaddleOCR detection + TrOCR recognition, with
optional local-LLM stages for reading order and error correction.

Layout:
    config.py       every path, model name and toggle (start here)
    io_utils.py     locating images, reading/writing per-image text files
    postprocess.py  deterministic cleanup applied to every arm
    ocr/            detection, reading order, recognition, orchestration
    llm/            Ollama-backed reranker, restructurer and corrector
    evaluation/     CER/WER and clinical metrics
    cli/            the commands you actually run
"""
__version__ = "1.0.0"
