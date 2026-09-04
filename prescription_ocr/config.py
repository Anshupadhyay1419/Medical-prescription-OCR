from pathlib import Path

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------

# config.py lives at <repo>/prescription_ocr/config.py, so the root is 2 up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"                # input scans
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"    # image<N>.txt references

RESULTS_DIR = PROJECT_ROOT / "results"

# A complete pipeline run fills exactly these three directories.
TEXT_DIR = RESULTS_DIR / "text"                 # final transcriptions
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed" # DIP output fed to PaddleOCR
DETECTION_BOXES_DIR = RESULTS_DIR / "detection_boxes"   # boxes in reading order

# Earlier comparison arms, kept for scoring but out of the way so the three
# directories above are the only ones a normal run touches.
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
TROCR_RAW_DIR = EXPERIMENTS_DIR / "trocr_raw"   # TrOCR only, no LLM
HYBRID_DIR = EXPERIMENTS_DIR / "hybrid"         # PP-OCR/TrOCR per-box pick
QWEN_VL_DIR = EXPERIMENTS_DIR / "qwen_vl"       # Qwen2.5-VL baseline
CANDIDATES_DIR = EXPERIMENTS_DIR / "candidates" # per-box dual-recogniser dumps

# --------------------------------------------------------------------------
# File naming
# --------------------------------------------------------------------------

# Outputs are <arm dir>/final_clean<N>.txt, references are image<N>.txt.
OUTPUT_PREFIX = "final_clean"
GROUND_TRUTH_PREFIX = "image"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Handwriting recogniser pulled from the HuggingFace hub.
TROCR_MODEL = "microsoft/trocr-large-handwritten"

# Local Ollama server used by every LLM stage.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

MAX_NEW_TOKENS = 64     # per detected text line
NUM_BEAMS = 5           # only used when the Point A reranker is enabled

# --------------------------------------------------------------------------
# Pipeline toggles
# --------------------------------------------------------------------------

# Document image processing — illumination correction, denoising, CLAHE,
# deskew and binarisation — applied before detection. PaddleOCR and TrOCR both
# see the processed image, so detection boxes are in its coordinate space.
USE_PREPROCESSING = True

# Point A — beam search + LLM reranking of the candidates for each line.
#           Off by default: beams hallucinate more than greedy on handwriting.
USE_RERANKER = False

# Point D — LLM reflows the detected boxes into correct reading order.
USE_RESTRUCTURER = True

# Point B — LLM fixes known OCR confusions line by line.
USE_CORRECTOR = True

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

# Arms shown by `compare`, in table order: (display name, results directory).
COMPARISON_ARMS = [
    ("Raw TrOCR", TROCR_RAW_DIR),
    ("TrOCR + LLM", TEXT_DIR),
    ("Hybrid PP+TrOCR", HYBRID_DIR),
    ("Qwen2.5-VL", QWEN_VL_DIR),
]

# These scans are clinical narrative, not prescriptions. Scoring them next to
# prescriptions mixes two different tasks, so they are reported separately.
NON_PRESCRIPTION_IMAGES = {11, 12, 13, 14, 15}

# Per-box confidence at or above which the hybrid arm trusts PP-OCR over TrOCR.
HYBRID_PADDLE_THRESHOLD = 0.80
