# Prescription OCR

Reading handwritten Indian medical prescriptions: **PaddleOCR** finds the text
lines, **TrOCR** reads the handwriting, and a **local LLM** (via Ollama) fixes
the reading order and known OCR confusions.

Four recogniser arms are kept side by side so any change can be measured rather
than guessed at.

---

## Quick start

```bash
pip install -r requirements.txt          # see "Requirements" below for Ollama

python -m prescription_ocr.cli.run_pipeline --only 17   # one image, ~25s
python -m prescription_ocr.cli.evaluate                 # score it
python -m prescription_ocr.cli.compare                  # all arms side by side
```

Every command runs from the repository root and takes `--help`.

---

## Where things live

```
prescription_ocr/          the package — all the logic
├── config.py              ► EVERY path, model name and toggle. Start here.
├── io_utils.py            finding images, reading/writing per-image text files
├── postprocess.py         deterministic cleanup applied to every arm
│
├── ocr/                   the recognition pipeline
│   ├── reading_order.py   sorting detected boxes the way a human reads
│   ├── models.py          loading the detector + TrOCR
│   ├── recognizer.py      detect → sort → transcribe each box
│   └── pipeline.py        end-to-end orchestration for one image
│
├── llm/                   the local-LLM stages
│   ├── client.py          the one place we talk to Ollama
│   ├── prompts/*.txt      ► the prompts, as plain text. Tune without code.
│   ├── restructurer.py    Point D — rebuild reading order from coordinates
│   ├── corrector.py       Point B — fix known OCR confusions, line by line
│   └── reranker.py        Point A — pick the best of N beam candidates
│
├── evaluation/metrics.py  CER/WER + drug and dose recall
└── cli/                   the commands you actually run

data/
├── images/                input scans (image<N>.png)
└── ground_truth/          reference transcriptions (image<N>.txt)

results/                   one directory per arm, all as final_clean<N>.txt
├── trocr_raw/             TrOCR only, no LLM        (baseline)
├── trocr_llm/             TrOCR + LLM stages        (this project's pipeline)
├── hybrid/                PP-OCR or TrOCR per box   (best non-VLM arm)
├── qwen_vl/               Qwen2.5-VL                (reference ceiling)
├── candidates/            per-box dual-recogniser dumps (feeds `hybrid`)
└── detection_boxes/       debug visualisations of detection + reading order

archive/                   superseded prototypes, kept for reference only
docs/architecture.md       how the pipeline works and why
```

---

## How to make a change

| I want to… | Edit this |
|---|---|
| Point at a different dataset | `config.py` → `IMAGES_DIR`, `GROUND_TRUTH_DIR` |
| Swap the handwriting model | `config.py` → `TROCR_MODEL` |
| Swap the LLM | `config.py` → `OLLAMA_MODEL`, `OLLAMA_URL` |
| Turn an LLM stage on/off | `config.py` → `USE_RESTRUCTURER`, `USE_CORRECTOR`, `USE_RERANKER` |
| Change what the LLM is told | `prescription_ocr/llm/prompts/*.txt` — no Python involved |
| Change how boxes group into rows | `ocr/reading_order.py` → `ROW_GROUPING_FACTOR` |
| Add a new safety check on LLM output | `llm/restructurer.py` or `llm/corrector.py` |
| Add a new metric | `evaluation/metrics.py` |
| Add a new arm to the comparison | `config.py` → `COMPARISON_ARMS` |

Nothing outside `config.py` hardcodes a path or a model name, and every path is
anchored to the repository root — commands behave the same from any directory.

---

## The commands

| Command | What it does |
|---|---|
| `run_pipeline` | Recognise the corpus. Writes the raw **and** LLM arms from one pass. |
| `evaluate` | Per-image CER/WER table for one arm. Takes a results directory. |
| `compare` | All four arms side by side, on the images they all produced. |
| `dump_candidates` | Records both recognisers' reading of every box (GPU, run once). |
| `build_hybrid` | Builds the hybrid arm from those dumps. Sweep `--thr` in seconds. |
| `draw_boxes` | Renders detection boxes numbered in reading order — the debug tool. |

Useful flags on `run_pipeline`:

```bash
--only 17     # one image, to smoke-test a change before a full batch
--force       # redo images whose output already exists (default: resume)
--no-llm      # raw TrOCR baseline only
```

Batches resume: finished images are skipped, and one bad scan is logged rather
than aborting the run.

---

## Requirements

Python 3.9+, plus `pip install -r requirements.txt`.

The LLM stages need a **local Ollama server** — not a Python package:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Without it, the LLM stages log an error and fall back to the unmodified OCR
text, so the pipeline still produces output. A CUDA GPU is optional; TrOCR runs
on CPU, just slowly.

---

## Current results

32 prescriptions, corpus totals. Lower CER/WER is better, higher recall is better.

| Arm | CER | WER | nCER | nWER | drug recall | dose recall |
|---|---|---|---|---|---|---|
| Raw TrOCR | 0.452 | 0.906 | 0.380 | 0.659 | 0.550 | 0.661 |
| TrOCR + LLM | 0.442 | 0.841 | 0.378 | 0.634 | 0.558 | 0.685 |
| Hybrid PP+TrOCR | 0.354 | 0.557 | 0.330 | 0.542 | 0.646 | **0.958** |
| Qwen2.5-VL | **0.218** | **0.369** | **0.196** | **0.371** | **0.695** | 0.807 |

`nCER`/`nWER` normalise away cosmetic differences (case, spacing, `5ml` vs
`5 ml`) so they isolate genuine misreads. Images 11–15 are clinical narrative
rather than prescriptions and are reported separately by `compare`.

Reproduce with `python -m prescription_ocr.cli.compare`.
