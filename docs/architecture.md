# How the pipeline works

## The problem

A prescription is not a paragraph. It is a two-dimensional layout: a drug name
on the left, its dose schedule in the middle, its duration on the right, all on
the same visual row. The detector emits each of those as a separate box, in no
useful order. Reading the handwriting correctly is only half the job — putting
the fragments back together in the right order is the other half, and getting it
wrong attaches the wrong dose to the wrong drug.

That is why the pipeline carries bounding-box coordinates all the way from
detection into the LLM stage, instead of flattening to text early.

## Stages

```
image
  │
  ├─ 1. detection          PaddleOCR TextDetection → polygons
  │                        ocr/recognizer.py :: detect_boxes
  │
  ├─ 2. reading order      group boxes into visual rows, sort left-to-right
  │                        ocr/reading_order.py :: sort_reading_order
  │
  ├─ 3. recognition        TrOCR reads each cropped box
  │                        ocr/recognizer.py :: read_crop
  │                        └─ optional Point A: beam search + LLM reranking
  │
  │        ├──────────────► raw arm: cleanup only, no LLM  → results/trocr_raw/
  │        │
  ├─ 4. Point D            LLM reflows boxes into reading order, using coords
  │                        llm/restructurer.py
  │
  ├─ 5. Point B            LLM fixes known OCR confusions, line by line
  │                        llm/corrector.py
  │
  └─ 6. cleanup            strip LLM commentary, normalise dose codes
                           postprocess.py    → results/trocr_llm/
```

Both arms come from a single recognition pass, so the comparison between them is
never confounded by two different detection runs.

## Why greedy decoding by default

Beam search produces more *fluent* text, which on handwriting means text that
drifts toward plausible English and away from what is actually written. Greedy
decoding is the default; beams are only generated when the reranker (Point A) is
enabled to choose between them. `USE_RERANKER` is off in `config.py`.

## Why the LLM stages are wrapped in safety checks

An LLM asked to reorganise text will occasionally drop a line, merge two
medications, or narrate what it just did. On a prescription those are not
cosmetic failures. So every LLM output must survive a set of checks before it is
accepted, and anything suspicious falls back to the unmodified OCR text:

**Restructurer** (`llm/restructurer.py`)
- *Structural collapse* — output with far fewer lines than the page has visual
  rows means the model answered in one paragraph. Rejected.
- *Character bounds* — output below 40% or above 150% of the input character
  count means text was dropped or hallucinated. Rejected.
- *Medication preservation* — every drug name in the input must appear in the
  output. Matched by name, not by count, so a substitution is caught too.
  Missing ones are appended back rather than silently lost.

**Corrector** (`llm/corrector.py`)
- Meta-replies ("unchanged", "no correction") are treated as no-ops.
- Length must stay within 0.4×–2.5× the original.
- Numbers may be *gained* (`HLAIC` → `HbA1c` gains a `1`) but never lost.
- Dose schedules present in the input must survive verbatim.

A rejected correction costs a typo. An accepted hallucination costs a wrong
dose. The checks are deliberately biased toward rejection.

## Why dose codes are fixed with regex, not the LLM

Handwritten `1` and `0` are routinely read as `I` and `O`, so `I-O-1` needs to
become `1-0-1`. These are the highest-value characters on the page, so
`postprocess.py` normalises them deterministically rather than hoping an LLM
does it consistently.

## The hybrid arm

PP-OCR is strong on printed text (letterheads, pre-printed labels); TrOCR is
strong on handwriting. `dump_candidates` records *both* recognisers' reading of
every box, once, on the GPU. `build_hybrid` then picks per box: PP-OCR when its
confidence clears a threshold, TrOCR otherwise.

Dumping first is the point — sweeping the threshold costs seconds instead of a
full GPU pass per candidate value:

```bash
python -m prescription_ocr.cli.dump_candidates          # once, slow
python -m prescription_ocr.cli.build_hybrid --thr 0.9   # instant, repeatable
```

At the default 0.80 threshold, PP-OCR wins 94% of boxes, and the arm reaches
0.958 dose recall — the best of any non-VLM arm.

## Evaluation

Headline metrics are **document-level** CER/WER: the whole page is joined into
one string before comparison. This is standard for full-page OCR and is
insensitive to how text happened to be split into lines.

Corpus totals aggregate as *(total edits / total reference length)*, not as an
average of per-file rates, so one short file cannot dominate.

For a prescription system, **drug recall** and **dose recall** matter more than
CER: getting the drug name and the schedule right is the entire point.

> A previous version of the evaluator averaged a per-line "best match" CER.
> Dividing an edit distance by the length of a possibly very short reference
> line let a single collapsed-paragraph prediction score a CER of 29. Those
> numbers were measurement artefacts. The statistic survives only as a
> diagnostic, clamped to [0, 1].

## Debugging reading order

```bash
python -m prescription_ocr.cli.draw_boxes --only 17
```

This writes the scan with every detected box outlined and numbered in the
pipeline's own reading order. If the numbers jump between a left and a right
column instead of running down one and then the other, the sort is the problem —
not TrOCR.
