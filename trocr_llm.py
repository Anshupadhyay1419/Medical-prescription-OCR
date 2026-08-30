import os
import time
import cv2
from PIL import Image
from paddleocr import TextDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# LLM modules
from llm_corrector import correct_lines_batch      
from llm_reranker import rerank_candidates         
from llm_restructurer import restructure_document   


IMG_PATH = "images/bw.png"    
OUTPUT_DIR = "outputs"
GROUND_TRUTH = "prescription_02_ground_truth.txt"                      

# pipeline toggles
USE_POINT_A = False  
USE_POINT_D = True   
USE_POINT_B = True  


def sort_reading_order(boxes):
    """Basic sort — grouping boxes into visual rows, then left-to-right."""
    if len(boxes) == 0:           
        return []
    
    box_data = []
    for box in boxes:
        y1 = min(p[1] for p in box)
        y2 = max(p[1] for p in box)
        x1 = min(p[0] for p in box)
        box_data.append((box, y1, x1, y2 - y1))
    
    box_data.sort(key=lambda x: x[1])
    
    lines = [[box_data[0]]]
    for item in box_data[1:]:
        avg_y = sum(b[1] for b in lines[-1]) / len(lines[-1])
        avg_h = sum(b[3] for b in lines[-1]) / len(lines[-1])
        if abs(item[1] - avg_y) < avg_h * 0.5:
            lines[-1].append(item)
        else:
            lines[-1].sort(key=lambda x: x[2])
            lines.append([item])
    lines[-1].sort(key=lambda x: x[2])
    
    return [item[0] for line in lines for item in line]



def run_pipeline(
    img_path=IMG_PATH,
    output_dir=OUTPUT_DIR,
    use_point_a=USE_POINT_A,
    use_point_d=USE_POINT_D,
    use_point_b=USE_POINT_B,
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PRESCRIPTION OCR PIPELINE")
    print("=" * 60)
    print(f"Input:  {img_path}")
    print(f"Output: {output_dir}/")
    print(f"Config: Point A={use_point_a}, Point D={use_point_d}, Point B={use_point_b}")
    print()
    
    print("[LOAD] Loading models...")
    t0 = time.time()
    det = TextDetection()
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    print(f"       Loaded in {time.time() - t0:.0f}s\n")
    
    # ---- STAGE 1: DETECTION ----
    print(f"[1/5] Detecting text lines...")
    t0 = time.time()
    result = det.predict(img_path)
    raw_boxes = result[0]['dt_polys']
    print(f"      Detected {len(raw_boxes)} boxes in {time.time() - t0:.0f}s\n")
    
    # ---- STAGE 2: READING-ORDER SORT ----
    print(f"[2/5] Sorting boxes in reading order...")
    sorted_boxes = sort_reading_order(raw_boxes)
    print(f"      Sorted {len(sorted_boxes)} boxes\n")
    
    # save debug image with box numbers
    img = cv2.imread(img_path)
    debug_img = img.copy()
    for i, box in enumerate(sorted_boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(f"{output_dir}/debug_detection.png", debug_img)
    
    # ---- STAGE 3: TrOCR RECOGNITION (with optional Point A) ----
    label = "TrOCR + Point A reranker" if use_point_a else "TrOCR (greedy)"
    print(f"[3/5] Recognition: {label}")
    t0 = time.time()
    
    raw_lines_with_boxes = []  # list of (text, x1, y1, x2, y2)
    
    for i, box in enumerate(sorted_boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values
        
        if use_point_a:
            # beam-5 with reranker
            outputs = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=5,
                num_return_sequences=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            candidates = processor.batch_decode(outputs, skip_special_tokens=True)
            text = rerank_candidates(candidates)
        else:
            # greedy (default — faster, avoids beam hallucinations)
            outputs = model.generate(pixel_values, max_new_tokens=64, num_beams=1)
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        raw_lines_with_boxes.append((text, x1, y1, x2, y2))
        print(f"      Line {i:2d}: [{x1:4d},{y1:4d}] {text}")
    
    print(f"      Recognition done in {time.time() - t0:.0f}s\n")
    
    # save raw output (checkpoint 1)
    with open(f"{output_dir}/output_1_raw.txt", "w", encoding="utf-8") as f:
        for i, (text, x1, y1, x2, y2) in enumerate(raw_lines_with_boxes):
            f.write(f"Line {i}: [{x1},{y1}] {text}\n")
    
    # ---- STAGE 4: POINT D (LLM RESTRUCTURER) ----
    if use_point_d:
        print(f"[4/5] Point D: LLM document restructuring...")
        t0 = time.time()
        restructured_lines = restructure_document(raw_lines_with_boxes, verbose=True)
        print(f"      Restructured {len(raw_lines_with_boxes)} → "
              f"{len(restructured_lines)} lines in {time.time() - t0:.0f}s\n")
        
        # save (checkpoint 2)
        with open(f"{output_dir}/output_2_restructured.txt", "w", encoding="utf-8") as f:
            for i, line in enumerate(restructured_lines):
                f.write(f"Line {i}: {line}\n")
    else:
        # skip Point D — just extract text without coords
        restructured_lines = [t for t, *_ in raw_lines_with_boxes]
    
    # ---- STAGE 5: POINT B (LLM CORRECTOR) ----
    if use_point_b:
        print(f"[5/5] Point B: LLM line-level correction...")
        t0 = time.time()
        final_lines = correct_lines_batch(restructured_lines, verbose=True)
        print(f"      Corrected {len(restructured_lines)} lines in {time.time() - t0:.0f}s\n")
    else:
        final_lines = restructured_lines
    
    # ---- SAVE FINAL OUTPUT ----
    final_path = f"{output_dir}/output_final.txt"
    with open(final_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(final_lines):
            f.write(f"Line {i}: {line}\n")
    
    # also save a "clean" version without line numbers (for CER measurement)
    clean_path = f"{output_dir}/output_final_clean.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(f"{line}\n")
    
    # ---- SUMMARY ----
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Files in {output_dir}/:")
    print(f"  debug_detection.png       (visual: green boxes on image)")
    print(f"  output_1_raw.txt          (checkpoint: after TrOCR)")
    if use_point_d:
        print(f"  output_2_restructured.txt (checkpoint: after Point D)")
    print(f"  output_final.txt          (final with line numbers)")
    print(f"  output_final_clean.txt    (final without line numbers)")
    print()
    
    # # ---- OPTIONAL: AUTO-EVALUATE ----
    # if GROUND_TRUTH and os.path.exists(GROUND_TRUTH):
    #     try:
    #         from jiwer import cer
    #         with open(GROUND_TRUTH) as f:
    #             truth = " ".join(l.strip() for l in f if l.strip())
            
    #         raw_text = " ".join(t for t, *_ in raw_lines_with_boxes)
    #         final_text = " ".join(final_lines)
            
    #         raw_cer = cer(truth, raw_text)
    #         final_cer = cer(truth, final_text)
            
    #         print(f"CER (raw TrOCR):      {raw_cer:.3f}")
    #         print(f"CER (after pipeline): {final_cer:.3f}")
    #         print(f"Improvement:          {(raw_cer - final_cer) * 100:+.1f}%")
    #     except ImportError:
    #         print("(install `jiwer` for auto-CER measurement)")
    
    # return final_lines


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()