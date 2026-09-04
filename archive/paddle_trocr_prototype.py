from paddleocr import TextDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2

# ----- config -----
img_path = "images/bw.png"
output_file = "bw_1.txt"

# ----- reading order sort -----
def sort_reading_order(boxes):
    """Top-to-bottom, left-to-right sort using line grouping."""
    box_data = []
    for box in boxes:
        y1 = min(p[1] for p in box)
        y2 = max(p[1] for p in box)
        x1 = min(p[0] for p in box)
        box_data.append((box, y1, x1, y2 - y1))  # (box, top_y, left_x, height)
    
    # sort by top_y first
    box_data.sort(key=lambda b: b[1])
    
    # group into lines (boxes within ~50% of current line height are same line)
    lines = []
    current_line = [box_data[0]]
    
    for item in box_data[1:]:
        avg_y = sum(b[1] for b in current_line) / len(current_line)
        avg_h = sum(b[3] for b in current_line) / len(current_line)
        
        if abs(item[1] - avg_y) < avg_h * 0.5:
            current_line.append(item)
        else:
            current_line.sort(key=lambda b: b[2])  # sort line by left_x
            lines.append(current_line)
            current_line = [item]
    
    current_line.sort(key=lambda b: b[2])
    lines.append(current_line)
    
    # flatten
    return [item[0] for line in lines for item in line]

# ----- load models -----
print("Loading detection model...")
det = TextDetection()

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# ----- detection -----
print(f"Detecting lines in {img_path}...")
result = det.predict(img_path)
boxes = result[0]['dt_polys']
print(f"Detected {len(boxes)} lines")

# ----- SORT boxes in reading order -----
boxes = sort_reading_order(boxes)

# ----- recognition -----
img = cv2.imread(img_path)

with open(output_file, "w", encoding="utf-8") as f:
    for i, box in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in box]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values
        ids = model.generate(pixel_values, max_new_tokens=64)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        
        print(f"Line {i}: {text}")
        f.write(f"Line {i}: {text}\n")

print(f"\nSaved to {output_file}")