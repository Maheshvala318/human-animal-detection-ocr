"""
Offline Industrial AI Suite: Part A (Detection) & Part B (OCR)
------------------------------------------------------------
Status: Production-Ready Pipeline
Constraints: Offline execution, No YOLO, No Cloud APIs.
"""

import os
import cv2
import torch
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# --------------------------------------------------
# 1. GLOBAL CONFIGURATION & DEVICE SETUP
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "test_images/test1.jpeg" 
MODEL_PATH = "models/classifier.pth"

# --------------------------------------------------
# 2. PART B: OFFLINE OCR PIPELINE (STENCILED TEXT)
# --------------------------------------------------
def run_industrial_ocr(img_path):
    print("\n--- Starting Part B: Industrial OCR Pipeline ---")
    
    # Step 0: Image Loading
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: {img_path} not found.")
        return
    
    # Step 1: Grayscale Conversion
    # Essential for reducing complexity before MSER
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: MSER (Maximally Stable Extremal Regions)
    # Chosen for stenciled text because it handles disconnected character fragments well
    mser = cv2.MSER_create()
    mser.setMinArea(200)
    mser.setMaxArea(8000)
    regions, _ = mser.detectRegions(gray)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        hull = cv2.convexHull(region)
        cv2.fillPoly(mask, [hull], 255)

    # Step 3: Morphological Cleaning
    # Closing gaps in stenciled characters to create solid bounding boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Region Extraction
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 15: # Noise filtering
            boxes.append((x, y, w, h))

    # Step 5: Offline Tesseract Inference
    # Configured for industrial alphanumeric stencils
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    results = []

    for x, y, w, h in boxes:
        crop = gray[y:y+h, x:x+w]
        # Local preprocessing to improve OCR accuracy
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(crop, config=custom_config)
        if text.strip():
            results.append(text.strip())
    
    print(f"OCR Results: {results}")
    return results, gray, boxes

# --------------------------------------------------
# 3. PART A: DETECTION & CLASSIFICATION PIPELINE
# --------------------------------------------------

# Transforms
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
detector_transform = transforms.ToTensor()

def load_models():
    """Loads localization and classification models."""
    print("Loading Detection Models to Device:", device)
    # Stage 1: Detector (Faster R-CNN)
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    detector.to(device).eval()
    
    # Stage 2: Classifier (ResNet18)
    classifier = models.resnet18(pretrained=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.to(device).eval()
    
    return detector, classifier

def run_detection_inference(img_path, detector, classifier):
    print("\n--- Starting Part A: Human/Animal Detection Pipeline ---")
    image = Image.open(img_path).convert("RGB")
    img_tensor = detector_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = detector(img_tensor)[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    img_np = np.array(image)

    for box, score in zip(boxes, scores):
        if score < 0.6: continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = classifier_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(crop_tensor)
            pred = torch.argmax(output, dim=1).item()

        label = "Human" if pred == 1 else "Animal"
        
        # Visualization
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{label} {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_np

# --------------------------------------------------
# 4. MAIN EXECUTION ENTRY POINT
# --------------------------------------------------
def main():
    # Ensure directories exist
    for d in ["outputs", "models", "test_images", "test_videos"]:
        os.makedirs(d, exist_ok=True)

    # 1. Run OCR (Part B)
    ocr_texts, gray_img, ocr_boxes = run_industrial_ocr(image_path)

    # 2. Run Detection (Part A)
    det_model, cls_model = load_models()
    processed_img = run_detection_inference(image_path, det_model, cls_model)

    # 3. Final Visualization / Save
    print("\nProcessing Complete. Saving to ./outputs/")
    output_path = os.path.join("outputs", "final_result.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
    
    plt.figure(figsize=(10,5))
    plt.imshow(processed_img)
    plt.title("Final Pipeline Output")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()