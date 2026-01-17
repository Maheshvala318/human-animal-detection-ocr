## Model Selection and Justification

### Object Detection Model
A pre-trained **Faster R-CNN with ResNet50-FPN backbone** was selected for object detection.

**Justification:**
- Provides strong localization accuracy for humans and animals
- Suitable for offline deployment (no cloud dependency)
- Industry-standard two-stage detector commonly used in safety-critical applications
- Explicitly avoids YOLO, as required by the assignment
- Pre-trained weights reduce training time under limited compute constraints

The detector is used only for **localization**, and its outputs (bounding boxes) are passed to a separate classification model.

---

### Classification Model (Primary)
A **ResNet18-based classifier** was used to classify detected objects as *Human* or *Animal*.

**Justification:**
- Lightweight architecture suitable for real-time inference
- Strong performance with limited training data using transfer learning
- Fast convergence compared to deeper networks
- Well-suited for binary classification tasks
- Widely adopted in production systems

---

# Part B: Offline Industrial OCR for Stenciled Text

This module is designed to extract alphanumeric markings from industrial surfaces, such as shipping containers, military crates, and machinery. It specifically addresses the challenges of "stenciled" fonts, where characters are naturally fragmented, and surfaces may be damaged or faded.

---

## 1. Technical Justification & Approach

### The Problem
Traditional OCR engines often fail on industrial stencils because:
1. **Character Fragmentation:** Stencil "bridges" (gaps in letters like B, O, or A) cause characters to be seen as multiple unrelated blobs.
2. **Low Contrast:** Paint fades over time, blending with the background.
3. **Surface Noise:** Rust, scratches, and dirt can be mistaken for text.

### Our Solution: MSER + Morphological Refinement
To overcome this, we implemented a custom computer vision pipeline before passing data to the OCR engine.

* **MSER (Maximally Stable Extremal Regions):** * **Why:** Unlike standard thresholding, MSER detects regions that stay stable across a range of thresholds. This makes it highly robust to varying lighting and faded paint.
* **Convex Hull & Poly-filling:**
    * **Why:** To "heal" stenciled characters, we calculate the convex hull of detected MSER regions. This closes the gaps (bridges) in the stencil, creating a solid character shape.
* **Morphological Closing:**
    * **Why:** A $3 \times 3$ kernel is used to unify nearby fragments, ensuring that an entire line of text is grouped into a single bounding box rather than individual letter boxes.



---

## 2. OCR Pipeline Steps

The system processes the input in the following sequence:

1.  **Grayscale Conversion:** Reduces the 3-channel BGR input to a single luminance channel to simplify feature detection.
2.  **MSER Region Detection:** Identifies stable intensity blobs likely to be text.
3.  **Mask Generation:** Convex hulls are drawn and filled to create a binary "text map."
4.  **Cleaning:** Morphological closing removes salt-and-pepper noise and bridges gaps.
5.  **Targeted Pre-processing:**
    * **Cubic Upsampling:** Crops are resized by $2\times$ to improve character edge definition.
    * **Otsu Binarization:** Calculates an optimal local threshold for each specific text region.
6.  **Offline Inference:** Tesseract 5.0 processes the cleaned crops using a whitelist (`A-Z, 0-9`) to prevent hallucinations from surface damage.

---

## 3. Installation & Requirements

### System Requirements
* **Python 3.8+**
* **Tesseract OCR Engine:** This is required for the offline extraction.
    * **Ubuntu:** `sudo apt install tesseract-ocr`
    * **Windows:** Install the Tesseract binary and add the installation folder to your `PATH`.

### Python Dependencies
Install the required libraries via the provided `requirements.txt`:

```bash
pip install opencv-python numpy pytesseract matplotlib

## Installation & Requirements

All required Python dependencies are listed in the `requirements.txt` file.

### Install dependencies
```bash
pip install -r requirements.txt
