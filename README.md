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