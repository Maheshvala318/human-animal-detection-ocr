# Datasets Used

Due to large size constraints, datasets are not included in this repository.
They can be downloaded directly from Kaggle using the links below.

## Task A: Human & Animal Detection / Classification

### Human Dataset
- **AI Segment – Human Matting Dataset**
- Link: https://www.kaggle.com/datasets/aisegment/aisegmentcom-matting-human-datasets
- Description: High-quality human images with diverse poses and backgrounds.
- Usage: Used to train and evaluate the human class in the classification model.

### Animal Dataset
- **Animals Detection Images Dataset**
- Link: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset
- Description: Contains multiple animal categories across varied environments.
- Usage: Used to train and evaluate the animal class in the classification model.


## Task B: Offline OCR Dataset

No training dataset was required for Task B.

The OCR system operates using a pre-trained offline OCR engine and is designed
to work directly on industrial or military-style images containing stenciled or
painted text.

### Evaluation Images
- A small set of example images depicting industrial/military boxes
- Characteristics:
  - Faded paint
  - Low contrast
  - Surface damage
  - Stenciled alphanumeric text

### Dataset Usage (Task B)

Task B does not involve training a new OCR model.
Instead, a pre-trained offline OCR engine is used in combination with
custom image preprocessing.

A small set of industrial-style example images was used for testing
and validation of the OCR pipeline.
