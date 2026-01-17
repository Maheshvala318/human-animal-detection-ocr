"""
Human vs Animal Detection & Classification
-----------------------------------------

This script contains:
1. Dataset preparation and training code (Kaggle-only, disabled by default)
2. Object detection using Faster R-CNN
3. Human/Animal classification using ResNet18
4. Offline inference on images

Training was performed in a Kaggle environment due to dataset size and GPU needs.
The default execution mode is inference only.
"""

import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# --------------------------------------------------
# DEVICE (SAFE FOR ALL MACHINES)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# DATASET UTILITIES (USED DURING TRAINING)
# --------------------------------------------------
def balanced_random_sample(root_dir, max_images=5000, per_folder_limit=100, seed=42):
    random.seed(seed)
    collected = []

    for root, _, files in os.walk(root_dir):
        imgs = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if imgs:
            random.shuffle(imgs)
            collected.extend(imgs[:per_folder_limit])

    random.shuffle(collected)
    return collected[:max_images]


class HumanAnimalDataset(Dataset):
    def __init__(self, image_paths, label, transform=None):
        self.image_paths = image_paths
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

detector_transform = transforms.ToTensor()


# --------------------------------------------------
# OBJECT DETECTION MODEL
# --------------------------------------------------
def load_detector():
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    detector.to(device)
    detector.eval()
    return detector


# --------------------------------------------------
# CLASSIFICATION MODEL
# --------------------------------------------------
def load_classifier(model_path="models/classifier.pth"):
    classifier = models.resnet18(pretrained=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    return classifier


# --------------------------------------------------
# TRAINING CODE (KAGGLE-ONLY, DISABLED BY DEFAULT)
# --------------------------------------------------
RUN_TRAINING = False  # ⚠️ KEEP FALSE FOR GITHUB SUBMISSION

if RUN_TRAINING:
    human_imgs = balanced_random_sample(
        "/kaggle/input/aisegmentcom-matting-human-datasets",
        max_images=5000,
        per_folder_limit=50
    )

    animal_imgs = balanced_random_sample(
        "/kaggle/input/animals-detection-images-dataset",
        max_images=5000,
        per_folder_limit=50
    )

    human_train, human_test = train_test_split(human_imgs, test_size=0.2)
    animal_train, animal_test = train_test_split(animal_imgs, test_size=0.2)

    train_dataset = ConcatDataset([
        HumanAnimalDataset(human_train, 1, classifier_transform),
        HumanAnimalDataset(animal_train, 0, classifier_transform)
    ])

    test_dataset = ConcatDataset([
        HumanAnimalDataset(human_test, 1, classifier_transform),
        HumanAnimalDataset(animal_test, 0, classifier_transform)
    ])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    classifier = models.resnet18(pretrained=True)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    for epoch in range(5):
        classifier.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} Accuracy:", correct / total)

    torch.save(classifier.state_dict(), "models/classifier.pth")
    print("Classifier saved to models/classifier.pth")


# --------------------------------------------------
# INFERENCE PIPELINE (DEFAULT EXECUTION)
# --------------------------------------------------
def run_image_inference(image_path):
    detector = load_detector()
    classifier = load_classifier()

    image = Image.open(image_path).convert("RGB")
    img_tensor = detector_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = detector(img_tensor)[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    img_np = np.array(image)

    for box, score in zip(boxes, scores):
        if score < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = classifier_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.argmax(classifier(crop_tensor), dim=1).item()

        label = "Human" if pred == 1 else "Animal"

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_np, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Detection + Classification Output")
    plt.show()


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------
def main():
    # Example test image (replace with your own)
    test_image_path = "test_images/sample.jpg"
    if os.path.exists(test_image_path):
        run_image_inference(test_image_path)
    else:
        print("No test image found. Please add one to test_images/.")


if __name__ == "__main__":
    main()
