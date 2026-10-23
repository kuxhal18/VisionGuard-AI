
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

DATA_DIR = Path("data/mvtec-ad/bottle")
OUTPUT_DIR = Path("data")
IMAGE_SIZE = (128, 128)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0

# Build a reference using normal training images only
train_paths = sorted((DATA_DIR / "train/good").glob("*.png"))

if not train_paths:
    raise FileNotFoundError("No training images found.")

train_images = np.stack([load_image(path) for path in train_paths])
normal_reference = train_images.mean(axis=0)

# Evaluate each test image individually
categories = {
    "good": 0,
    "broken_small": 1,
    "broken_large": 1,
    "contamination": 1,
}

results = []

for category, label in categories.items():
    image_paths = sorted(
        (DATA_DIR / "test" / category).glob("*.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG images found for category: {category}"
        )

    for path in image_paths:
        image = load_image(path)

        difference = np.abs(image - normal_reference).mean(axis=2)
        difference = cv2.GaussianBlur(
            difference, (5, 5), 0
        )

        score = float(difference.mean())

        results.append({
            "filename": path.name,
            "category": category,
            "label": label,
            "anomaly_score": score,
        })

labels = np.array([row["label"] for row in results])
scores = np.array([row["anomaly_score"] for row in results])

if len(np.unique(labels)) < 2:
    raise ValueError("Both normal and defective test images are required.")

auc = roc_auc_score(labels, scores)

print("Training images:", len(train_paths))
print("Test images:", len(results))
print("Normal test images:", int((labels == 0).sum()))
print("Defective test images:", int((labels == 1).sum()))
print(f"Image-level ROC-AUC: {auc:.4f}")

# Save individual predictions for later analysis
csv_path = OUTPUT_DIR / "baseline_results.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Saved individual scores:", csv_path)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Baseline AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", label="Random ranking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VisionGuard AI: Baseline ROC Curve")
plt.legend()
plt.tight_layout()

plot_path = OUTPUT_DIR / "baseline_roc.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("Saved ROC curve:", plot_path)
