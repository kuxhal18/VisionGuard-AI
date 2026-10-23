
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, ResNet18_Weights

DATA_DIR = Path("data/mvtec-ad/bottle")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["broken_small", "broken_large", "contamination"]

device = torch.device("cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

feature_extractor = nn.Sequential(
    *list(model.children())[:7]
).to(device).eval()

preprocess = weights.transforms()


def extract_feature_map(image):
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_map = feature_extractor(tensor)

    return feature_map.squeeze(0).permute(1, 2, 0).cpu().numpy()


def list_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


train_paths = list_images(DATA_DIR / "train" / "good")

if not train_paths:
    raise FileNotFoundError("Normal training images were not found.")

print("Building normal feature memory bank...")

feature_vectors = []

for path in train_paths:
    image = Image.open(path).convert("RGB")
    fmap = extract_feature_map(image)
    feature_vectors.append(fmap.reshape(-1, fmap.shape[-1]))

memory_bank = np.concatenate(feature_vectors, axis=0).astype(np.float32)

rng = np.random.default_rng(42)
max_features = 5000

if len(memory_bank) > max_features:
    indices = rng.choice(len(memory_bank), max_features, replace=False)
    memory_bank = memory_bank[indices]

print("Memory bank size:", len(memory_bank))

# Nearest-neighbor search against normal image features.
from sklearn.neighbors import NearestNeighbors

neighbor_model = NearestNeighbors(
    n_neighbors=1,
    algorithm="brute",
    n_jobs=-1
)
neighbor_model.fit(memory_bank)

all_scores = []
all_labels = []
pixel_aucs = []

for category in CATEGORIES:
    test_dir = DATA_DIR / "test" / category
    mask_dir = DATA_DIR / "ground_truth" / category

    for image_path in list_images(test_dir):
        # MVTec defect masks use the image filename with _mask.png appended.
        mask_path = mask_dir / f"{image_path.stem}_mask.png"

        if not mask_path.exists():
            print("Mask missing, skipping:", mask_path)
            continue

        image = Image.open(image_path).convert("RGB")
        feature_map = extract_feature_map(image)
        height, width, channels = feature_map.shape

        distances, _ = neighbor_model.kneighbors(
            feature_map.reshape(-1, channels).astype(np.float32)
        )

        anomaly_map = distances.reshape(height, width)

        anomaly_map = cv2.resize(
            anomaly_map,
            (image.width, image.height),
            interpolation=cv2.INTER_CUBIC
        )
        anomaly_map = np.maximum(anomaly_map, 0)

        mask = np.array(
            Image.open(mask_path).convert("L").resize(
                image.size, Image.Resampling.NEAREST
            )
        ) > 0

        # Image-level anomaly map vs actual defect pixels.
        scores = anomaly_map.ravel()
        labels = mask.ravel().astype(np.uint8)

        if labels.min() == labels.max():
            continue

        auc = roc_auc_score(labels, scores)
        pixel_aucs.append(auc)

        all_scores.append(scores)
        all_labels.append(labels)

        print(f"{image_path.name}: pixel ROC-AUC = {auc:.4f}")

if not pixel_aucs:
    raise RuntimeError(
        "No valid image-mask pairs were evaluated. Check the ground-truth filenames."
    )

overall_auc = roc_auc_score(
    np.concatenate(all_labels),
    np.concatenate(all_scores)
)

print("\nMean per-image pixel ROC-AUC:", round(float(np.mean(pixel_aucs)), 4))
print("Overall pixel ROC-AUC:", round(float(overall_auc), 4))
print("Images evaluated:", len(pixel_aucs))

import csv
import matplotlib.pyplot as plt

summary_path = RESULTS_DIR / "localization_summary.csv"

with open(summary_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Mean per-image pixel ROC-AUC", round(float(np.mean(pixel_aucs)), 4)])
    writer.writerow(["Overall pixel ROC-AUC", round(float(overall_auc), 4)])
    writer.writerow(["Images evaluated", len(pixel_aucs)])

plt.figure(figsize=(8, 5))
plt.bar(
    ["Mean per-image", "Overall pixel"],
    [float(np.mean(pixel_aucs)), float(overall_auc)]
)
plt.ylim(0, 1)
plt.ylabel("ROC-AUC")
plt.title("VisionGuard-AI: Pixel-Level Localization")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "localization_metrics.png", dpi=200)
plt.close()

print("Saved summary:", summary_path)
print("Saved chart:", RESULTS_DIR / "localization_metrics.png")
