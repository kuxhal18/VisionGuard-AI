
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet18, ResNet18_Weights

DATA_DIR = Path("data/mvtec-ad/bottle")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
MAX_MEMORY_FEATURES = 5000

device = torch.device("cpu")
print("Using device:", device)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

feature_extractor = nn.Sequential(
    *list(model.children())[:7]
).to(device).eval()

preprocess = weights.transforms()


def extract_feature_map(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_map = feature_extractor(image_tensor)

    return feature_map.squeeze(0).permute(1, 2, 0).cpu().numpy()


train_paths = sorted((DATA_DIR / "train/good").glob("*.png"))

if not train_paths:
    raise FileNotFoundError(f"No training images found in {DATA_DIR / 'train/good'}")

print("Building normal feature memory bank...")

memory_features = []

for path in train_paths:
    image = Image.open(path).convert("RGB")
    feature_map = extract_feature_map(image)
    memory_features.append(feature_map.reshape(-1, feature_map.shape[-1]))

memory_features = np.concatenate(memory_features, axis=0)

rng = np.random.default_rng(42)

if len(memory_features) > MAX_MEMORY_FEATURES:
    selected = rng.choice(
        len(memory_features),
        size=MAX_MEMORY_FEATURES,
        replace=False
    )
    memory_features = memory_features[selected]

memory_features = memory_features.astype(np.float32)

neighbor_model = NearestNeighbors(
    n_neighbors=1,
    algorithm="brute",
    n_jobs=-1
)
neighbor_model.fit(memory_features)

# Choose a defective example for visualization.
test_dir = DATA_DIR / "test" / "contamination"
test_paths = sorted(test_dir.glob("*.png"))

if not test_paths:
    raise FileNotFoundError(f"No test images found in {test_dir}")

image_path = test_paths[0]
original = Image.open(image_path).convert("RGB")

feature_map = extract_feature_map(original)
height, width, channels = feature_map.shape

distances, _ = neighbor_model.kneighbors(
    feature_map.reshape(-1, channels).astype(np.float32)
)

anomaly_map = distances.reshape(height, width)

anomaly_map = cv2.resize(
    anomaly_map,
    (original.width, original.height),
    interpolation=cv2.INTER_CUBIC
)

anomaly_map = np.maximum(anomaly_map, 0)
anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=4)

low, high = np.percentile(anomaly_map, [1, 99])

if high > low:
    normalized = np.clip((anomaly_map - low) / (high - low), 0, 1)
else:
    normalized = np.zeros_like(anomaly_map)

heatmap = cv2.applyColorMap(
    (normalized * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

original_array = np.array(original)

overlay = cv2.addWeighted(
    original_array, 0.55,
    heatmap, 0.45,
    0
)

output_path = RESULTS_DIR / "feature_anomaly_heatmap.png"

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_array)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title("Feature Anomaly Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Defect Localization Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print("Test image:", image_path)
print("Saved visualization:", output_path)
print("Note: Heatmap highlights feature differences; it is not yet validated defect segmentation.")
