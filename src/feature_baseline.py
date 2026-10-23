
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, ResNet18_Weights

DATA_DIR = Path("data/mvtec-ad/bottle")
BATCH_SIZE = 8
MAX_MEMORY_FEATURES = 5000

device = torch.device("cpu")
print("Using device:", device)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

feature_extractor = nn.Sequential(
    *list(model.children())[:7]
).to(device).eval()

preprocess = weights.transforms()


def extract_features(paths):
    all_features = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        images = []

        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            images.append(preprocess(image))

        batch = torch.stack(images).to(device)

        with torch.no_grad():
            features = feature_extractor(batch)

        features = features.permute(0, 2, 3, 1)
        features = features.reshape(-1, features.shape[-1])
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


train_paths = sorted((DATA_DIR / "train/good").glob("*.png"))

if not train_paths:
    raise FileNotFoundError(
        f"No training images found in {DATA_DIR / 'train/good'}"
    )

print("Extracting features from normal training images...")
normal_features = extract_features(train_paths)

rng = np.random.default_rng(42)
if len(normal_features) > MAX_MEMORY_FEATURES:
    selected = rng.choice(
        len(normal_features),
        size=MAX_MEMORY_FEATURES,
        replace=False
    )
    normal_features = normal_features[selected]

memory_bank = normal_features.astype(np.float32)

print("Memory bank size:", len(memory_bank))
print("Feature dimensions:", memory_bank.shape[1])

neighbor_model = NearestNeighbors(
    n_neighbors=1,
    algorithm="brute",
    n_jobs=-1
)
neighbor_model.fit(memory_bank)

categories = {
    "good": 0,
    "broken_small": 1,
    "broken_large": 1,
    "contamination": 1,
}

image_scores = []
image_labels = []

print("\nEvaluating test images...")

for category, label in categories.items():
    paths = sorted((DATA_DIR / "test" / category).glob("*.png"))

    if not paths:
        raise FileNotFoundError(
            f"No test images found in {DATA_DIR / 'test' / category}"
        )

    for path in paths:
        features = extract_features([path]).astype(np.float32)

        distances, _ = neighbor_model.kneighbors(features)

        # Use the most anomalous 10% of local features for the image score.
        local_distances = distances.ravel()
        score = float(np.percentile(local_distances, 90))

        image_scores.append(score)
        image_labels.append(label)

    print(f"Processed {category}: {len(paths)} images")

auc = roc_auc_score(image_labels, image_scores)

print("\nFeature-based baseline ROC-AUC:", round(auc, 4))
print("Pixel-based baseline ROC-AUC: 0.7643")
