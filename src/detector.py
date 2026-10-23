from pathlib import Path
import streamlit as st

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mvtec-ad" / "bottle"
DEVICE = torch.device("cpu")
MEMORY_SIZE = 5000

@torch.inference_mode()
def extract_feature_map(image, extractor, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    features = extractor(tensor)
    return features[0].permute(1, 2, 0).cpu().numpy()

@st.cache_resource
def build_detector():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    extractor = nn.Sequential(*list(model.children())[:7]).to(DEVICE).eval()
    preprocess = weights.transforms()

    paths = sorted((DATA_DIR / "train" / "good").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No normal training images found in {DATA_DIR / 'train/good'}")

    feature_maps = []
    for path in paths:
        with Image.open(path) as image:
            feature_maps.append(extract_feature_map(image, extractor, preprocess))

    features = np.concatenate([f.reshape(-1, f.shape[-1]) for f in feature_maps])
    rng = np.random.default_rng(42)
    if len(features) > MEMORY_SIZE:
        indices = rng.choice(len(features), MEMORY_SIZE, replace=False)
        features = features[indices]

    neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    neighbor.fit(features.astype(np.float32))
    return extractor, preprocess, neighbor

def analyze_image(image, extractor, preprocess, neighbor):
    feature_map = extract_feature_map(image, extractor, preprocess)
    height, width, channels = feature_map.shape
    distances, _ = neighbor.kneighbors(
        feature_map.reshape(-1, channels).astype(np.float32)
    )
    anomaly_map = distances.reshape(height, width)

    anomaly_map = cv2.resize(
        anomaly_map,
        (image.width, image.height),
        interpolation=cv2.INTER_CUBIC,
    )
    anomaly_map = np.maximum(anomaly_map, 0)
    low, high = np.percentile(anomaly_map, [1, 99])
    normalized = np.clip((anomaly_map - low) / (high - low + 1e-8), 0, 1)

    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = np.asarray(image.convert("RGB"))
    overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

    score = float(np.percentile(anomaly_map, 95))
    return score, heatmap, overlay, normalized
