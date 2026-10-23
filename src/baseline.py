
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

DATA_DIR = Path("data/mvtec-ad/bottle")
IMAGE_SIZE = (128, 128)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0

# Load normal training images
train_paths = sorted((DATA_DIR / "train/good").glob("*.png"))

if not train_paths:
    raise FileNotFoundError("No training images found.")

train_images = np.stack([load_image(path) for path in train_paths])

# Learn the average appearance of normal bottles
normal_reference = train_images.mean(axis=0)

print("Training images:", len(train_paths))
print("Reference image shape:", normal_reference.shape)
print("\nTesting images...")

test_folders = [
    "test/good",
    "test/broken_small",
    "test/broken_large",
    "test/contamination",
]

for folder in test_folders:
    paths = sorted((DATA_DIR / folder).glob("*.png"))

    if not paths:
        print(f"{folder}: no images found")
        continue

    scores = []

    for path in paths:
        image = load_image(path)

        # Pixel-wise difference from the normal reference
        difference = np.abs(image - normal_reference).mean(axis=2)

        # Smooth small pixel variations
        difference = cv2.GaussianBlur(difference, (5, 5), 0)

        # Higher score means greater visual difference
        scores.append(float(difference.mean()))

    print(
        f"{folder}: {len(paths)} images, "
        f"average anomaly score = {np.mean(scores):.4f}"
    )
