
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

DATA_DIR = Path("data/mvtec-ad/bottle")
IMAGE_SIZE = (128, 128)

def load_image(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0

# Use normal training images to build the reference
train_paths = sorted((DATA_DIR / "train/good").glob("*.png"))
train_images = np.stack([load_image(path) for path in train_paths])
normal_reference = train_images.mean(axis=0)

# Select one defective image
test_path = sorted((DATA_DIR / "test/contamination").glob("*.png"))[0]
image = load_image(test_path)

# Calculate pixel-level difference
difference = np.abs(image - normal_reference).mean(axis=2)
heatmap = cv2.GaussianBlur(difference, (5, 5), 0)

# Normalize for visualization only
heatmap_display = heatmap / (heatmap.max() + 1e-8)
colored_heatmap = cv2.applyColorMap(
    np.uint8(heatmap_display * 255),
    cv2.COLORMAP_JET
)
colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

overlay = np.uint8(image * 255) * 0.55 + colored_heatmap * 0.45
overlay = np.clip(overlay, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image)
axes[0].set_title("Original image")
axes[1].imshow(heatmap_display, cmap="jet")
axes[1].set_title("Anomaly heatmap")
axes[2].imshow(overlay)
axes[2].set_title("Heatmap overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/anomaly_preview.png", dpi=150, bbox_inches="tight")
print("Saved: data/anomaly_preview.png")
plt.show()
