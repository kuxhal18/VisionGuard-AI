
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("data/mvtec-ad/bottle")
categories = [
    ("train/good", "Training: normal"),
    ("test/good", "Test: normal"),
    ("test/broken_small", "Defect: broken small"),
    ("test/broken_large", "Defect: broken large"),
    ("test/contamination", "Defect: contamination"),
]

fig, axes = plt.subplots(1, len(categories), figsize=(15, 4))

for ax, (folder, title) in zip(axes, categories):
    image_paths = sorted((DATA_DIR / folder).glob("*.png"))

    if not image_paths:
        ax.set_title(f"{title}\nNo PNG found")
        ax.axis("off")
        continue

    image = Image.open(image_paths[0]).convert("RGB")
    ax.imshow(image)
    ax.set_title(f"{title}\n{len(image_paths)} images")
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/dataset_preview.png", dpi=150, bbox_inches="tight")
print("Preview saved to data/dataset_preview.png")
plt.show()
