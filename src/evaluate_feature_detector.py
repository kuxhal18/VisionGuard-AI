from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data" / "mvtec-ad" / "bottle"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

from detector import build_detector, analyze_image

TEST_DIR = DATA_DIR / "test"

def main():
    print("Loading the ResNet18 feature-based detector...")
    extractor, preprocess, neighbor = build_detector()

    rows = []
    categories = sorted(
        path.name for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Test categories: {categories}")

    for category in categories:
        category_dir = TEST_DIR / category
        image_paths = sorted(
            path for path in category_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        label = 0 if category == "good" else 1

        for index, path in enumerate(image_paths, start=1):
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    score, _, _, _ = analyze_image(
                        image, extractor, preprocess, neighbor
                    )

                rows.append({
                    "filename": path.name,
                    "category": category,
                    "label": label,
                    "anomaly_score": score,
                })

                if index % 10 == 0 or index == len(image_paths):
                    print(
                        f"Processed {category}: "
                        f"{index}/{len(image_paths)} images"
                    )

            except Exception as exc:
                print(f"WARNING: Skipping {path.name}: {exc}")

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No test images were successfully evaluated.")

    if results["label"].nunique() != 2:
        raise RuntimeError(
            "Both normal and defective images are required to calculate ROC-AUC."
        )

    output_csv = RESULTS_DIR / "feature_detector_scores.csv"
    results.to_csv(output_csv, index=False)

    y_true = results["label"].to_numpy()
    scores = results["anomaly_score"].to_numpy()
    auc = roc_auc_score(y_true, scores)

    print("\nRESNET18 FEATURE DETECTOR RESULTS")
    print("=" * 45)
    print(f"Images evaluated: {len(results)}")
    print(f"Normal images: {(y_true == 0).sum()}")
    print(f"Defective images: {(y_true == 1).sum()}")
    print(f"Image-level ROC-AUC: {auc:.4f}")
    print(f"Saved image scores: {output_csv.relative_to(ROOT)}")

    # Compare with the existing baseline if its results are available.
    baseline_path = ROOT / "data" / "baseline_results.csv"

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)

        if {"filename", "label", "anomaly_score"}.issubset(baseline.columns):
            baseline_auc = roc_auc_score(
                baseline["label"].astype(int),
                baseline["anomaly_score"].astype(float),
            )

            print(f"\nExisting baseline ROC-AUC: {baseline_auc:.4f}")
            print(f"Feature detector ROC-AUC:  {auc:.4f}")
            print(f"AUC difference:            {auc - baseline_auc:+.4f}")

    # Save the ROC curve data and image.
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    curve_df = pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": thresholds,
    })

    curve_csv = RESULTS_DIR / "feature_detector_roc.csv"
    curve_df.to_csv(curve_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ResNet18 feature detector (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("VisionGuard AI — Image-Level ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "feature_detector_roc.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"Saved ROC curve: {plot_path.relative_to(ROOT)}")
    print(f"Saved ROC data:  {curve_csv.relative_to(ROOT)}")
    print("\nEvaluation complete.")
    print(
        "Note: ROC-AUC evaluates ranking performance, not a calibrated "
        "pass/fail decision."
    )

if __name__ == "__main__":
    main()
