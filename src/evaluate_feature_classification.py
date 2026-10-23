from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "feature_detector_scores.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = {"label", "anomaly_score"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.dropna(subset=["label", "anomaly_score"]).copy()

y_true = df["label"].astype(int).to_numpy()
scores = df["anomaly_score"].astype(float).to_numpy()

if set(np.unique(y_true)) != {0, 1}:
    raise ValueError("Expected both normal (0) and defective (1) images.")

normal_scores = scores[y_true == 0]
defective_scores = scores[y_true == 1]

# Exploratory threshold only: derived from the current test-set boundaries.
# Do not treat this as a validated production threshold.
threshold = float((normal_scores.max() + defective_scores.min()) / 2)

y_pred = (scores >= threshold).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Defective"],
    output_dict=True,
    zero_division=0,
)

auc = roc_auc_score(y_true, scores)
accuracy = accuracy_score(y_true, y_pred)

print("\nVISIONGUARD AI — RESNET18 CLASSIFICATION EVALUATION")
print("=" * 58)
print(f"Images evaluated: {len(df)}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Exploratory threshold: {threshold:.6f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion matrix")
print("Rows = actual class; columns = predicted class")
print("                 Predicted normal | Predicted defective")
print(f"Actual normal:       {tn:>5}       |       {fp:>5}")
print(f"Actual defective:    {fn:>5}       |       {tp:>5}")

print("\nClassification report")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Defective"],
        zero_division=0,
    )
)

summary = pd.DataFrame([
    {"Metric": "Image-level ROC-AUC", "Value": auc},
    {"Metric": "Exploratory threshold", "Value": threshold},
    {"Metric": "Accuracy", "Value": accuracy},
    {"Metric": "Normal precision", "Value": report["Normal"]["precision"]},
    {"Metric": "Normal recall", "Value": report["Normal"]["recall"]},
    {"Metric": "Defective precision", "Value": report["Defective"]["precision"]},
    {"Metric": "Defective recall", "Value": report["Defective"]["recall"]},
    {"Metric": "Defective F1-score", "Value": report["Defective"]["f1-score"]},
    {"Metric": "True negatives", "Value": tn},
    {"Metric": "False positives", "Value": fp},
    {"Metric": "False negatives", "Value": fn},
    {"Metric": "True positives", "Value": tp},
])

summary_path = OUTPUT_DIR / "feature_classification_metrics.csv"
summary.to_csv(summary_path, index=False)

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Defective"],
)
display.plot(ax=ax, colorbar=False, values_format="d")
ax.set_title("ResNet18 Feature Detector — Confusion Matrix")
fig.tight_layout()

plot_path = OUTPUT_DIR / "feature_confusion_matrix.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print(f"\nSaved metrics: {summary_path.relative_to(ROOT)}")
print(f"Saved confusion matrix: {plot_path.relative_to(ROOT)}")
print("\nCaution: threshold is exploratory, not production-validated.")
