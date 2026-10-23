from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "baseline_results.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required_columns = {"label", "anomaly_score", "category", "filename"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

df = df.dropna(subset=["label", "anomaly_score"]).copy()
df["label"] = df["label"].astype(int)

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("The CSV must contain both normal (0) and defective (1) images.")

y_true = df["label"].to_numpy()
scores = df["anomaly_score"].to_numpy()

auc = roc_auc_score(y_true, scores)

# Exploratory threshold: classify scores at or above the median
# of the saved normal-image scores as defective.
# This is for exploration, not a validated deployment threshold.
normal_scores = df.loc[df["label"] == 0, "anomaly_score"]
threshold = float(normal_scores.median())

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

print("\nVISIONGUARD AI — BASELINE CLASSIFICATION EVALUATION")
print("=" * 55)
print(f"Images evaluated: {len(df)}")
print(f"Normal images: {(y_true == 0).sum()}")
print(f"Defective images: {(y_true == 1).sum()}")
print(f"Image-level ROC-AUC: {auc:.4f}")
print(f"Exploratory anomaly threshold: {threshold:.6f}")

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

summary = pd.DataFrame(
    [
        {"Metric": "Image-level ROC-AUC", "Value": auc},
        {"Metric": "Exploratory threshold", "Value": threshold},
        {"Metric": "True negatives", "Value": tn},
        {"Metric": "False positives", "Value": fp},
        {"Metric": "False negatives", "Value": fn},
        {"Metric": "True positives", "Value": tp},
        {"Metric": "Normal precision", "Value": report["Normal"]["precision"]},
        {"Metric": "Normal recall", "Value": report["Normal"]["recall"]},
        {"Metric": "Defective precision", "Value": report["Defective"]["precision"]},
        {"Metric": "Defective recall", "Value": report["Defective"]["recall"]},
        {"Metric": "Defective F1-score", "Value": report["Defective"]["f1-score"]},
    ]
)

summary_path = OUTPUT_DIR / "baseline_classification_metrics.csv"
summary.to_csv(summary_path, index=False)

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Defective"],
)
display.plot(ax=ax, colorbar=False, values_format="d")
ax.set_title("VisionGuard AI — Baseline Confusion Matrix")
fig.tight_layout()

plot_path = OUTPUT_DIR / "baseline_confusion_matrix.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print(f"\nSaved metrics: {summary_path.relative_to(ROOT)}")
print(f"Saved confusion matrix: {plot_path.relative_to(ROOT)}")
print("\nNote: the threshold is exploratory and has not been validated.")
