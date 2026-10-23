# VisionGuard-AI
## Deep Learning-Based Industrial Anomaly Detection

VisionGuard-AI is a computer vision prototype that detects anomalous visual patterns in industrial bottle images using a pretrained ResNet18 feature extractor and nearest-neighbor comparison.

## Features

- Upload bottle images for inspection.
- Generate anomaly heatmaps.
- Visualize suspicious regions with localization overlays.
- Evaluate image-level anomaly detection.
- Evaluate pixel-level defect localization.
- View evaluation metrics in an interactive Streamlit dashboard.

## Technology Stack

- Python
- PyTorch and torchvision
- ResNet18
- OpenCV
- NumPy and Pandas
- scikit-learn
- Streamlit
- Pillow
- Matplotlib

## Dataset

This project uses the bottle category of the MVTec Anomaly Detection (AD) dataset. The detector is configured for this category and is not guaranteed to work on other product types.

## How It Works

1. Extract visual features using a pretrained ResNet18.
2. Build a reference memory bank from normal training images.
3. Compare local image features against the normal reference features.
4. Generate anomaly heatmaps and localization overlays.
5. Evaluate the predictions using image-level and pixel-level metrics.

## Preliminary Results

| Metric | Result |
|---|---:|
| Image-level ROC-AUC | 1.0000 |
| Mean per-image pixel ROC-AUC | 0.9748 |
| Overall pixel ROC-AUC | 0.9616 |
| Test images | 83 |

These results are preliminary. The exploratory classification threshold was derived from the same test set used for evaluation, so the reported classification performance is not an independent estimate of real-world performance. Further validation on genuinely unseen data is required.

## Installation

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Run the Application

From the project directory:

    streamlit run app.py

## Project Structure

    VisionGuard-AI/
    ├── app.py
    ├── app_backup.py
    ├── requirements.txt
    ├── src/
    ├── data/
    └── results/

## Limitations

- Currently configured for the MVTec AD bottle category.
- Anomaly scores are not calibrated probabilities.
- The classification threshold is exploratory.
- Independent validation is needed before real-world deployment.
- Heatmap intensity indicates relative feature anomaly, not guaranteed physical damage.

## Future Improvements

- Validate on a separate, independent dataset.
- Calibrate a classification threshold using validation data.
- Evaluate additional product categories.
- Improve robustness to lighting and image quality.
- Measure inference speed and deployment performance.

## Purpose

This project demonstrates an educational prototype for visual anomaly detection and defect localization using deep-learning features.
