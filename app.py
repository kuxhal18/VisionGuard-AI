from pathlib import Path
import sys
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from detector import build_detector, analyze_image

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🔍",
    layout="wide",
)

RESULTS_DIR = ROOT / "results"
LOCALIZATION_CSV = RESULTS_DIR / "localization_summary.csv"
LOCALIZATION_PLOT = RESULTS_DIR / "localization_metrics.png"
CLASSIFICATION_CSV = RESULTS_DIR / "feature_classification_metrics.csv"
CONFUSION_PLOT = RESULTS_DIR / "feature_confusion_matrix.png"
ROC_PLOT = RESULTS_DIR / "feature_detector_roc.png"
BASELINE_CSV = RESULTS_DIR / "baseline_classification_metrics.csv"

st.title("🔍 VisionGuard AI")
st.caption("Deep-learning visual anomaly detection and defect localization")
st.markdown("---")

inspection_tab, evaluation_tab, about_tab = st.tabs(
    ["🔎 Inspection", "📊 Evaluation Dashboard", "ℹ️ About"]
)

# -------------------------------------------------------------------
# INSPECTION TAB
# -------------------------------------------------------------------
with inspection_tab:
    st.markdown(
        """
        Upload a bottle image to compare its local visual features against
        normal bottle images from the MVTec AD training dataset.
        """
    )

    uploaded = st.file_uploader(
        "Upload a bottle image",
        type=["png", "jpg", "jpeg"],
        key="inspection_upload",
    )

    with st.sidebar:
        st.header("Inspection settings")
        st.write("**Model:** ResNet18 feature extractor")
        st.write("**Method:** Nearest-neighbor anomaly detection")
        st.write("**Reference:** Normal bottle training images")
        st.warning(
            "This model is designed for the MVTec bottle dataset. "
            "Results on other product types may not be meaningful."
        )

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")
            st.stop()

        with st.spinner("Loading the model and analyzing image features..."):
            extractor, preprocess, neighbor = build_detector()
            score, heatmap, overlay, anomaly_map = analyze_image(
                image, extractor, preprocess, neighbor
            )

        st.subheader("Inspection results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Anomaly score (95th percentile)", f"{score:.4f}")
        c2.metric("Input image width", f"{image.width} px")
        c3.metric("Input image height", f"{image.height} px")

        a, b, c = st.columns(3)
        with a:
            st.subheader("Original image")
            st.image(image, use_container_width=True)
        with b:
            st.subheader("Feature anomaly heatmap")
            st.image(heatmap, use_container_width=True)
        with c:
            st.subheader("Localization overlay")
            st.image(overlay, use_container_width=True)

        st.info(
            "The heatmap highlights regions that differ from normal training "
            "features. Red regions are relatively more anomalous within this "
            "image; they are not guaranteed to be physical defects. The "
            "anomaly score is not a probability, and a production-ready "
            "pass/fail threshold has not been independently validated."
        )

        overlay_image = Image.fromarray(overlay.astype(np.uint8))
        buffer = BytesIO()
        overlay_image.save(buffer, format="PNG")

        st.download_button(
            "⬇️ Download localization overlay (PNG)",
            data=buffer.getvalue(),
            file_name="visionguard_overlay.png",
            mime="image/png",
        )
    else:
        st.info("Upload a bottle image to begin inspection.")
        st.markdown("#### What this app does")
        st.write("- Extracts visual features using a pretrained ResNet18.")
        st.write("- Compares image patches against normal training features.")
        st.write("- Generates an anomaly heatmap and localization overlay.")

# -------------------------------------------------------------------
# EVALUATION DASHBOARD
# -------------------------------------------------------------------
with evaluation_tab:
    st.header("Model Evaluation Dashboard")
    st.write(
        "Evaluation results for the MVTec AD bottle dataset. These are "
        "benchmark results and do not guarantee performance on new images."
    )

    # Image-level classification
    st.subheader("1. Image-Level Classification")

    if CLASSIFICATION_CSV.exists():
        try:
            clf_df = pd.read_csv(CLASSIFICATION_CSV)
            clf_metrics = {
                str(row["Metric"]): float(row["Value"])
                for _, row in clf_df.iterrows()
            }

            def metric_value(name, fmt=".4f"):
                value = clf_metrics.get(name)
                return "N/A" if value is None else format(value, fmt)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Image-level ROC-AUC",
                metric_value("Image-level ROC-AUC"),
            )
            k2.metric("Accuracy", metric_value("Accuracy"))
            k3.metric(
                "Defective precision",
                metric_value("Defective precision"),
            )
            k4.metric(
                "Defective recall",
                metric_value("Defective recall"),
            )

            k5, k6 = st.columns(2)
            k5.metric(
                "Defective F1-score",
                metric_value("Defective F1-score"),
            )
            k6.metric(
                "Exploratory threshold",
                metric_value("Exploratory threshold", ".6f"),
            )

            st.caption(
                "The displayed threshold was derived from this test set and "
                "is exploratory, not a production-validated decision boundary."
            )

            if CONFUSION_PLOT.exists():
                st.subheader("Confusion Matrix")
                st.image(
                    str(CONFUSION_PLOT),
                    caption="ResNet18 feature detector: image-level predictions",
                    use_container_width=True,
                )

            with st.expander("View classification metrics table"):
                st.dataframe(clf_df, use_container_width=True)

        except (KeyError, ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not read classification metrics: {exc}")
    else:
        st.warning("Classification metrics file was not found.")

    # Baseline comparison
    st.subheader("2. Baseline vs Feature-Based Detector")

    if ROC_PLOT.exists():
        st.image(
            str(ROC_PLOT),
            caption="Image-level ROC curve comparison",
            use_container_width=True,
        )

    if BASELINE_CSV.exists() and CLASSIFICATION_CSV.exists():
        st.write(
            "The feature-based detector achieved an image-level ROC-AUC of "
            "1.0000 on the current 83-image test set. The earlier baseline "
            "achieved 0.7643 on that test set."
        )
        st.caption(
            "This comparison is preliminary. Independent validation and "
            "checks for duplicate or overlapping images are recommended."
        )

    # Pixel-level localization
    st.subheader("3. Pixel-Level Defect Localization")

    if LOCALIZATION_CSV.exists():
        try:
            loc_df = pd.read_csv(LOCALIZATION_CSV)
            loc_metrics = {
                str(row["Metric"]): row["Value"]
                for _, row in loc_df.iterrows()
            }

            mean_auc = float(loc_metrics["Mean per-image pixel ROC-AUC"])
            overall_auc = float(loc_metrics["Overall pixel ROC-AUC"])
            image_count = int(float(loc_metrics["Images evaluated"]))

            p1, p2, p3 = st.columns(3)
            p1.metric("Mean per-image pixel ROC-AUC", f"{mean_auc:.4f}")
            p2.metric("Overall pixel ROC-AUC", f"{overall_auc:.4f}")
            p3.metric("Defective images evaluated", f"{image_count}")

            if LOCALIZATION_PLOT.exists():
                st.image(
                    str(LOCALIZATION_PLOT),
                    caption="Pixel-level localization evaluation",
                    use_container_width=True,
                )

            with st.expander("View localization evaluation data"):
                st.dataframe(loc_df, use_container_width=True)

        except (KeyError, ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Could not read localization metrics: {exc}")
    else:
        st.warning("Localization metrics file was not found.")
        st.code("python src/evaluate_localization.py", language="bash")

    st.warning(
        "Research prototype: test-set performance is not a guarantee of "
        "real-world performance. Calibrate the decision threshold on a "
        "separate validation set before using predictions operationally."
    )

# -------------------------------------------------------------------
# ABOUT TAB
# -------------------------------------------------------------------
with about_tab:
    st.header("About VisionGuard AI")
    st.write(
        "VisionGuard AI is a prototype for deep-learning-based visual "
        "anomaly detection and defect localization."
    )

    st.subheader("Current approach")
    st.write("1. Use a pretrained ResNet18 to extract image features.")
    st.write("2. Compare local features against normal bottle training features.")
    st.write("3. Produce an anomaly map and localization overlay.")
    st.write("4. Evaluate localization against pixel-level ground-truth masks.")

    st.subheader("Technology")
    st.write(
        "Python, PyTorch, torchvision, OpenCV, scikit-learn, NumPy, "
        "Pandas, Pillow, and Streamlit."
    )

    st.subheader("Current limitations")
    st.write("- Currently configured for the MVTec AD bottle category.")
    st.write("- Anomaly scores are not calibrated probabilities.")
    st.write("- The displayed classification threshold is exploratory.")
    st.write("- Independent validation is needed before real-world use.")

st.markdown("---")
st.caption(
    "VisionGuard AI | Computer Vision • Deep Learning • Anomaly Localization"
)
