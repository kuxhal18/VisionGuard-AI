# VisionGuard-AI
### Deep Learning-Based Industrial Anomaly Detection & Defect Localization

<p align="center">
  <strong>An AI-powered computer vision system for identifying visual anomalies in industrial bottle images.</strong>
</p>

---

## 📌 Project Overview

**VisionGuard-AI** is a deep learning-based computer vision project designed to identify unusual visual patterns and potential defects in industrial bottle images.

In manufacturing industries, quality inspection is essential for ensuring product reliability, reducing waste, and maintaining production standards. Traditional manual inspection can be time-consuming, inconsistent, and difficult to scale.

VisionGuard-AI explores how computer vision and deep learning can support automated visual inspection by learning patterns from normal product images and identifying deviations from those patterns.

The system uses a **pretrained ResNet18 feature extractor** and a **nearest-neighbor anomaly detection approach** to compare input images against a reference memory bank built from normal training images.

It generates anomaly heatmaps and localization overlays to help visualize regions that differ from the learned normal patterns. An interactive Streamlit application provides a user interface for inspecting images and exploring evaluation results.

> **Project Type:** Computer Vision | Deep Learning | Industrial AI | Unsupervised Anomaly Detection  
> **Primary Application:** Industrial Visual Quality Inspection  
> **Dataset:** MVTec Anomaly Detection (AD) — Bottle Category  
> **Current Status:** Educational Research Prototype

---

## 🎯 Problem Statement

Industrial manufacturing processes can produce products with visual defects such as:

- Cracks and fractures
- Surface irregularities
- Contamination
- Unexpected structural changes
- Other deviations from normal product appearance

Manually identifying every defect requires time and consistent human attention. Furthermore, collecting sufficient examples of every possible defect can be challenging.

This project investigates an alternative approach: learning visual representations from normal product images and using them as a reference for identifying potentially anomalous patterns.

### Our Approach

VisionGuard-AI extracts deep visual features from normal bottle images and stores them in a reference memory bank. During inspection, it compares features from a new image against this reference to estimate how unusual different regions appear.

The resulting anomaly scores are used to generate visual heatmaps and support image-level and pixel-level evaluation.

---

## 🚀 Key Features

### 1. AI-Based Image Inspection
Inspect bottle images using a deep learning-based anomaly detection pipeline.

### 2. Pretrained Deep Feature Extraction
Use ResNet18, a convolutional neural network pretrained on ImageNet, to extract meaningful visual representations from images.

### 3. Normal-Image Reference Memory Bank
Build a reference bank from normal training images to represent the visual characteristics of defect-free bottles.

### 4. Nearest-Neighbor Anomaly Detection
Compare local image features against the normal reference features to estimate the degree of visual deviation.

### 5. Anomaly Heatmap Generation
Generate heatmaps that highlight regions with relatively high anomaly scores.

### 6. Defect Localization Visualization
Overlay anomaly heatmaps on input images to make potentially suspicious regions easier to inspect.

### 7. Image-Level Evaluation
Evaluate how effectively anomaly scores distinguish normal images from anomalous images using the Receiver Operating Characteristic Area Under the Curve (ROC-AUC).

### 8. Pixel-Level Evaluation
Evaluate the spatial quality of anomaly maps using pixel-level ROC-AUC metrics.

### 9. Interactive Streamlit Dashboard
Provide an accessible interface for image inspection and visualization of evaluation results.

### 10. Modular Python Implementation
Organize the project into application, source-code, dataset, and results components to support experimentation and future development.

---

## 🧠 How VisionGuard-AI Works

The system follows a feature-based anomaly detection pipeline.

### Step 1: Input Image

A bottle image is supplied to the system for inspection.

The image is prepared for the feature extraction model using the required preprocessing operations.

### Step 2: Deep Feature Extraction

The image is passed through a pretrained ResNet18 network.

Instead of using the network directly as a conventional defect classifier, the system uses its intermediate visual representations to capture patterns such as:

- Edges and contours
- Surface textures
- Shapes and structures
- Local visual characteristics

These representations provide a feature space in which normal and unusual visual patterns can be compared.

### Step 3: Build the Normal Reference Memory Bank

Normal bottle images from the training dataset are processed through the feature extractor.

Their local visual features are collected to form a reference memory bank representing normal product appearance.

This reference bank acts as the baseline against which new images are compared.

### Step 4: Nearest-Neighbor Feature Comparison

Features extracted from an inspection image are compared with the normal reference features.

The nearest-neighbor approach estimates how closely each local feature matches the learned normal patterns.

Features that are less similar to the reference may receive higher anomaly scores.

**Important:** A high anomaly score indicates an unusual feature pattern, not definitive proof of a physical defect.

### Step 5: Generate Anomaly Maps

Local anomaly scores are arranged spatially to form an anomaly map.

The map indicates which image regions differ most strongly from the normal reference patterns.

The anomaly map is converted into a heatmap for easier interpretation.

### Step 6: Visualize Potentially Defective Regions

The heatmap is overlaid on the original bottle image.

This produces a localization visualization that helps users examine potentially suspicious areas.

The overlay is intended to support inspection rather than replace expert judgment.

### Step 7: Evaluate the Detector

The system's predictions are compared with the available dataset labels and ground-truth defect masks.

Evaluation includes image-level anomaly detection and pixel-level anomaly localization.

### Step 8: Interactive Application

The Streamlit application provides a convenient interface for interacting with the inspection pipeline and viewing supported visualizations and evaluation results.

---

## 🏗️ System Architecture

```text
                 NORMAL TRAINING IMAGES
                           |
                           v
                 Image Preprocessing
                           |
                           v
                Pretrained ResNet18
                  Feature Extractor
                           |
                           v
               Local Feature Extraction
                           |
                           v
             Normal Reference Memory Bank
                           |
                           |
                           v
INPUT BOTTLE IMAGE --> Preprocessing
                           |
                           v
                  ResNet18 Feature
                      Extraction
                           |
                           v
               Nearest-Neighbor Search
                           |
                           v
                  Anomaly Score Map
                           |
                           v
                Heatmap Generation
                           |
                           v
              Localization Visualization
                           |
                           v
                 Inspection Results
                           |
                           v
                 Streamlit Interface
```

*Conceptual overview of the feature-based anomaly detection pipeline.*

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning operations and model execution |
| Torchvision | Pretrained ResNet18 and image transformations |
| ResNet18 | Deep visual feature extraction |
| OpenCV | Image processing and visualization support |
| NumPy | Numerical computation and array operations |
| Pandas | Data organization and metric reporting |
| scikit-learn | Evaluation metrics and performance analysis |
| Streamlit | Interactive web application |
| Pillow | Image loading and manipulation |
| Matplotlib | Heatmap generation and visualization |

---

## 📂 Dataset

VisionGuard-AI uses the **bottle category of the MVTec Anomaly Detection (AD) dataset**, a dataset developed for research into industrial anomaly detection.

The selected category contains normal bottle images as well as images representing different types of anomalies.

The dataset structure includes:

- Normal training images
- Normal test images
- Anomalous test images
- Ground-truth masks for applicable anomalous samples

### Dataset Organization

The bottle category is organized around directories such as:

```text
data/
└── mvtec-ad/
    └── bottle/
        ├── train/
        │   └── good/
        ├── test/
        │   ├── good/
        │   ├── broken_small/
        │   ├── broken_large/
        │   └── contamination/
        └── ground_truth/
            ├── broken_small/
            ├── broken_large/
            └── contamination/
```

*Directory names may vary depending on how the dataset was downloaded or organized locally.*

### Why This Dataset?

The dataset provides normal examples, anomalous examples, and pixel-level ground-truth masks, making it useful for evaluating both anomaly detection and defect localization.

**Scope:** The current implementation is configured for the bottle category. Performance on other product categories has not been established.

---

## 📊 Preliminary Evaluation Results

The current experiment was evaluated on **83 test images**.

| Evaluation Metric | Result |
|---|---:|
| Image-Level ROC-AUC | 1.0000 |
| Mean Per-Image Pixel ROC-AUC | 0.9748 |
| Overall Pixel ROC-AUC | 0.9616 |
| Number of Test Images | 83 |

### 1. Image-Level ROC-AUC — 1.0000

Image-level ROC-AUC measures how well the anomaly scores rank anomalous images above normal images across different decision thresholds.

A score of 1.0000 indicates perfect separation on the evaluated test set according to the metric used in this experiment.

However, this result should not be interpreted as proof of perfect real-world performance.

### 2. Mean Per-Image Pixel ROC-AUC — 0.9748

This metric calculates pixel-level ROC-AUC for individual images and averages the results across the evaluated images.

It measures how effectively anomaly-map scores distinguish defective pixels from normal pixels within individual images.

### 3. Overall Pixel ROC-AUC — 0.9616

This metric evaluates pixel-level anomaly localization across the evaluated dataset using the overall collection of pixel predictions and ground-truth labels.

It provides a complementary view of localization performance.

### Important Evaluation Disclaimer

**These results are preliminary and specific to the current experimental setup.**

The exploratory image-classification threshold was derived using the same test set used for evaluation. Consequently, threshold-dependent classification performance may be optimistic and should not be treated as an independent estimate of generalization.

ROC-AUC measures ranking performance across thresholds; it does not, by itself, establish that a particular classification threshold is appropriate.

Before considering practical deployment, the system should be evaluated using an independent validation strategy, a threshold selected without using the final test labels, and additional genuinely unseen data.

Results may also vary with preprocessing, dataset composition, feature extraction settings, and implementation details.

---

## 📈 Evaluation Methodology

VisionGuard-AI considers two complementary evaluation levels.

### Image-Level Anomaly Detection

**Objective:** Determine whether an image is normal or anomalous.

The anomaly score for an image is compared with its ground-truth image label.

ROC-AUC measures the model's ability to rank anomalous images above normal images across possible thresholds.

### Pixel-Level Defect Localization

**Objective:** Determine whether the anomaly map highlights the correct spatial regions.

Predicted anomaly maps are compared with the dataset's ground-truth defect masks.

Pixel-level ROC-AUC evaluates the ability of the anomaly scores to distinguish anomalous pixels from normal pixels.

### Future Evaluation Metrics

Further evaluation could include:

- Precision and recall
- F1-score
- Confusion matrix
- Precision-recall AUC
- Image-level false-positive rate
- Pixel-level precision and recall
- Per-defect-category performance
- Inference latency and throughput

These metrics would provide a more comprehensive understanding of the detector's practical strengths and limitations.

---

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the selected dependencies
- pip
- Git
- Sufficient memory and storage for the dataset and model operations

The exact Python and library versions should be checked against the project's `requirements.txt`.

### Step 1: Clone the Repository

Replace `<your-github-username>` with your GitHub username.

```bash
git clone https://github.com/<your-github-username>/VisionGuard-AI.git
cd VisionGuard-AI
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Prepare the Dataset

Download the MVTec AD dataset from its official source and place the bottle category in the directory expected by the project.

Confirm that the training images, test images, and ground-truth masks are available in the expected locations.

Dataset files may need to be downloaded separately and may not be included in the GitHub repository.

---

## ▶️ Running the Application

From the project root directory, activate the virtual environment and execute:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to access the application.

The exact interface and available controls depend on the current implementation in `app.py`.

### Typical Inspection Workflow

1. Launch the application.
2. Upload a supported bottle image if the interface provides image upload.
3. Run the inspection workflow.
4. Examine the generated anomaly heatmap.
5. Review the localization overlay.
6. Interpret highlighted regions as potential anomalies requiring further inspection.

The model's anomaly score is not a calibrated probability of a defect.

---

## 📁 Project Structure

```text
VisionGuard-AI/
│
├── app.py
├── app_backup.py
├── requirements.txt
│
├── src/
│   └── Core implementation modules
│
├── data/
│   └── mvtec-ad/
│       └── bottle/
│
├── results/
│   └── Evaluation outputs and visualizations
│
├── .gitignore
└── README.md
```

*This is a high-level representation. The exact contents of `src/` and `results/` depend on the files included in the repository.*

### Main Components

| Component | Responsibility |
|---|---|
| `app.py` | Main Streamlit application |
| `app_backup.py` | Backup version of the application |
| `requirements.txt` | Python dependency definitions |
| `src/` | Core model, processing, and evaluation code |
| `data/` | Dataset storage |
| `results/` | Experimental outputs and visualizations |
| `README.md` | Project documentation |

---

## 🔬 Technical Concepts Demonstrated

This project demonstrates the practical integration of several important computer vision and deep learning concepts.

### Transfer Learning

Using a pretrained ResNet18 avoids training an entire convolutional neural network from scratch.

The network's learned visual representations are reused for a different task: identifying anomalous patterns.

### Feature Embeddings

An image is represented by numerical feature vectors rather than being analyzed only as raw pixel values.

These vectors encode visual information that can be compared computationally.

### Nearest-Neighbor Search

The detector compares inspection features against reference features from normal training images.

Feature distances help estimate how unusual a local region appears relative to the reference bank.

### Anomaly Detection

Instead of relying exclusively on a classifier trained with examples of every defect, the approach models normal appearance and searches for deviations.

This is useful in settings where examples of every possible defect may not be available.

### Anomaly Localization

The system produces spatial anomaly scores, allowing users to inspect where unusual patterns occur rather than receiving only a single image-level score.

### Model Evaluation

The project uses ROC-AUC at both image and pixel levels to evaluate detection and localization performance.

Together, these concepts provide a foundation for further exploration of industrial computer vision systems.

---

## ⚠️ Limitations and Responsible Use

VisionGuard-AI is an educational prototype, not a production-certified industrial inspection system.

### Current Limitations

- **Limited product coverage:** The implementation is configured for bottle images from the selected MVTec AD category.
- **No universal defect guarantee:** Unseen defect types may not be detected reliably.
- **Uncalibrated anomaly scores:** Scores are relative measures of feature deviation, not probabilities of failure.
- **Exploratory threshold:** The current classification threshold requires independent validation.
- **Dataset-specific evaluation:** The reported results do not establish performance on real factory production lines.
- **Sensitivity to image conditions:** Lighting, orientation, reflections, background changes, and image quality may affect predictions.
- **Potential false positives:** Normal visual variations may be highlighted as anomalies.
- **Potential false negatives:** Subtle or unfamiliar defects may not receive sufficiently high anomaly scores.
- **Localization limitations:** Highlighted regions may not precisely match the physical boundaries of a defect.
- **Deployment requirements:** Runtime, memory consumption, and production throughput require separate measurement.

### Intended Use

The system is intended for academic learning, experimentation, demonstrations, and research into deep-learning-based anomaly detection.

It should not be used as the sole quality-control mechanism in a real industrial environment without rigorous validation, monitoring, and appropriate human oversight.

---

## 🔮 Future Improvements

VisionGuard-AI can be extended in several directions.

### 1. Independent Validation

Introduce a validation dataset separate from the final test set.

Choose classification thresholds using validation data and report final performance on untouched test data.

### 2. Multi-Category Detection

Extend the pipeline to additional MVTec AD categories, evaluating each category separately rather than assuming one configuration generalizes to every product.

### 3. Improved Feature Representations

Experiment with alternative pretrained architectures and feature extraction strategies to determine whether they improve anomaly detection and localization.

### 4. Efficient Nearest-Neighbor Search

Explore memory-bank compression, feature sampling, and approximate nearest-neighbor methods to reduce memory consumption and improve inference speed.

### 5. Robustness Testing

Evaluate how image quality, lighting variation, rotations, and other realistic changes affect performance.

### 6. Comprehensive Evaluation

Report precision, recall, F1-score, confusion matrices, and false-positive rates alongside ROC-AUC.

### 7. Performance Benchmarking

Measure inference latency, memory usage, and throughput on the intended hardware.

### 8. Explainability Improvements

Improve the presentation of anomaly maps and investigate techniques that help users understand which visual patterns contribute to high anomaly scores.

### 9. Deployment

Explore packaging the validated pipeline into a deployable application or API, with appropriate input validation, error handling, and monitoring.

---

## 🎓 Learning Outcomes

Developing VisionGuard-AI provides practical experience with:

- Building computer vision applications in Python
- Using PyTorch and pretrained deep learning models
- Applying transfer learning to a visual inspection problem
- Extracting and comparing image features
- Implementing a feature-based anomaly detection pipeline
- Generating anomaly heatmaps and localization overlays
- Working with industrial anomaly detection datasets
- Evaluating image-level and pixel-level predictions
- Building an interactive Streamlit application
- Organizing a machine learning project for reproducibility
- Identifying evaluation leakage and limitations in experimental results
- Planning improvements for a more robust machine learning system

---

## 🌍 Real-World Relevance

Automated visual inspection is relevant to manufacturing environments in which products must be checked for visual inconsistencies.

Related application areas include:

- Bottle and container inspection
- Packaging quality assurance
- Surface-defect analysis
- Component inspection
- Manufacturing process monitoring
- Research into automated quality-control systems

VisionGuard-AI demonstrates a possible technical foundation for these applications, but its current implementation and evaluation are limited to the selected bottle dataset.

Additional data, validation, engineering, and testing would be necessary before applying the system to a real production environment.

---

## 🤝 Contributions

Contributions and suggestions are welcome.

Potential areas for contribution include:

- Improving anomaly detection performance
- Adding support for additional product categories
- Enhancing visualization and usability
- Improving testing and evaluation
- Optimizing inference performance
- Improving documentation and reproducibility

For substantial changes, consider opening an issue to discuss the proposed improvement before submitting a pull request.

---

## 📜 Dataset and References

**Dataset:** MVTec Anomaly Detection (AD)

Official dataset website:

https://www.mvtec.com/company/research/datasets/mvtec-ad

**Model:** ResNet18

PyTorch model documentation:

https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

**Framework:** PyTorch

https://pytorch.org/

**Application Framework:** Streamlit

https://streamlit.io/

Please consult the dataset's official terms and conditions before redistributing dataset images or using the dataset in other contexts.

---

## 👨‍💻 Project Information

**Project Name:** VisionGuard-AI  
**Domain:** Artificial Intelligence and Machine Learning  
**Specialization:** Computer Vision and Deep Learning  
**Application Area:** Industrial Anomaly Detection  
**Interface:** Streamlit  
**Project Purpose:** Educational and experimental

---

## ⭐ Final Note

VisionGuard-AI explores how pretrained deep learning features and nearest-neighbor comparisons can be used to identify unusual visual patterns in industrial bottle images.

By combining feature extraction, reference-memory comparison, anomaly heatmaps, localization visualization, and quantitative evaluation, the project demonstrates an end-to-end approach to an important computer vision problem.

The current results are preliminary and should be interpreted within the scope of the evaluated dataset. Independent validation and further engineering are essential steps toward establishing reliable real-world performance.

**VisionGuard-AI — Making visual anomalies easier to identify, understand, and investigate.**
