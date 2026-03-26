# Multi-Model Face Recognition Evaluation Pipeline 🚀

An end-to-end framework to evaluate and compare the performance of multiple state-of-the-art [InsightFace](https://github.com/deepinsight/insightface) models (ArcFace, MobileNet, etc.) across diverse face datasets using pure **Cosine Similarity** matching.

## Overview
This pipeline systematically extracts face embeddings using exactly one consistent face detector (`SCRFD`) to guarantee fair apples-to-apples comparisons. It processes **Reference Galleries** (Identity Mean Embeddings) and matches **Probe Images** to evaluate the **Rank-1 Closed-Set Accuracy** of 5 different recognition models.

### Evaluated Models:
- `buffalo_l` (ResNet50 / High Accuracy)
- `buffalo_m` (ResNet34 / Medium)
- `buffalo_s` (MobileNetV2 / Lightweight)
- `buffalo_sc` (MobileNet0.25 / Mobile)
- `antelopev2` (ResNet100)

### Evaluated Datasets:
- **ORL (AT&T Faces)**
- **IMDB-Crop**
- **IMFDB** (Indian Movie Face Database)

---

## 🏆 Final Model Comparison (Cosine Similarity Rank-1 Accuracy)

| Model        | ORL Rank-1  | IMDB Rank-1 | IMFDB Rank-1 | Avg Accuracy |
|--------------|:-----------:|:-----------:|:------------:|:------------:|
| **buffalo_l**| 1.0000      | 0.8395      | 0.9228       | **0.9208**   |
| **buffalo_s**| 1.0000      | 0.8202      | 0.8607       | 0.8937       |
| **buffalo_sc**| 1.0000      | 0.8202      | 0.8607       | 0.8937       |
| *buffalo_m*  | 0.0000*     | 0.0000*     | 0.0000*      | 0.0000*      |
| *antelopev2* | 0.0000*     | 0.0000*     | 0.0000*      | 0.0000*      |

*(Note: `buffalo_m` and `antelopev2` omitted from scoring due to uninitialized model weights during the final pipeline run).*

🌟 **Best Model: `buffalo_l` (Avg Accuracy: 0.9208)**

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/avinash064/Face-Recognition.git
   cd Face-Recognition/face_eval_pipeline
   ```
2. Install requirements (GPU support via ONNX execution providers is highly recommended):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The pipeline is intentionally decoupled into two stages to prevent GPU VRAM exhaustion on smaller cards (e.g. RTX 3050 4GB):

### 1. Extract Features
Use the extraction script to process all datasets one model at a time. It will output heavily-compressed `.pkl` files containing the facial embeddings.
```bash
python scripts/extract_all_features.py
```

### 2. Evaluate Cosine Similarity
Load the extracted features, calculate the mean identity representations, and evaluate the cosine similarity against the probe dataset images.
```bash
python scripts/evaluate_similarities.py
```

This will print the metrics table to the terminal and save a detailed JSON hierarchy to `results/similarity_metrics.json`.
