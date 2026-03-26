# Face Recognition Evaluation Pipeline using InsightFace 🎯

A production-ready, multi-model face recognition evaluation pipeline using SCRFD face detection, ArcFace embeddings, and pure cosine similarity matching  evaluated across 3 diverse datasets.

## Features

- 🔍 **SCRFD** face detector (loaded **once** across all models)
- 🧠 **5 InsightFace models** evaluated fairly on identical face crops
- 📐 **Identity mean embeddings** (gallery-level representation)
- ⚡ **Vectorized cosine similarity** matching (NumPy)
- 📊 **Multi-dataset evaluation** (ORL, IMDB-Crop, IMFDB)
- 💾 **Decoupled pipeline**: extract embeddings first, match second
- 🗂️ YAML config for reproducibility

---

## Architecture

```
Detection (SCRFD) → Face Crop → Embedding (ArcFace/MobileNet) → Cosine Matching → Rank-1 Accuracy
```

---

## Results

| Model       | ORL Rank-1 | IMDB Rank-1 | IMFDB Rank-1 | Avg Accuracy |
|-------------|:----------:|:-----------:|:------------:|:------------:|
| **buffalo_l**  | 1.0000 | 0.8395 | 0.9228 | **0.9208** ⭐ |
| buffalo_s   |   1.0000   |   0.8202    |    0.8607    |    0.8937    |
| buffalo_sc  |   1.0000   |   0.8202    |    0.8607    |    0.8937    |
| buffalo_m   |   N/A*     |   N/A*      |    N/A*      |    0.0000    |
| antelopev2  |   N/A*     |   N/A*      |    N/A*      |    0.0000    |

> **Best Model: `buffalo_l` (Avg Accuracy: 0.9208)** — ResNet50 ArcFace trained on WebFace600K

*Note: `buffalo_m` and `antelopev2` encountered a model initialization error during this run and are excluded from scoring.*

---

## Repository Structure

```
Face-Recognition/
│
├── face_eval_pipeline/
│   ├── scripts/
│   │   ├── build_imfdb_gallery.py     # Build reference gallery from IMFDB
│   │   ├── build_imdb_wiki_gallery.py # Build reference gallery from IMDB/WIKI
│   │   ├── extract_all_features.py    # Stage 1: Extract all embeddings per model
│   │   └── evaluate_similarities.py   # Stage 2: Cosine matching + metrics table
│   │
│   ├── src/                           # Legacy evaluation pipeline modules
│   │   ├── data_loader.py
│   │   ├── embedding.py
│   │   ├── evaluator.py
│   │   ├── matcher.py
│   │   ├── metrics.py
│   │   ├── model_manager.py
│   │   └── pose_selector.py
│   │
│   └── results/
│       ├── embeddings/                # Saved .pkl embedding files (gitignored)
│       └── similarity_metrics.json    # Final evaluation output
│
├── utils/                             # Shared utilities
│   ├── detection.py                   # SCRFD loader + face selector
│   ├── embedding.py                   # Recognition model loader + extractor
│   ├── matching.py                    # Gallery builder + cosine matching
│   └── evaluation.py                  # Metrics + table printing + JSON saver
│
├── configs/
│   └── paths.yaml                     # Centralized dataset and output paths
│
├── run_pipeline.sh                    # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/avinash064/Face-Recognition.git
cd Face-Recognition

# Install dependencies (GPU support strongly recommended)
pip install -r face_eval_pipeline/requirements.txt
```

> Requires: Python 3.8+, CUDA-compatible GPU (tested on RTX 3050 4GB)

---

## Dataset Setup

Datasets must be downloaded manually and placed in the correct directories as specified in `configs/paths.yaml`.

| Dataset     | Download URL | Expected Path |
|-------------|-------------|--------------|
| AT&T (ORL)  | [AT&T](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) | `face_eval_pipeline/orl_faces/` |
| IMDB-Crop   | [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | `datasets/imdb_wiki/imdb_crop/` |
| IMFDB       | [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/) | `datasets/IMFDB FR dataset/` |

---

## Usage

### Option A: One Command (end-to-end)
```bash
bash run_pipeline.sh
```

### Option B: Manual Step-by-Step

**Step 1 — Build Reference Galleries** *(run once per dataset)*
```bash
python face_eval_pipeline/scripts/build_imfdb_gallery.py
python face_eval_pipeline/scripts/build_imdb_wiki_gallery.py
```

**Step 2 — Extract Embeddings**
```bash
python face_eval_pipeline/scripts/extract_all_features.py
```

**Step 3 — Evaluate**
```bash
python face_eval_pipeline/scripts/evaluate_similarities.py
```

---

## Configuration

Edit `configs/paths.yaml` to point to your dataset directories:

```yaml
datasets:
  orl:
    gallery: "face_eval_pipeline/orl_faces/reference_gallery"
    probe:   "face_eval_pipeline/orl_faces"
  imdb:
    gallery: "datasets/imdb_wiki/reference_gallery_imdb"
    probe:   "datasets/imdb_wiki/imdb_crop"
```

---

## Methodology

1. **Gallery Construction**: The top 5–10 most frontal images per identity are selected using pose scores (yaw + pitch from InsightFace).
2. **Feature Extraction**: SCRFD detects the largest face in each image. All 5 recognition models then extract normalized 512-D embeddings from that same crop.
3. **Identity Representation**: Gallery embeddings are averaged per identity and re-normalized → **Identity Mean Embedding**.
4. **Matching**: Probe embeddings are matched against the gallery using vectorized dot-product cosine similarity.
5. **Evaluation**: Rank-1 accuracy (highest similarity = predicted identity).

---

## Future Work

- [ ] FAISS integration for large-scale retrieval
- [ ] Multi-model embedding fusion
- [ ] ROC curve generation & AUC metrics
- [ ] Threshold tuning for open-set recognition
- [ ] Top-5 accuracy support

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this pipeline, please cite the InsightFace project:
```
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang et al.},
  booktitle={CVPR},
  year={2019}
}
```
