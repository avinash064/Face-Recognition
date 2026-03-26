# Face Recognition Pipeline — Multi-Model + Pose Selection

A production-ready face recognition evaluation pipeline using **InsightFace (ArcFace)**. 
This system evaluates face recognition performance across diverse models and datasets using a novel **pose-diversity based gallery selection** strategy.

---

## 🌟 Key Features

1. **Multi-Model Support:** Easily evaluate and compare multiple models side-by-side.
2. **Pose-Diversity Selection:** Gallery references are selected intelligently to maximize angular diversity (frontal + extreme profiles) instead of random selection.
3. **GPU / CPU Auto-Detection:** Automatically leverages NVIDIA GPUs via ONNX Runtime CUDA Execution Provider if available, with seamless fallback to CPU.
4. **Reproducibility:** Fixed random seeds and deterministic pose selection ensure consistent metrics across runs.
5. **Detailed Metrics:** Computes Rank-1 Accuracy, Top-5 Accuracy, TAR@FAR (0.1%, 1%), AUC, and plots ROC / TAR-FAR curves.

---

## 🚀 Supported Models

| Model Pack | Type | Focus | Size |
| :--- | :--- | :--- | :--- |
| **`buffalo_l`** | Large | Default ArcFace R100. Highest accuracy. | ~300MB |
| **`buffalo_m`** | Medium | Balance of speed and accuracy. | ~100MB |
| **`buffalo_s`** | Small | Lightweight, good for edge devices. | ~40MB |
| **`buffalo_sc`** | Mobile | Very lightweight. Ideal for CPU-only inference. | ~15MB |

*Note: Models are automatically downloaded by InsightFace on first run.*

---

## 🧠 Approach: Pose-Based Reference Selection

To ensure robust identification across varying camera angles, this pipeline does **not** select gallery reference images randomly. 

Instead, it extracts **Yaw**, **Pitch**, and **Roll** for every image, and uses a greedy selection algorithm:
1. **First Reference:** Select the image closest to a frontal face (minimum `|yaw| + |pitch|`).
2. **Subsequent References:** Select the image that has the maximum angular distance from the already selected references.
3. **Result:** The gallery captures the widest possible variety of poses (e.g., 1 frontal, 1 left profile, 1 right profile/tilted).

---

## 🛠️ Setup & Requirements

### Environments

For this evaluation to run efficiently, the system utilizes GPU acceleration if available.

```bash
# Provide virtual environment path or create one
python3 -m venv face_env
source face_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Datasets Supported
1. **AT&T Database of Faces (ORL)** — Small dataset, excellent for debugging.
2. **IMFDB (Indian Movie Face Database)** — High pose variance, illumination changes, and occlusion.
3. **IMDB-WIKI Crop** — Large-scale dataset in-the-wild.

---

## 🏃‍♂️ Usage

The main entry point is `scripts/run_evaluation.py`. 

**1. Evaluate Default Models (`buffalo_l`, `buffalo_sc`) on all datasets:**
```bash
python scripts/run_evaluation.py
```

**2. Evaluate Specific Models on AT&T (Fast):**
```bash
python scripts/run_evaluation.py --dataset att --models buffalo_m,buffalo_s
```

**3. Evaluate All Supported Models on IMFDB:**
```bash
python scripts/run_evaluation.py --dataset imfdb --all_models
```

**4. Specify/Override Dataset Directories:**
```bash
python scripts/run_evaluation.py \
    --att_dir /my/custom/path/att_faces \
    --imfdb_dir /my/custom/path/IMFDB \
    --dataset att
```

### Important Command-Line Arguments:
* `--models`: Comma-separated list of models to run (e.g., `buffalo_l,buffalo_sc`).
* `--all_models`: Run all available variants (`l`, `m`, `s`, `sc`).
* `--device`: Force `cpu` or `cuda` (default is `auto` detection).
* `--n_train`: Number of reference images per identity in the gallery (default: 2).

---

## 📊 Evaluation Metrics & Outputs

For every `{dataset}_{model}` combination, the pipeline produces:

1. **`results/{dataset}_{model}_results.json`**: Detailed metrics (Rank-1, Top-K, TAR, FAR).
2. **`results/{dataset}_{model}_roc.png`**: ROC curve plot.
3. **`results/{dataset}_{model}_tar_far.png`**: TAR vs FAR logarithm plot.

After evaluating multiple models, a comparison is generated:
* **`results/model_comparison.json`**: Side-by-side JSON comparison of all models per dataset.
* **Terminal Output Table**: A clean table summarizing the performance of all tested models.

---

## 💡 Performance Trade-offs & Analysis

### 1. Large vs Edge Models
* **`buffalo_l` (ArcFace R100)**: Excels on complex, in-the-wild datasets (IMFDB, IMDB-WIKI). Very robust against heavy occlusion or severe lighting, but demands GPU acceleration for reasonable processing times on large galleries.
* **`buffalo_sc`**: Drops ~5-15% Rank-1 accuracy on complex datasets compared to `buffalo_l`, but infers significantly faster on CPUs. Ideal for constrained environments.

### 2. Pose Diversity Impact
* Random reference selection often accidentally picks multiple frontal faces, causing extreme profile queries to fail. 
* By enforcing pose-diversity, the gallery acts as a multi-view 3D anchor for the identity, measurably increasing verification confidence (TAR) at strict FARs (0.1%).

### 3. CPU vs GPU
* The pipeline automatically uses `CUDAExecutionProvider` via `onnxruntime-gpu`. 
* On an NVIDIA GPU, extraction speeds for `buffalo_l` increase drastically (~10-20x) over the `CPUExecutionProvider`.
