"""
build_imdb_wiki_gallery.py

Select top 5–10 frontal images per identity from IMDB and WIKI datasets
using InsightFace (GPU-safe). Processes the metadata from .mat files.

Usage:
    python build_imdb_wiki_gallery.py
"""

import cv2
import math
import shutil
import numpy as np
import scipy.io
from pathlib import Path
from insightface.app import FaceAnalysis

# ── Config ────────────────────────────────────────────────────────────────

# Paths to datasets
DATASETS = [
    {
        "name": "imdb",
        "root": Path("/home/avinash/datasets/imdb_wiki/imdb_crop"),
        "mat_file": Path("/home/avinash/datasets/imdb_wiki/imdb_crop/imdb.mat"),
        "out_dir": Path("/home/avinash/datasets/imdb_wiki/reference_gallery_imdb")
    },
    {
        "name": "wiki",
        "root": Path("/home/avinash/datasets/imdb_wiki/wiki_crop"),
        "mat_file": Path("/home/avinash/datasets/imdb_wiki/wiki_crop/wiki.mat"),
        "out_dir": Path("/home/avinash/datasets/imdb_wiki/reference_gallery_wiki")
    }
]

MIN_K = 5
MAX_K = 10
FACE_SCORE_THRESH = 1.0

# ── INIT MODEL (GPU SAFE) ─────────────────────────────────────────────────
app = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(320, 320))  # safer for RTX 3050


# ── SCORING (using pose from InsightFace) ─────────────────────────────────
def score_face(face):
    if face.pose is None:
        return 0.0

    yaw, pitch, roll = face.pose

    # frontal preference
    yaw_score = max(0.0, 1 - abs(yaw) / 90) * 100
    pitch_score = max(0.0, 1 - abs(pitch) / 90) * 100

    return round(0.5 * yaw_score + 0.5 * pitch_score, 2)


def score_image(path: Path):
    if not path.exists():
        return path, 0.0

    img = cv2.imread(str(path))
    if img is None:
        return path, 0.0

    faces = app.get(img)
    if len(faces) == 0:
        return path, 0.0

    # pick largest face
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

    if face.det_score < 0.6:
        return path, 0.0

    score = score_face(face)
    return path, score


# ── MAIN ──────────────────────────────────────────────────────────────────
def process_dataset(ds_info):
    mat_path = ds_info["mat_file"]
    dataset_name = ds_info["name"]
    root_dir = ds_info["root"]
    gallery_dir = ds_info["out_dir"]
    
    if not mat_path.exists():
        print(f"\nSkipping {dataset_name.upper()}, mat file not found: {mat_path}")
        return

    print(f"\n{'='*60}")
    print(f"--- Processing {dataset_name.upper()} ---")
    print(f"Loading {mat_path}...")
    
    mat = scipy.io.loadmat(str(mat_path))
    struct = mat[dataset_name][0, 0]
    
    full_paths = struct['full_path'][0]
    names = struct['name'][0]
    face_scores = struct['face_score'][0]
    second_face_scores = struct['second_face_score'][0]
    
    print("Filtering metadata...")
    identities = {} # map identity name to list of valid paths
    
    num_entries = len(full_paths)
    valid_count = 0
    
    for i in range(num_entries):
        path_arr = full_paths[i]
        name_arr = names[i]
        
        # Check if arrays are valid lengths
        if len(path_arr) == 0 or len(name_arr) == 0:
            continue
            
        rel_path = str(path_arr[0])
        val = name_arr[0]
        
        if isinstance(val, np.ndarray) and len(val) > 0:
             identity_name = str(val[0]).strip()
        else:
             identity_name = str(val).strip()
        
        # Avoid empty names
        if not identity_name:
            continue
            
        # MATLAB mat loader can return NaNs or Infinities
        f_score = float(face_scores[i])
        sf_score = float(second_face_scores[i])
        
        # Must have good face score and NOT have a valid second face score
        if math.isinf(f_score) or math.isnan(f_score) or f_score < FACE_SCORE_THRESH:
            continue
            
        if not math.isnan(sf_score):
            continue
            
        abs_path = root_dir / rel_path
        
        if identity_name not in identities:
            identities[identity_name] = []
        identities[identity_name].append(abs_path)
        valid_count += 1
        
    print(f"Found {valid_count} valid images across {len(identities)} identities.")
    
    # Filter identities with enough images
    valid_identities = {k: v for k, v in identities.items() if len(v) >= MIN_K}
    print(f"Identities with >= {MIN_K} images: {len(valid_identities)}")
    
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    skipped = 0
    
    identity_names = sorted(list(valid_identities.keys()))
    
    for i, id_name in enumerate(identity_names, 1):
        images = valid_identities[id_name]
        
        # Score images
        scored = []
        for path in images:
            if not path.exists():
                continue
            scored.append(score_image(path))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        # Select valid scored images (score > 0)
        selected = [x for x in scored if x[1] > 0][:MAX_K]
        
        if len(selected) < MIN_K:
            print(f"[{i:4}/{len(identity_names)}] SKIP {id_name} (Not enough frontal valid images: {len(selected)})")
            skipped += 1
            continue
            
        dest = gallery_dir / id_name.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        
        for src, sc in selected:
            shutil.copy2(src, dest / src.name)
            
        names_str = ", ".join(f"{p.name}({s:.1f})" for p, s in selected)
        print(f"[{i:4}/{len(identity_names)}] {id_name}: {names_str}")
        
        total_saved += len(selected)
        
    print(f"\n✓ Gallery for {dataset_name.upper()}: {gallery_dir}")
    print(f"  {len(identity_names) - skipped} identities | {total_saved} images | {skipped} skipped")


def main():
    for ds in DATASETS:
        process_dataset(ds)

if __name__ == "__main__":
    main()
