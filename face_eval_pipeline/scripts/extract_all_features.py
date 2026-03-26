"""
extract_all_features.py

Extracts embeddings for BOTH the reference galleries and the probe datasets.
- Uses EXACTLY ONE SCRFD face detector (from buffalo_l).
- To preserve GPU VRAM (4GB), it loads one Recognition model at a time, processes 
  all images, unloads it, and loads the next.
- Extracts and saves .pkl files for every dataset-model combination.

Usage:
    python scripts/extract_all_features.py
"""

import os
import cv2
import pickle
import copy
import math
import gc
import numpy as np
import scipy.io
import onnxruntime as ort
from pathlib import Path
from insightface.app import FaceAnalysis

MODELS = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"]

GALLERIES = {
    "ORL":   Path("/home/avinash/Desktop/Bidaal/face_eval_pipeline/orl_faces/reference_gallery"),
    "IMDB":  Path("/home/avinash/datasets/imdb_wiki/reference_gallery_imdb"),
    "IMFDB": Path("/home/avinash/Desktop/Bidaal/datasets/IMFDB FR dataset/reference_gallery")
}

PROBES = {
    "ORL":   Path("/home/avinash/Desktop/Bidaal/face_eval_pipeline/orl_faces"),
    "IMDB":  Path("/home/avinash/datasets/imdb_wiki/imdb_crop"),
    "IMFDB": Path("/home/avinash/Desktop/Bidaal/datasets/IMFDB FR dataset/IMFDB FR dataset")
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_probe_images(db_name):
    """ Returns list of (Path, identity_name) for probe datasets. """
    probes = []
    root = PROBES[db_name]
    if not root.exists(): return probes
        
    if db_name == "ORL":
        for subj_dir in root.iterdir():
            if not subj_dir.is_dir() or not subj_dir.name.startswith("s"): continue
            identity = subj_dir.name
            for p in subj_dir.iterdir():
                if p.is_file() and p.name.lower().endswith(('.jpg','.jpeg','.png','.bmp','.pgm')):
                    probes.append((p, identity))
                    
    elif db_name == "IMFDB":
        for actor_dir in root.iterdir():
            if not actor_dir.is_dir(): continue
            identity = actor_dir.name
            for p in actor_dir.rglob("*"):
                if "reference_gallery" in str(p): continue
                if p.is_file() and p.name.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                    probes.append((p, identity))
                    
    elif db_name == "IMDB":
        mat_file = root / "imdb.mat"
        if not mat_file.exists(): return []
        mat = scipy.io.loadmat(str(mat_file))
        struct = mat['imdb'][0,0]
        full_paths = struct['full_path'][0]
        names = struct['name'][0]
        face_scores = struct['face_score'][0]
        second_face_scores = struct['second_face_score'][0]
        
        for i in range(len(full_paths)):
            path_arr = full_paths[i]
            name_arr = names[i]
            if len(path_arr) == 0 or len(name_arr) == 0: continue
            
            rel_path = str(path_arr[0])
            val = name_arr[0]
            identity_name = str(val[0]).strip() if isinstance(val, np.ndarray) and len(val)>0 else str(val).strip()
            if not identity_name: continue
            
            f_score = float(face_scores[i])
            sf_score = float(second_face_scores[i])
            if math.isinf(f_score) or math.isnan(f_score) or f_score < 1.0: continue
            if not math.isnan(sf_score): continue
                
            abs_path = root / rel_path
            probes.append((abs_path, identity_name))
            
    return probes


def get_gallery_images(db_name):
    """ Returns dict {identity: [paths]} for gallery dataset. """
    images = {}
    db_path = GALLERIES[db_name]
    if not db_path.exists(): return images
    for identity_dir in db_path.iterdir():
        if not identity_dir.is_dir(): continue
        identity = identity_dir.name
        images[identity] = []
        for p in identity_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.pgm'}:
                images[identity].append(p)
    return images


def main():
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print("CRITICAL WARNING: GPU not detected! Falling back to CPU.")
    else:
        print("✓ GPU (CUDA) detected successfully!")
        
    print("\nInitializing Base SCRFD Face Detector (from buffalo_l)...")
    base_detector_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    try:
        base_detector_app.prepare(ctx_id=0, det_size=(320, 320))
    except Exception as e:
        print(f"Failed to load base detector: {e}")
        return

    # To avoid repeated reads to disk for the same image across different models,
    # we ideally want to read them once. But given memory constraints, reading from SSD is fast enough.
    # The image caching strategy ensures we scale to massive datasets.
    
    for m in MODELS:
        print(f"\n{'='*60}\n Loading Model: {m}")
        try:
            # We load the full app normally to prevent insightface internal bugs, 
            # and extract the recognizer module dynamically.
            app = FaceAnalysis(name=m, providers=['CUDAExecutionProvider'])
            app.prepare(ctx_id=0)
            if 'recognition' not in app.models:
                print(f"  [WARN] Recognition module missing for {m}!")
                continue
            recognizer = app.models['recognition']
        except Exception as e:
            print(f"  [WARN] Could not load {m}: {e}")
            continue

        # 1. PROCESS GALLERIES
        print(" >> Extracting Gallery Features...")
        for db in GALLERIES:
            gallery_data = get_gallery_images(db)
            if not gallery_data: continue
            
            out_file = OUTPUT_DIR / f"gallery_embeddings_{db.lower()}_{m}.pkl"
            if out_file.exists():
                print(f"    Skipping {out_file.name} (Already exists)")
                continue
            
            emb_dict = {}
            for identity, paths in gallery_data.items():
                emb_dict[identity] = {}
                for p in paths:
                    img = cv2.imread(str(p))
                    if img is None: continue
                    faces = base_detector_app.get(img)
                    if len(faces) == 0: continue
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    
                    recognizer.get(img, face)
                    if face.normed_embedding is not None:
                        emb_dict[identity][p.name] = face.normed_embedding.astype(np.float32)
                        
            with open(out_file, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"    ✓ Saved {out_file.name}")

        # 2. PROCESS PROBES
        print("\n >> Extracting Probe Features...")
        for db in PROBES:
            probes = get_probe_images(db)
            if not probes: continue
            
            out_file = OUTPUT_DIR / f"probe_embeddings_{db.lower()}_{m}.pkl"
            if out_file.exists():
                print(f"    Skipping {out_file.name} (Already exists)")
                continue

            # Identify which images were in the gallery so we skip them
            gallery_data = get_gallery_images(db)
            gallery_files = set()
            for id_key, paths in gallery_data.items():
                for p in paths: gallery_files.add(p.name)

            emb_dict = {}
            total_p = len(probes)
            
            print(f"    Total probe set for {db}: {total_p} images")
            
            for idx, (p, identity) in enumerate(probes, 1):
                if p.name in gallery_files:
                    continue # Skip images used in Gallery
                    
                img = cv2.imread(str(p))
                if img is None: continue
                faces = base_detector_app.get(img)
                if len(faces) == 0: continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                
                recognizer.get(img, face)
                if face.normed_embedding is not None:
                    if identity not in emb_dict:
                        emb_dict[identity] = {}
                    emb_dict[identity][p.name] = face.normed_embedding.astype(np.float32)

                if idx % 2000 == 0:
                    print(f"      Processed {idx}/{total_p} ...")
                    
            with open(out_file, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"    ✓ Saved {out_file.name}")

        # Delete loaded face analysis apps to free VRAM for the next model!
        print(f"  Unloading {m} to free VRAM...")
        try:
            del app
            del recognizer
        except NameError:
            pass
        gc.collect()

    print("\nExtraction of all Galleries and Probes finished successfully!")

if __name__ == "__main__":
    main()
