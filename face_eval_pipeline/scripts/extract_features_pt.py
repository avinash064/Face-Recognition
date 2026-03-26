"""
scripts/extract_features_pt.py

Standalone script requested by the user to evaluate buffalo_m and 
antelopev2 using PyTorch (.pt) weights directly, avoiding the ONNX loader issues.

Expects:
    models/buffalo_m.pt
    models/antelopev2.pt
These files should be saved as TorchScript files. If they are raw state dictionaries, 
you must instantiate the ArcFace network first.
"""

import os
import sys
import cv2
import pickle
import yaml
import numpy as np
import torch
from pathlib import Path

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Append parent dir to path so we can import the parser functions
sys.path.append(str(Path(__file__).resolve().parent))
from extract_all_features import get_probe_images, get_gallery_images

with open("configs/paths.yaml", 'r') as f:
    config = yaml.safe_load(f)

MODELS = ["buffalo_m", "antelopev2"]
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PT_PATHS = {
    m: MODEL_DIR / f"{m}.pt" for m in MODELS
}

OUTPUT_DIR = Path(config['embeddings_dir'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pt_model(model_name):
    pt_path = PT_PATHS[model_name]
    if not pt_path.exists():
        print(f"  [ERROR] {pt_path} not found!")
        print(f"          Please place your PyTorch file at: {pt_path}")
        return None, None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading {model_name} onto {device}...")
    try:
        # Load TorchScript map
        model = torch.jit.load(str(pt_path), map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"  [ERROR] Failed to load {model_name} via torch.jit.load: {e}")
        print("          If this file is a state_dict, you MUST instantiate the specific ArcFace network architecture first locally.")
        return None, None


def extract_embedding_pt(model, device, img, face):
    # ArcFace standard alignment: norm_crop returns 112x112 chip
    aligned_img = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    
    # Preprocess BGR -> RGB -> (C,H,W) -> Normalize -1 to 1
    rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    img_tensor = np.transpose(rgb_img, (2, 0, 1))
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    img_tensor.div_(255).sub_(0.5).div_(0.5)
    
    with torch.no_grad():
        feat = model(img_tensor).cpu().numpy().flatten()
        
    norm = np.linalg.norm(feat)
    if norm > 0: feat = feat / norm
    return feat


def main():
    print("Initializing Base SCRFD Face Detector (from buffalo_l)...")
    detector_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    try:
        detector_app.prepare(ctx_id=0, det_size=(320, 320))
    except Exception as e:
        print(f"Failed to load base detector: {e}")
        return

    for m in MODELS:
        print(f"\n=======================================================\n Processing PyTorch Model: {m}")
        model, device = load_pt_model(m)
        if model is None:
            continue
            
        print("\n >> Extracting PyTorch Gallery Features...")
        for db in config['datasets'].keys():
            db = db.upper()
            gallery_data = get_gallery_images(db)
            if not gallery_data: continue
            
            out_file = OUTPUT_DIR / f"gallery_embeddings_{db.lower()}_{m}_pt.pkl"
            emb_dict = {}
            for identity, paths in gallery_data.items():
                emb_dict[identity] = {}
                for p in paths:
                    img = cv2.imread(str(p))
                    if img is None: continue
                    faces = detector_app.get(img)
                    if len(faces) == 0: continue
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    
                    emb = extract_embedding_pt(model, device, img, face)
                    if emb is not None:
                        emb_dict[identity][p.name] = emb
                        
            with open(out_file, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"    ✓ Saved {out_file.name}")

        print("\n >> Extracting PyTorch Probe Features...")
        for db in config['datasets'].keys():
            db = db.upper()
            probes = get_probe_images(db)
            if not probes: continue
            
            out_file = OUTPUT_DIR / f"probe_embeddings_{db.lower()}_{m}_pt.pkl"
            
            gallery_data = get_gallery_images(db)
            gallery_files = set()
            for paths in gallery_data.values():
                for p in paths: gallery_files.add(p.name)

            emb_dict = {}
            total_p = len(probes)
            print(f"    Extracting {total_p} probes for {db}...")
            
            for idx, (p, identity) in enumerate(probes, 1):
                if p.name in gallery_files: continue
                
                img = cv2.imread(str(p))
                if img is None: continue
                faces = detector_app.get(img)
                if len(faces) == 0: continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                
                emb = extract_embedding_pt(model, device, img, face)
                if emb is not None:
                    if identity not in emb_dict:
                        emb_dict[identity] = {}
                    emb_dict[identity][p.name] = emb

                if idx % 2000 == 0:
                    print(f"      Processed {idx}/{total_p} ...")
                    
            with open(out_file, 'wb') as f:
                pickle.dump(emb_dict, f)
            print(f"    ✓ Saved {out_file.name}")

if __name__ == "__main__":
    main()
