"""
evaluate_similarities_pt.py

Standalone script to evaluate the cosine similarity metrics exclusively for 
the PyTorch (.pt) embeddings of buffalo_m and antelopev2.
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
import yaml

with open("configs/paths.yaml", 'r') as f:
    config = yaml.safe_load(f)

MODELS = ["buffalo_m", "antelopev2"]
DATASETS = ["ORL", "IMDB", "IMFDB"]

EMBEDDING_DIR = Path(config['embeddings_dir'])
OUTPUT_DIR = Path(config['results_dir'])

def load_gallery(db_name, model_name):
    pkl_file = EMBEDDING_DIR / f"gallery_embeddings_{db_name.lower()}_{model_name}_pt.pkl"
    if not pkl_file.exists(): return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    identities, mean_embeddings = [], []
    for identity, imgs in data.items():
        if not imgs: continue
        mean_emb = np.mean(list(imgs.values()), axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0: mean_emb = mean_emb / norm
        identities.append(identity)
        mean_embeddings.append(mean_emb)
        
    if not mean_embeddings: return None
    return {'features': np.array(mean_embeddings), 'labels': np.array(identities)}

def load_probes(db_name, model_name):
    pkl_file = EMBEDDING_DIR / f"probe_embeddings_{db_name.lower()}_{model_name}_pt.pkl"
    if not pkl_file.exists(): return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    probes = []
    for identity, imgs in data.items():
        for img_name, emb in imgs.items():
            probes.append({'identity': identity, 'embedding': emb})
    return probes

def main():
    metrics = {db: {m: 0.0 for m in MODELS} for db in DATASETS}
    
    for db_name in DATASETS:
        for m in MODELS:
            gallery = load_gallery(db_name, m)
            probes = load_probes(db_name, m)
            if not gallery or not probes: continue
                
            valid_identities = set(gallery['labels'])
            gal_feats, gal_labels = gallery['features'], gallery['labels']
            
            correct = 0
            total = 0
            for p in probes:
                true_id = p['identity']
                if true_id not in valid_identities: continue
                
                sims = np.dot(gal_feats, p['embedding'])
                if gal_labels[np.argmax(sims)] == true_id:
                    correct += 1
                total += 1
                
            if total > 0:
                metrics[db_name][m] = correct / total

    print("\n" + "="*80)
    print("PYTORCH MODEL COMPARISON (.pt Weights - Cosine Similarity Rank-1 Accuracy)")
    print("="*80)
    
    header = f"| {'Model':<12} | " + " | ".join([f"{db:<10} Rank-1" for db in DATASETS]) + " | Avg Accuracy |"
    print(header)
    print("-" * len(header))
    
    for m in MODELS:
        row = f"| {m:<12} | "
        valid_accs = []
        for db in DATASETS:
            acc = metrics[db].get(m, 0.0)
            valid_accs.append(acc)
            row += f"{acc:<17.4f} | "
        avg_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
        row += f"{avg_acc:<12.4f} |"
        print(row)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
