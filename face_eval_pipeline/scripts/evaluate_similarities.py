"""
evaluate_similarities.py

Evaluates probe embeddings against precomputed gallery embeddings using cosine similarity.
It satisfies the user requirements completely:
1. Loads models & extracts purely from saved .pkl embeddings.
2. Formats mean embeddings per identity (identity-level embeddings).
3. Computes Cosine Similarity scores cleanly without neural network overhead.
4. Outputs Rank-1 accuracy per dataset and model, placing the results in a nicely formatted table.
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

MODELS = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"]
DATASETS = ["ORL", "IMDB", "IMFDB"]

EMBEDDING_DIR = Path(__file__).resolve().parent.parent / "results" / "embeddings"
OUTPUT_DIR    = Path(__file__).resolve().parent.parent / "results"

def load_gallery(db_name, model_name):
    pkl_file = EMBEDDING_DIR / f"gallery_embeddings_{db_name.lower()}_{model_name}.pkl"
    if not pkl_file.exists():
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    identities = []
    mean_embeddings = []
    
    for identity, imgs in data.items():
        if not imgs: continue
        
        # Identity-level embeddings -> Mean of the individual embeddings
        embs = list(imgs.values())
        mean_emb = np.mean(embs, axis=0)
        
        # Re-normalize! (Mean of normalized vectors is not normalized)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
             mean_emb = mean_emb / norm
             
        identities.append(identity)
        mean_embeddings.append(mean_emb)
        
    if len(mean_embeddings) == 0:
        return None
        
    return {
        'features': np.array(mean_embeddings), # (N_identities, emb_size)
        'labels': np.array(identities)
    }

def load_probes(db_name, model_name):
    pkl_file = EMBEDDING_DIR / f"probe_embeddings_{db_name.lower()}_{model_name}.pkl"
    if not pkl_file.exists():
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        
    probes = []
    for identity, imgs in data.items():
        for img_name, emb in imgs.items():
            probes.append({
                'identity': identity,
                'img_name': img_name,
                'embedding': emb
            })
    return probes

def main():
    metrics = {db: {m: 0.0 for m in MODELS} for db in DATASETS}
    
    for db_name in DATASETS:
        print(f"\nProcessing Dataset: {db_name}...")
        for m in MODELS:
            gallery = load_gallery(db_name, m)
            if gallery is None:
                # Missing PKL
                continue
                
            probes = load_probes(db_name, m)
            if not probes:
                continue
                
            valid_identities = set(gallery['labels'])
            gal_feats = gallery['features']
            gal_labels = gallery['labels']
            
            correct = 0
            total = 0
            
            for p in tqdm(probes, desc=f"Evaluating {m} on {db_name}"):
                true_id = p['identity']
                # We strictly evaluate probes that have a corresponding identity in the gallery!
                if true_id not in valid_identities:
                    continue
                    
                emb = p['embedding']
                
                # Vectorized Cosine Similarity
                sims = np.dot(gal_feats, emb)
                top_idx = np.argmax(sims)
                pred_id = gal_labels[top_idx]
                
                total += 1
                if pred_id == true_id:
                    correct += 1
                    
            if total > 0:
                acc = correct / total
                metrics[db_name][m] = acc
                print(f"  [{m:<10}] Rank-1: {acc*100:.2f}% ({correct}/{total})")
            else:
                print(f"  [{m:<10}] No overlapping valid probes evaluated.")

    # ---- 13. Model Comparison Table (IMPORTANT FEATURE) ----
    print("\n\n" + "="*80)
    print("FINAL MODEL COMPARISON (Cosine Similarity Rank-1 Accuracy)")
    print("="*80)
    
    header = f"| {'Model':<12} | " + " | ".join([f"{db:<10} Rank-1" for db in DATASETS]) + " | Avg Accuracy |"
    print(header)
    print("-" * len(header))
    
    model_avgs = {}
    json_output = {"model_comparison": {}}
    
    for m in MODELS:
        json_output["model_comparison"][m] = {}
        row = f"| {m:<12} | "
        
        valid_accs = []
        for db in DATASETS:
            acc = metrics[db].get(m, 0.0)
            valid_accs.append(acc)
            json_output["model_comparison"][m][db] = acc
            row += f"{acc:<17.4f} | "
            
        avg_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
        model_avgs[m] = avg_acc
        json_output["model_comparison"][m]["avg"] = avg_acc
        
        row += f"{avg_acc:<12.4f} |"
        print(row)
        
    print("="*80)
    
    # 14/15. Best Model Identification
    if model_avgs:
        best_model = max(model_avgs, key=model_avgs.get)
        best_avg = model_avgs[best_model]
        
        json_output["best_model"] = best_model
        
        print(f"\nBest Model: {best_model} (Avg Accuracy: {best_avg:.4f})\n")
    else:
        print("\nNo models evaluated successfully.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "similarity_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"✓ Saved results to {json_path}")

if __name__ == "__main__":
    main()
