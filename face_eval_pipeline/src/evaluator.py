"""
evaluator.py
============
Orchestrates the complete face recognition evaluation pipeline:

  1.  Load dataset (via DataLoader)
  2.  Extract ArcFace embeddings and pose for ALL images (1 pass)
  3.  Perform pose-diversity based train (gallery) / test (probe) split
  4.  Build the gallery index (FaceMatcher)
  5.  Match every probe against the gallery
  6.  Compute all metrics (Rank-1, Top-5, TAR/FAR, ROC/AUC)
  7.  Save per-model JSON results + PNG plots
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .data_loader import DataLoader
from .embedding import FaceEmbedder
from .matcher import FaceMatcher
from .pose_selector import PoseSelector
from .metrics import (
    compute_rank1_accuracy,
    compute_topk_accuracy,
    compute_tar_far,
    compute_roc_auc,
    extract_verification_scores,
    plot_roc_curve,
    plot_tar_far,
    summarize_metrics,
    tar_at_far,
)

logger = logging.getLogger(__name__)

class Evaluator:
    """
    End-to-end multi-model face recognition evaluator.
    Supports both live inference and loading from pre-built .npz caches.
    """

    def __init__(
        self,
        embedder: FaceEmbedder,
        model_name: str,
        n_train: int = 2,
        top_k: int = 5,
        results_dir: str = "results",
        use_mean_gallery: bool = True,
        embed_cache_dir: Optional[str] = None,
    ):
        self.embedder         = embedder
        self.model_name       = model_name
        self.n_train          = n_train
        self.top_k            = top_k
        self.results_dir      = Path(results_dir)
        self.use_mean_gallery = use_mean_gallery
        self.embed_cache_dir  = Path(embed_cache_dir) if embed_cache_dir else None
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, dataset_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Load pre-built embeddings from a .npz cache file.
        Returns { identity: { path: {'embedding': np.ndarray, 'pose': dict} } }
        or None if the cache does not exist.
        """
        if self.embed_cache_dir is None:
            return None

        cache_path = self.embed_cache_dir / f"{dataset_name}_{self.model_name}.npz"
        if not cache_path.exists():
            logger.info(f"No cache file found at {cache_path}. Will run live inference.")
            return None

        logger.info(f"Loading pre-built embeddings from {cache_path.name} …")
        data = np.load(str(cache_path), allow_pickle=True)

        paths      = data['paths']
        labels     = data['labels']
        embeddings = data['embeddings']
        yaw_arr    = data['yaw']
        pitch_arr  = data['pitch']
        roll_arr   = data['roll']
        det_scores = data['det_scores']

        all_face_data: Dict[str, Dict[str, Any]] = {}
        n_failed = 0

        for i in range(len(paths)):
            identity = str(labels[i])
            path     = str(paths[i])
            score    = float(det_scores[i])

            # det_score == -1.0 means detection had failed during build
            if score < 0:
                n_failed += 1
                continue

            all_face_data.setdefault(identity, {})[path] = {
                'embedding': embeddings[i],
                'pose': {
                    'yaw':   float(yaw_arr[i]),
                    'pitch': float(pitch_arr[i]),
                    'roll':  float(roll_arr[i]),
                },
            }

        logger.info(
            f"  Cache loaded: {len(all_face_data)} identities, {n_failed} skipped (no face)."
        )
        return all_face_data


    def evaluate(self, loader: DataLoader, dataset_name: str) -> Dict:
        """
        Run the full pipeline on a single dataset for the current model.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  Evaluating dataset: {dataset_name} | Model: {self.model_name}")
        logger.info(f"{'='*60}")

        # ── Step 1 : Load dataset ──────────────────────────────────────
        logger.info("Step 1 / 5 : Loading dataset …")
        identity_images = loader.load()
        if not identity_images:
            logger.error(f"No images found in {loader.dataset_path}. Aborting.")
            return {}

        # ── Step 2 : Load cache or extract Embeddings & Pose ──────────
        safe_dataset = dataset_name.lower().replace(' ', '_').replace('-', '_').replace('&', 'and')
        cached = self._load_cache(safe_dataset)

        if cached is not None:
            logger.info("Step 2 / 5 : Using pre-built embeddings from cache ✓")
            all_face_data = cached
            n_failed = 0  # already excluded when cache was built
        else:
            logger.info("Step 2 / 5 : Extracting embeddings & pose for all images (live) …")
            all_face_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
            n_failed = 0
            total_images = sum(len(paths) for paths in identity_images.values())

            t0 = time.perf_counter()
            processed = 0

            for identity, paths in identity_images.items():
                all_face_data[identity] = {}
                for path in paths:
                    info = self.embedder.get_face_info(path)
                    if info is not None:
                        all_face_data[identity][path] = {
                            'embedding': info['embedding'],
                            'pose': info['pose']
                        }
                    else:
                        n_failed += 1

                    processed += 1
                    if processed % 500 == 0:
                        logger.info(f"  Processed {processed}/{total_images} images")

            elapsed = time.perf_counter() - t0
            logger.info(f"  Extraction done in {elapsed:.1f}s ({n_failed} failed detections).")

        # ── Step 3 : Pose-based split ─────────────────────────────────
        logger.info(f"Step 3 / 5 : Pose-diversity split (n_gallery={self.n_train} per identity) …")
        
        gallery_embs: Dict[str, List[Optional[np.ndarray]]] = {}
        probe_items: Dict[str, List[Tuple[str, Optional[np.ndarray]]]] = {}
        
        n_gallery_total = 0
        n_probes_total = 0
        
        for identity, path_data in all_face_data.items():
            if len(path_data) < self.n_train + 1:
                # Not enough valid images for this identity to form proper gallery+probe
                # Skip identity to maintain strict cross-validation integrity, or 
                # put all in probe. We will skip identities that don't meet minimum requirements.
                continue
                
            # Prepare pose dict for selector
            pose_dict = {p: data['pose'] for p, data in path_data.items()}
            
            ref_paths, query_paths = PoseSelector.select_diverse_references(pose_dict, n_ref=self.n_train)
            
            gallery_embs[identity] = [path_data[p]['embedding'] for p in ref_paths]
            probe_items[identity] = [(p, path_data[p]['embedding']) for p in query_paths]
            
            n_gallery_total += len(ref_paths)
            n_probes_total += len(query_paths)

        n_identities = len(gallery_embs)
        
        # If no valid identities left after filtering
        if n_identities == 0:
            logger.error("No valid identities left after filtering. Aborting evaluation.")
            return {}
            
        logger.info(f"  {n_identities} identities | {n_gallery_total} gallery | {n_probes_total} probe images.")

        # ── Step 4 : Match ────────────────────────────────────────────
        logger.info("Step 4 / 5 : Building gallery index and matching …")
        matcher = FaceMatcher(use_mean_gallery=self.use_mean_gallery)
        matcher.build_gallery(gallery_embs)
        gallery_id_set = set(matcher.gallery_identities)

        top1_preds:   List[str]         = []
        topk_preds:   List[List[str]]   = []
        ground_truths: List[str]        = []

        for identity, items in probe_items.items():
            for path, emb in items:
                ranked = matcher.match(emb, top_k=self.top_k)
                top1_preds.append(ranked[0][0] if ranked else "")
                topk_preds.append([r[0] for r in ranked])
                ground_truths.append(identity)

        # ── Step 5 : Metrics ──────────────────────────────────────────
        logger.info("Step 5 / 5 : Computing metrics …")
        
        rank1 = compute_rank1_accuracy(top1_preds, ground_truths)
        topk  = compute_topk_accuracy(topk_preds, ground_truths, k=self.top_k)

        genuine_scores, impostor_scores = extract_verification_scores(
            probe_items, matcher, gallery_id_set
        )

        thresholds, tar_vals, far_vals = compute_tar_far(genuine_scores, impostor_scores)
        tar001 = tar_at_far(thresholds, tar_vals, far_vals, target_far=0.01)
        tar01  = tar_at_far(thresholds, tar_vals, far_vals, target_far=0.10)

        fpr, tpr, auc_score = compute_roc_auc(genuine_scores, impostor_scores)

        # ── Package metrics ────────────────────────────────────────────
        metrics = {
            "dataset":          dataset_name,
            "model":            self.model_name,
            "rank1_accuracy":   rank1,
            "top5_accuracy":    topk,
            "auc":              auc_score,
            "tar_at_far001":    tar001,
            "tar_at_far01":     tar01,
            "n_identities":     n_identities,
            "n_gallery":        n_gallery_total,
            "n_probes":         len(ground_truths),
            "n_failed":         n_failed,
            "n_genuine_pairs":  len(genuine_scores),
            "n_impostor_pairs": len(impostor_scores),
        }

        print(f"\n{summarize_metrics(metrics)}\n")

        # ── Save JSON ─────────────────────────────────────────────────
        safe_data_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
        safe_name = f"{safe_data_name}_{self.model_name}"
        
        json_path = self.results_dir / f"{safe_name}_results.json"
        
        json_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in metrics.items()}
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=4)
        logger.info(f"Results saved → {json_path}")

        # ── Save plots ────────────────────────────────────────────────
        roc_path    = self.results_dir / f"{safe_name}_roc.png"
        tarfar_path = self.results_dir / f"{safe_name}_tar_far.png"

        plot_title = f"{dataset_name} ({self.model_name})"
        plot_roc_curve(fpr, tpr, auc_score, plot_title, str(roc_path))

        if len(thresholds) > 0:
            plot_tar_far(thresholds, tar_vals, far_vals, plot_title, str(tarfar_path))

        return metrics
