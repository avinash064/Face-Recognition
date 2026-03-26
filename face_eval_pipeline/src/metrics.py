"""
metrics.py
==========
All evaluation metrics required by the pipeline specification:

  * Rank-1 Identification Accuracy
  * Top-K Identification Accuracy
  * TAR (True Acceptance Rate) / FAR (False Acceptance Rate) curve
  * ROC curve + AUC (sklearn-backed)
  * Plotting helpers (non-interactive, saved to disk)
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')        # headless – no display required
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identification metrics
# ---------------------------------------------------------------------------

def compute_rank1_accuracy(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Rank-1 closed-set identification accuracy.

    Fraction of probes whose top-1 gallery match is the correct identity.

    Parameters
    ----------
    predictions   : top-1 predicted identity per probe
    ground_truths : true identity per probe (same order)

    Returns
    -------
    float in [0, 1]
    """
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def compute_topk_accuracy(
    ranked_predictions: List[List[str]],
    ground_truths: List[str],
    k: int = 5,
) -> float:
    """
    Top-K identification accuracy.

    Fraction of probes where the true identity appears within the
    first *k* ranked candidates.

    Parameters
    ----------
    ranked_predictions : ranked list of identities per probe (descending similarity)
    ground_truths      : true identity per probe
    k                  : rank cutoff (default 5)

    Returns
    -------
    float in [0, 1]
    """
    if not ranked_predictions:
        return 0.0
    correct = sum(gt in preds[:k] for preds, gt in zip(ranked_predictions, ground_truths))
    return correct / len(ranked_predictions)


# ---------------------------------------------------------------------------
# Verification metrics  (TAR / FAR)
# ---------------------------------------------------------------------------

def compute_tar_far(
    genuine_scores: List[float],
    impostor_scores: List[float],
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep a similarity threshold and compute TAR and FAR at each point.

    TAR (True  Acceptance Rate) = TP / (TP + FN)  → genuine pairs accepted
    FAR (False Acceptance Rate) = FP / (FP + TN)  → impostor pairs accepted

    Parameters
    ----------
    genuine_scores  : cosine similarities for same-identity pairs
    impostor_scores : cosine similarities for cross-identity pairs
    thresholds      : explicit threshold array; auto-computed if None
    n_thresholds    : number of uniformly-spaced thresholds to try

    Returns
    -------
    thresholds, tar_array, far_array
    """
    all_scores = genuine_scores + impostor_scores
    if not all_scores:
        return np.array([]), np.array([]), np.array([])

    if thresholds is None:
        lo, hi = float(min(all_scores)), float(max(all_scores))
        thresholds = np.linspace(lo, hi, n_thresholds)

    gen_arr = np.asarray(genuine_scores,  dtype=np.float32)
    imp_arr = np.asarray(impostor_scores, dtype=np.float32)

    tar_vals = np.array([
        float(np.mean(gen_arr >= t)) if len(gen_arr) else 0.0
        for t in thresholds
    ])
    far_vals = np.array([
        float(np.mean(imp_arr >= t)) if len(imp_arr) else 0.0
        for t in thresholds
    ])

    return thresholds, tar_vals, far_vals


def tar_at_far(
    thresholds: np.ndarray,
    tar_vals: np.ndarray,
    far_vals: np.ndarray,
    target_far: float,
) -> float:
    """
    Interpolate TAR at a specific FAR operating point.

    Parameters
    ----------
    target_far : e.g. 0.01  (1 %) or 0.1  (10 %)

    Returns
    -------
    TAR value at the requested FAR (or 0.0 if not achievable).
    """
    # FAR decreases as threshold increases; find index where FAR ≤ target_far
    idx = np.where(far_vals <= target_far)[0]
    if len(idx) == 0:
        return 0.0
    # Highest TAR in that region
    return float(np.max(tar_vals[idx]))


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------

def compute_roc_auc(
    genuine_scores: List[float],
    impostor_scores: List[float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute full ROC curve and AUC using sklearn.

    Labels: 1 = genuine pair, 0 = impostor pair.

    Returns
    -------
    fpr, tpr, auc_score
    """
    if not genuine_scores or not impostor_scores:
        logger.warning("Insufficient scores to compute ROC – returning diagonal.")
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    scores = np.array(genuine_scores + impostor_scores, dtype=np.float64)
    labels = np.array(
        [1] * len(genuine_scores) + [0] * len(impostor_scores), dtype=np.int32
    )

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score    = float(auc(fpr, tpr))
    return fpr, tpr, auc_score


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def extract_verification_scores(
    test_items: Dict[str, List[Tuple[str, Optional[np.ndarray]]]],
    matcher,
    gallery_identity_set: set,
) -> Tuple[List[float], List[float]]:
    """
    Build genuine and impostor cosine-similarity distributions.

    For each test probe whose identity is in the gallery:
      * Genuine score  = similarity to the matching gallery identity
      * Impostor scores = similarity to every *other* gallery identity

    Parameters
    ----------
    test_items           : {identity: [(path, embedding), …]}
    matcher              : fitted FaceMatcher
    gallery_identity_set : set of identity labels in the gallery

    Returns
    -------
    genuine_scores, impostor_scores
    """
    genuine_scores:  List[float] = []
    impostor_scores: List[float] = []

    for identity, items in test_items.items():
        if identity not in gallery_identity_set:
            continue

        for _, emb in items:
            if emb is None:
                continue

            all_scores = matcher.get_all_scores(emb)  # {id: cosine_sim}

            if identity in all_scores:
                genuine_scores.append(all_scores[identity])

            for other_id, score in all_scores.items():
                if other_id != identity:
                    impostor_scores.append(score)

    return genuine_scores, impostor_scores


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    dataset_name: str,
    save_path: str,
) -> None:
    """Save ROC curve PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color='royalblue', label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], lw=1.5, color='gray', linestyle='--', label='Chance')
    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Acceptance Rate (TAR)',  fontsize=12)
    ax.set_title(f'ROC Curve — {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"ROC curve saved → {save_path}")


def plot_tar_far(
    thresholds: np.ndarray,
    tar_vals: np.ndarray,
    far_vals: np.ndarray,
    dataset_name: str,
    save_path: str,
) -> None:
    """Save TAR/FAR vs Threshold PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, tar_vals, lw=2, color='green', label='TAR')
    ax.plot(thresholds, far_vals, lw=2, color='red',   label='FAR')
    ax.set_xlabel('Cosine Similarity Threshold', fontsize=12)
    ax.set_ylabel('Rate',                        fontsize=12)
    ax.set_title(f'TAR & FAR vs Threshold — {dataset_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"TAR/FAR plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------

def summarize_metrics(metrics: Dict) -> str:
    """Pretty-print a metrics dictionary."""
    lines = [
        "=" * 58,
        f"  Dataset          : {metrics.get('dataset', 'N/A')}",
        "=" * 58,
        f"  Rank-1 Accuracy  : {metrics.get('rank1_accuracy', 0):.4f}  "
        f"({metrics.get('rank1_accuracy', 0) * 100:.2f}%)",
        f"  Top-5  Accuracy  : {metrics.get('top5_accuracy', 0):.4f}  "
        f"({metrics.get('top5_accuracy', 0) * 100:.2f}%)",
        f"  AUC (ROC)        : {metrics.get('auc', 0):.4f}",
        f"  TAR @ FAR=1%     : {metrics.get('tar_at_far001', 0):.4f}",
        f"  TAR @ FAR=10%    : {metrics.get('tar_at_far01',  0):.4f}",
        "-" * 58,
        f"  Test probes      : {metrics.get('n_probes', 0)}",
        f"  Gallery images   : {metrics.get('n_gallery', 0)}",
        f"  Identities       : {metrics.get('n_identities', 0)}",
        f"  Failed detections: {metrics.get('n_failed', 0)}",
        "=" * 58,
    ]
    return "\n".join(lines)
