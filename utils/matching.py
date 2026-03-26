"""
utils/matching.py

Cosine similarity matching utilities for face recognition evaluation.
"""

import numpy as np


def build_gallery_matrix(embeddings_dict: dict):
    """
    Build mean identity-level embeddings from a saved gallery dict.

    Args:
        embeddings_dict: Dict of {identity: {img_name: embedding_vector}}.

    Returns:
        Tuple of (feature_matrix, labels_array).
        feature_matrix shape: (N_identities, embedding_dim)
        labels_array shape:   (N_identities,)
    """
    identities, mean_embeddings = [], []
    for identity, imgs in embeddings_dict.items():
        if not imgs:
            continue
        embs = list(imgs.values())
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        identities.append(identity)
        mean_embeddings.append(mean_emb)

    if not mean_embeddings:
        return None, None

    return np.array(mean_embeddings), np.array(identities)


def cosine_match(probe_embedding: np.ndarray, gallery_features: np.ndarray, gallery_labels: np.ndarray):
    """
    Match a probe embedding against the gallery using dot-product cosine similarity.

    Since all embeddings are L2 normalized, dot product equals cosine similarity.

    Args:
        probe_embedding:  1-D normalized embedding vector.
        gallery_features: (N, D) normalized gallery embeddings.
        gallery_labels:   (N,) identity labels.

    Returns:
        Tuple of (predicted_identity, similarity_score).
    """
    sims = np.dot(gallery_features, probe_embedding)
    top_idx = np.argmax(sims)
    return gallery_labels[top_idx], float(sims[top_idx])
