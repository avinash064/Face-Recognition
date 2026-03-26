"""
matcher.py
==========
Nearest-neighbour face identification using cosine similarity.

Because InsightFace already returns L2-normalised embeddings,
cosine similarity reduces to a simple dot product:

    sim(a, b) = a · b   (both ||a|| = ||b|| = 1)

Gallery construction strategy
------------------------------
  use_mean_gallery=True  (default):
      All gallery embeddings for a given identity are averaged and
      re-normalised to a single representative vector.  Fast and often
      more robust than individual images.

  use_mean_gallery=False:
      Every gallery image is stored individually.  The similarity to an
      identity is taken as the *maximum* over all its gallery vectors
      (nearest-neighbour within the class).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceMatcher:
    """
    Gallery-based face identification / verification engine.

    Usage
    -----
    >>> matcher = FaceMatcher()
    >>> matcher.build_gallery({"alice": [emb_a1, emb_a2], "bob": [emb_b1]})
    >>> top5 = matcher.match(probe_embedding, top_k=5)
    """

    def __init__(self, use_mean_gallery: bool = True):
        self.use_mean_gallery = use_mean_gallery

        # Populated by build_gallery()
        self._gallery_labels: List[str] = []          # label per row
        self._gallery_matrix: Optional[np.ndarray] = None  # (N, 512) float32

    # ------------------------------------------------------------------
    # Gallery construction
    # ------------------------------------------------------------------

    def build_gallery(
        self, gallery_embeddings: Dict[str, List[Optional[np.ndarray]]]
    ) -> None:
        """
        Index gallery embeddings for fast similarity search.

        Parameters
        ----------
        gallery_embeddings : {identity_label: [embedding | None, …]}
            None values (failed detections) are silently skipped.
        """
        labels: List[str] = []
        vectors: List[np.ndarray] = []

        for identity, embs in gallery_embeddings.items():
            valid = [e for e in embs if e is not None]
            if not valid:
                logger.warning(
                    f"No valid gallery embedding for '{identity}' – skipped."
                )
                continue

            if self.use_mean_gallery:
                # Mean of unit vectors → re-normalise
                mean_vec = np.mean(valid, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vec /= norm
                labels.append(identity)
                vectors.append(mean_vec)
            else:
                # Keep individual embeddings; identity label repeated
                for emb in valid:
                    labels.append(identity)
                    vectors.append(emb)

        if not vectors:
            raise ValueError("Gallery is empty after filtering.")

        self._gallery_labels = labels
        self._gallery_matrix = np.array(vectors, dtype=np.float32)  # (N, 512)

        n_unique = len(set(labels))
        logger.info(
            f"Gallery built: {n_unique} identities, "
            f"{len(labels)} vectors, shape={self._gallery_matrix.shape}."
        )

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self._gallery_matrix is None or len(self._gallery_matrix) == 0:
            raise RuntimeError("Gallery is empty. Call build_gallery() first.")

    def compute_raw_similarities(self, probe: np.ndarray) -> np.ndarray:
        """
        Dot-product between probe and every gallery vector.

        Parameters
        ----------
        probe : (512,) float32, L2-normalised

        Returns
        -------
        similarities : (N_gallery,) float32
        """
        self._check_ready()
        return self._gallery_matrix @ probe  # shape: (N,)

    # ------------------------------------------------------------------
    # Top-K matching
    # ------------------------------------------------------------------

    def match(
        self, probe: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k gallery identities most similar to the probe.

        Parameters
        ----------
        probe  : L2-normalised (512,) array
        top_k  : number of candidates to return

        Returns
        -------
        List of (identity_label, cosine_similarity) sorted descending.
        """
        raw_sims = self.compute_raw_similarities(probe)

        if self.use_mean_gallery:
            # One vector per identity → direct argsort
            ranked_idx = np.argsort(raw_sims)[::-1][:top_k]
            return [
                (self._gallery_labels[i], float(raw_sims[i]))
                for i in ranked_idx
            ]
        else:
            # Multiple vectors per identity → aggregate by max-similarity
            id_scores: Dict[str, float] = {}
            for i, label in enumerate(self._gallery_labels):
                s = float(raw_sims[i])
                if label not in id_scores or s > id_scores[label]:
                    id_scores[label] = s
            return sorted(id_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # All-identity scores (needed for ROC / TAR-FAR computation)
    # ------------------------------------------------------------------

    def get_all_scores(self, probe: np.ndarray) -> Dict[str, float]:
        """
        Return a similarity score for **every** gallery identity.

        Useful for building genuine / impostor score distributions.
        """
        raw_sims = self.compute_raw_similarities(probe)

        if self.use_mean_gallery:
            return {
                lbl: float(sim)
                for lbl, sim in zip(self._gallery_labels, raw_sims)
            }
        else:
            id_scores: Dict[str, float] = {}
            for i, label in enumerate(self._gallery_labels):
                s = float(raw_sims[i])
                if label not in id_scores or s > id_scores[label]:
                    id_scores[label] = s
            return id_scores

    @property
    def gallery_identities(self) -> List[str]:
        """Unique identity labels present in the gallery."""
        return list(dict.fromkeys(self._gallery_labels))  # order-preserving unique
