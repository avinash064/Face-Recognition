"""
utils/embedding.py

Utilities for loading InsightFace recognition models and extracting
L2-normalized face embeddings.
"""

import numpy as np
from typing import Optional


def load_recognizer(model_name: str, providers=None):
    """
    Load an InsightFace full FaceAnalysis app for a given model pack.

    Args:
        model_name: InsightFace model name (e.g. 'buffalo_l', 'antelopev2').
        providers: ONNX providers list. Defaults to CUDAExecutionProvider.

    Returns:
        Tuple of (FaceAnalysis app, recognition module) or (None, None) on failure.
    """
    from insightface.app import FaceAnalysis
    if providers is None:
        providers = ['CUDAExecutionProvider']
    try:
        app = FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=0)
        rec = app.models.get('recognition')
        if rec is None:
            return None, None
        return app, rec
    except Exception as e:
        print(f"  [WARN] Could not load model '{model_name}': {e}")
        return None, None


def extract_embedding(img: np.ndarray, face, recognizer) -> Optional[np.ndarray]:
    """
    Extract the L2-normalized face embedding for a given face.

    Args:
        img:        Full image as a NumPy array (BGR format).
        face:       InsightFace face object (from detector).
        recognizer: InsightFace recognition model module.

    Returns:
        Normalized embedding as float32 ndarray, or None if extraction fails.
    """
    try:
        recognizer.get(img, face)
        emb = face.normed_embedding
        if emb is None:
            return None
        return emb.astype(np.float32)
    except Exception:
        return None
