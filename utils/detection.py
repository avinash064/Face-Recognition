"""
utils/detection.py

SCRFD-based face detection utilities using InsightFace.
"""

import cv2
import numpy as np
from typing import Optional


def load_detector(providers=None):
    """
    Load SCRFD face detector from InsightFace (buffalo_l model pack).

    Args:
        providers: List of ONNX providers. Defaults to CUDAExecutionProvider.

    Returns:
        Initialized FaceAnalysis app (detector only).
    """
    from insightface.app import FaceAnalysis
    if providers is None:
        providers = ['CUDAExecutionProvider']
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'], providers=providers)
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def detect_largest_face(app, img: np.ndarray):
    """
    Detect all faces and return the largest bounding-box face.

    Args:
        app: Loaded InsightFace FaceAnalysis app.
        img: BGR image as a NumPy array.

    Returns:
        The largest face object, or None if no face detected.
    """
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
