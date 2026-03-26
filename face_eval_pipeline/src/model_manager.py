"""
model_manager.py
================
Central manager for loading and switching between InsightFace models.
Handles automatic GPU/CPU detection via ONNX Runtime and caches models.

Supported models:
  - buffalo_l  (ArcFace R100, high accuracy)
  - buffalo_m  (Medium accuracy)
  - buffalo_s  (Lightweight)
  - buffalo_sc (Lightweight/Mobile, CPU friendly)
"""

import logging
import time
from typing import Dict, Any

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading InsightFace FaceAnalysis models with fallback to CPU.
    """
    
    SUPPORTED_MODELS = {
        'buffalo_l':  {"type": "Large", "desc": "High accuracy, default ArcFace R100"},
        'buffalo_m':  {"type": "Medium", "desc": "Balance of speed and accuracy"},
        'buffalo_s':  {"type": "Small", "desc": "Lightweight model"},
        'buffalo_sc': {"type": "Mobile", "desc": "Very lightweight, excellent for CPU"},
    }

    def __init__(self, device: str = 'auto', det_size: tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        """
        device: 'auto', 'cuda', or 'cpu'
        """
        self.device = device.lower()
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._cache = {}  # Cache loaded FaceAnalysis instances to avoid redundant loading
        
        self.active_providers, self.ctx_id = self._determine_providers()
        
    def _determine_providers(self) -> tuple[list[str], int]:
        if onnxruntime is None:
            raise ImportError("onnxruntime is not installed. Please install onnxruntime-gpu.")

        available = onnxruntime.get_available_providers()
        
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(
                "GPU ONLY execution is strictly enforced, but CUDAExecutionProvider "
                "is not available in onnxruntime. Found providers: " + str(available)
            )
            
        logger.info("GPU available. Enforcing CUDAExecutionProvider only.")
        return ['CUDAExecutionProvider'], 0

    def load_model(self, model_name: str):
        """
        Load an InsightFace model. Downloads automatically if missing.
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model '{model_name}' is not officially supported, but attempting to load anyway.")

        if model_name in self._cache:
            logger.debug(f"Returning cached model: {model_name}")
            return self._cache[model_name]

        logger.info(f"Loading model '{model_name}' ...")
        t0 = time.perf_counter()
        
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError("InsightFace is not installed. `pip install insightface`")

        app = FaceAnalysis(name=model_name, providers=self.active_providers)
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        
        # Override det_thresh internally (FaceAnalysis hides this deeply, 
        # but we handle filtering in our wrapper later anyway. 
        # Alternatively, we set it on the detector directly)
        app.models['detection'].det_thresh = self.det_thresh
        
        elapsed = time.perf_counter() - t0
        logger.info(f"Model '{model_name}' loaded in {elapsed:.2f}s.")
        
        self._cache[model_name] = app
        return app

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Return human-readable metadata about the model."""
        return self.SUPPORTED_MODELS.get(model_name, {"type": "Unknown", "desc": "Unknown model variant"})

