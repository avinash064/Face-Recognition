"""
embedding.py
============
Wrapper for extracting face embeddings and poses using InsightFace.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class FaceEmbedder:
    """
    Given a pre-loaded InsightFace FaceAnalysis application, extracts
    embeddings and pose information from images.
    """

    def __init__(self, face_app: Any):
        """
        :param face_app: A loaded insightface.app.FaceAnalysis instance
        """
        self.app = face_app

    def get_face_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load an image and return embedding, pose, bounding box, 
        and detection score for the most prominent face.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                return None
                
            faces = self.app.get(img)
            if len(faces) == 0:
                logger.debug(f"No face detected in {image_path}")
                return None
                
            # If multiple faces, use the largest one by bounding box area
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            # Usually pose = [pitch, yaw, roll] in InsightFace, but we must protect against missing attr
            pose_dict = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            if hasattr(best_face, 'pose'):
                pitch, yaw, roll = best_face.pose
                pose_dict = {
                    'yaw': float(yaw),
                    'pitch': float(pitch),
                    'roll': float(roll)
                }
                
            return {
                'embedding': best_face.normed_embedding.astype(np.float32),
                'pose': pose_dict,
                'det_score': float(getattr(best_face, 'det_score', 0.0)),
                'bbox': best_face.bbox.tolist() if hasattr(best_face, 'bbox') else None
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Legacy method for backward compatibility. 
        Returns just the L2-normalized embedding.
        """
        info = self.get_face_info(image_path)
        return info['embedding'] if info else None
