"""
pose_selector.py
================
Implements pose-diversity based reference image selection for a given identity.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import cv2

logger = logging.getLogger(__name__)

class PoseSelector:
    """
    Selects reference images based on pose diversity (yaw, pitch, roll).
    Goal: Maximize angular diversity for robustness across views.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def analyze_poses(image_paths: List[str], face_app: Any) -> Dict[str, Dict[str, float]]:
        """
        Extract pose estimation (yaw, pitch, roll) for each image.
        Returns a dictionary mapping image_path -> {'yaw': float, 'pitch': float, 'roll': float}
        """
        pose_data = {}
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            
            # Use InsightFace app to detect faces and extract pose
            try:
                faces = face_app.get(img)
            except Exception as e:
                logger.debug(f"Failed to detect pose for {path}: {e}")
                continue
                
            if not faces:
                continue
                
            # If multiple faces, choose the largest one by bounding box area
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            # pose is usually [pitch, yaw, roll] in InsightFace 
            if hasattr(best_face, 'pose'):
                pitch, yaw, roll = best_face.pose
                pose_data[path] = {
                    'yaw': float(yaw),
                    'pitch': float(pitch), 
                    'roll': float(roll)
                }
            else:
                # Some models might not output pose. Fallback randomly.
                logger.debug(f"Pose data missing from detected face in {path}")
                pose_data[path] = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
                
        return pose_data

    @staticmethod
    def angular_distance(pose1: Dict[str, float], pose2: Dict[str, float]) -> float:
        """Calculate L2 distance between yaw and pitch."""
        yaw_diff = pose1['yaw'] - pose2['yaw']
        pitch_diff = pose1['pitch'] - pose2['pitch']
        return float(np.sqrt(yaw_diff**2 + pitch_diff**2))

    @staticmethod
    def select_diverse_references(pose_data: Dict[str, Dict[str, float]], n_ref: int = 3) -> Tuple[List[str], List[str]]:
        """
        Greedy selection algorithm:
        1. Pick image closest to frontal (min |yaw| + |pitch|)
        2. Pick image with max angular distance from the already selected set
        3. Repeat until n_ref images are selected
        
        Returns: (reference_paths, probe_paths)
        """
        all_paths = list(pose_data.keys())
        
        if len(all_paths) <= n_ref:
            # Not enough images to split properly, return all available as query/probe later
            # But the requirement says n_train per identity.
            # Handle edge case where we just return everything we can
            return all_paths, []
            
        # 1. First reference: closest to frontal
        def frontal_score(p):
            return abs(p['yaw']) + abs(p['pitch'])
            
        frontal_path = min(all_paths, key=lambda p: frontal_score(pose_data[p]))
        
        selected_refs = [frontal_path]
        remaining = set(all_paths) - {frontal_path}
        
        # 2. Iteratively pick geographically furthest image from selected
        while len(selected_refs) < n_ref and remaining:
            best_dist = -1
            best_candidate = None
            
            for candidate in remaining:
                # Distance is the minimum distance to any already selected reference
                dist_to_set = min(PoseSelector.angular_distance(pose_data[candidate], pose_data[ref]) 
                                  for ref in selected_refs)
                
                if dist_to_set > best_dist:
                    best_dist = dist_to_set
                    best_candidate = candidate
                    
            selected_refs.append(best_candidate)
            remaining.remove(best_candidate)
            
        probe_paths = list(remaining)
        
        return selected_refs, probe_paths
