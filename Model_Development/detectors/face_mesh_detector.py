"""
Face Mesh Detector Module
MediaPipe Face Landmarker for eye tracking and fatigue detection.
Uses the new MediaPipe Tasks API (0.10+).
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import urllib.request
import ssl
import certifi

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class EyeLandmarks:
    """Eye landmarks for EAR calculation."""
    left_eye: List[Tuple[float, float]]
    right_eye: List[Tuple[float, float]]
    timestamp: float
    
    @property
    def is_valid(self) -> bool:
        return len(self.left_eye) >= 6 and len(self.right_eye) >= 6
    
    def to_dict(self) -> Dict[str, List[Tuple[float, float]]]:
        return {'left_eye': self.left_eye, 'right_eye': self.right_eye}


class FaceMeshDetector:
    """MediaPipe Face Landmarker for eye tracking."""
    
    # Eye landmark indices for EAR calculation (MediaPipe Face Mesh)
    # These indices work with the 478-point face mesh
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    
    # Model URL
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
    
    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, refine_landmarks: bool = True):
        """
        Initialize the face mesh detector using MediaPipe Tasks API.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            refine_landmarks: Whether to refine eye/lip landmarks
        """
        # Ensure model is downloaded
        self._ensure_model()
        
        # Create options
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.MODEL_PATH)
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,  # We only need geometry
            output_facial_transformation_matrixes=False
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._detection_count = 0
        self._last_timestamp_ms = 0
    
    def _ensure_model(self):
        """Download model if not present."""
        if self.MODEL_PATH.exists():
            return
        
        print(f"📥 Downloading face model...")
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            with urllib.request.urlopen(self.MODEL_URL, context=ssl_context) as response:
                with open(self.MODEL_PATH, 'wb') as f:
                    f.write(response.read())
            
            print(f"✅ Model saved to {self.MODEL_PATH}")
        except Exception as e:
            # Fallback: try without SSL verification
            try:
                print("⚠️  SSL verification failed, trying without verification...")
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                with urllib.request.urlopen(self.MODEL_URL, context=ssl_context) as response:
                    with open(self.MODEL_PATH, 'wb') as f:
                        f.write(response.read())
                print(f"✅ Model saved to {self.MODEL_PATH}")
            except Exception as e2:
                raise RuntimeError(f"Failed to download model: {e2}\n"
                                 f"Please manually download from:\n{self.MODEL_URL}\n"
                                 f"And save to: {self.MODEL_PATH}")
        
    def detect(self, frame: np.ndarray, timestamp: float) -> Optional[EyeLandmarks]:
        """Extract eye landmarks for fatigue analysis."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Ensure monotonic timestamp
        timestamp_ms = int(timestamp)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        
        # Detect face
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None
        
        self._detection_count += 1
        face_landmarks = result.face_landmarks[0]  # First face
        
        # Extract eye landmarks
        def extract_eye(indices: List[int]) -> List[Tuple[float, float]]:
            points = []
            for i in indices:
                if i < len(face_landmarks):
                    lm = face_landmarks[i]
                    points.append((lm.x, lm.y))
                else:
                    points.append((0.5, 0.5))  # Default if index out of range
            return points
        
        return EyeLandmarks(
            left_eye=extract_eye(self.LEFT_EYE_IDX),
            right_eye=extract_eye(self.RIGHT_EYE_IDX),
            timestamp=timestamp
        )
    
    @property
    def detection_count(self) -> int:
        return self._detection_count
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
