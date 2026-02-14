"""
Pose Detector Module
MediaPipe Pose Landmarker wrapper with privacy-first design.
Uses the new MediaPipe Tasks API (0.10+).
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path
import urllib.request
import ssl
import certifi

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class PoseLandmarks:
    """Extracted pose landmarks with visibility scores."""
    nose: Tuple[float, float, float, float]
    left_eye_inner: Tuple[float, float, float, float]
    left_eye: Tuple[float, float, float, float]
    left_eye_outer: Tuple[float, float, float, float]
    right_eye_inner: Tuple[float, float, float, float]
    right_eye: Tuple[float, float, float, float]
    right_eye_outer: Tuple[float, float, float, float]
    left_ear: Tuple[float, float, float, float]
    right_ear: Tuple[float, float, float, float]
    mouth_left: Tuple[float, float, float, float]
    mouth_right: Tuple[float, float, float, float]
    left_shoulder: Tuple[float, float, float, float]
    right_shoulder: Tuple[float, float, float, float]
    timestamp: float
    
    @property
    def is_valid(self) -> bool:
        return self.nose[3] > 0.5 and self.left_shoulder[3] > 0.5 and self.right_shoulder[3] > 0.5
    
    @property
    def average_visibility(self) -> float:
        visibilities = [self.nose[3], self.left_eye[3], self.right_eye[3],
                        self.left_ear[3], self.right_ear[3], self.left_shoulder[3], self.right_shoulder[3]]
        return sum(visibilities) / len(visibilities)
    
    def to_dict(self) -> Dict[str, Tuple[float, float, float, float]]:
        return {
            'nose': self.nose, 'left_eye_inner': self.left_eye_inner, 'left_eye': self.left_eye,
            'left_eye_outer': self.left_eye_outer, 'right_eye_inner': self.right_eye_inner,
            'right_eye': self.right_eye, 'right_eye_outer': self.right_eye_outer,
            'left_ear': self.left_ear, 'right_ear': self.right_ear,
            'mouth_left': self.mouth_left, 'mouth_right': self.mouth_right,
            'left_shoulder': self.left_shoulder, 'right_shoulder': self.right_shoulder
        }


class PoseDetector:
    """MediaPipe Pose Landmarker - extracts landmarks, never stores video."""
    
    # Landmark indices (MediaPipe Pose Landmarker)
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    # Model URLs
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker.task"
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5, model_complexity: int = 1):
        """
        Initialize the pose detector using MediaPipe Tasks API.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking  
            model_complexity: 0=Lite, 1=Full, 2=Heavy (affects model selection)
        """
        # Ensure model is downloaded
        self._ensure_model()
        
        # Create options
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.MODEL_PATH)
        )
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False  # Privacy: no segmentation
        )
        
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._detection_count = 0
        self._last_timestamp_ms = 0
    
    def _ensure_model(self):
        """Download model if not present."""
        if self.MODEL_PATH.exists():
            return
        
        print(f"📥 Downloading pose model...")
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Download with SSL context
            with urllib.request.urlopen(self.MODEL_URL, context=ssl_context) as response:
                with open(self.MODEL_PATH, 'wb') as f:
                    f.write(response.read())
            
            print(f"✅ Model saved to {self.MODEL_PATH}")
        except Exception as e:
            # Fallback: try without SSL verification (less secure but works)
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
        
    def detect(self, frame: np.ndarray, timestamp: float) -> Optional[PoseLandmarks]:
        """Process frame and return only landmark data (privacy-first)."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Ensure monotonic timestamp (MediaPipe requirement)
        timestamp_ms = int(timestamp)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        
        # Detect pose
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        
        self._detection_count += 1
        landmarks = result.pose_landmarks[0]  # First person
        
        def extract(idx: int) -> Tuple[float, float, float, float]:
            lm = landmarks[idx]
            # Note: Tasks API uses 'presence' instead of 'visibility' for some landmarks
            visibility = getattr(lm, 'visibility', 1.0) if hasattr(lm, 'visibility') else 0.9
            return (lm.x, lm.y, lm.z, visibility)
        
        return PoseLandmarks(
            nose=extract(self.NOSE), left_eye_inner=extract(self.LEFT_EYE_INNER),
            left_eye=extract(self.LEFT_EYE), left_eye_outer=extract(self.LEFT_EYE_OUTER),
            right_eye_inner=extract(self.RIGHT_EYE_INNER), right_eye=extract(self.RIGHT_EYE),
            right_eye_outer=extract(self.RIGHT_EYE_OUTER), left_ear=extract(self.LEFT_EAR),
            right_ear=extract(self.RIGHT_EAR), mouth_left=extract(self.MOUTH_LEFT),
            mouth_right=extract(self.MOUTH_RIGHT), left_shoulder=extract(self.LEFT_SHOULDER),
            right_shoulder=extract(self.RIGHT_SHOULDER), timestamp=timestamp
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
