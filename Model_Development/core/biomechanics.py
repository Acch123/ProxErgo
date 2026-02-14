"""
Biomechanics Analysis Module
Front-view posture analysis using pose landmarks. Computes CVA proxy, depth,
vertical metrics, and classifies posture (optimal, mild/severe forward head,
tilted, too close/far).
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
from enum import Enum

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_CALIBRATION_FILE = Path(__file__).parent.parent.parent / "data" / "calibration.json"


# -----------------------------------------------------------------------------
# Enums & Data Classes
# -----------------------------------------------------------------------------

class PostureClassification(Enum):
    """Posture classification categories (7-class system)."""
    OPTIMAL = "optimal"
    MILD_FORWARD = "forward_head_mild"
    SEVERE_FORWARD = "forward_head_severe"
    TILTED_LEFT = "tilted_left"
    TILTED_RIGHT = "tilted_right"
    TOO_CLOSE = "too_close"
    TOO_FAR = "too_far"


@dataclass
class BiomechanicsResult:
    """Biomechanical analysis output."""
    cva_angle: float
    cva_classification: PostureClassification
    shoulder_tilt_angle: float
    shoulder_distance_ratio: float
    head_lateral_offset: float
    head_depth_offset: float
    slouch_indicator: float
    vertical_ear_shoulder_ratio: float
    head_depth_delta: float
    overall_confidence: float
    
    @property
    def needs_correction(self) -> bool:
        """Check if posture needs correction."""
        return self.cva_classification not in [PostureClassification.OPTIMAL]
    
    @property
    def severity_score(self) -> int:
        """Get severity score (0-2)."""
        severity_map = {
            PostureClassification.OPTIMAL: 0,
            PostureClassification.MILD_FORWARD: 1,
            PostureClassification.TILTED_LEFT: 1,
            PostureClassification.TILTED_RIGHT: 1,
            PostureClassification.TOO_CLOSE: 1,
            PostureClassification.TOO_FAR: 1,
            PostureClassification.SEVERE_FORWARD: 2,
        }
        return severity_map.get(self.cva_classification, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cva_angle': self.cva_angle,
            'cva_classification': self.cva_classification.value,
            'shoulder_tilt_angle': self.shoulder_tilt_angle,
            'shoulder_distance_ratio': self.shoulder_distance_ratio,
            'head_lateral_offset': self.head_lateral_offset,
            'head_depth_offset': self.head_depth_offset,
            'slouch_indicator': self.slouch_indicator,
            'vertical_ear_shoulder_ratio': self.vertical_ear_shoulder_ratio,
            'head_depth_delta': self.head_depth_delta,
            'overall_confidence': self.overall_confidence,
            'needs_correction': self.needs_correction,
            'severity_score': self.severity_score
        }


class BiomechanicsAnalyzer:
    """
    Front-view posture analyzer using pose landmarks.

    Computes CVA proxy (arctan2 of horizontal ear-shoulder displacement over
    vertical), depth (nose vs shoulders), vertical ear-shoulder distance, and
    slouch. Supports per-user calibration for personalized baselines.
    """

    CVA_THRESHOLDS = {'optimal': 5.0, 'mild': 12.0, 'moderate': 20.0}

    def __init__(self):
        self._baseline_shoulder_distance: Optional[float] = None
        self._baseline_vertical_ear_shoulder: Optional[float] = None
        self._baseline_head_depth: Optional[float] = None
        self._baseline_tilt: Optional[float] = None
        self._calib_shoulder_dists: List[float] = []
        self._calib_verticals: List[float] = []
        self._calib_depths: List[float] = []
        self._calib_tilts: List[float] = []
        self._is_calibrated = False

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    def calibrate(
        self,
        nose: Tuple[float, float, float, float],
        left_ear: Tuple[float, float, float, float],
        right_ear: Tuple[float, float, float, float],
        left_shoulder: Tuple[float, float, float, float],
        right_shoulder: Tuple[float, float, float, float],
        num_samples: int = 30
    ):
        """Record baseline posture from landmarks. Call while user sits upright."""
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_mid_z = (left_shoulder[2] + right_shoulder[2]) / 2
        
        shoulder_distance = np.hypot(
            right_shoulder[0] - left_shoulder[0],
            right_shoulder[1] - left_shoulder[1]
        )
        ear_mid_y = (left_ear[1] + right_ear[1]) / 2
        vertical_ear_shoulder = shoulder_mid_y - ear_mid_y
        head_depth = shoulder_mid_z - nose[2]
        shoulder_x_diff = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_y_diff = right_shoulder[1] - left_shoulder[1]
        if shoulder_x_diff > 0.001:
            shoulder_tilt = np.degrees(np.arctan2(shoulder_y_diff, shoulder_x_diff))
        else:
            shoulder_tilt = 0.0

        self._calib_shoulder_dists.append(shoulder_distance)
        self._calib_verticals.append(vertical_ear_shoulder)
        self._calib_depths.append(head_depth)
        self._calib_tilts.append(shoulder_tilt)

        if len(self._calib_shoulder_dists) >= num_samples:
            self._baseline_shoulder_distance = float(np.median(self._calib_shoulder_dists))
            self._baseline_vertical_ear_shoulder = float(np.median(self._calib_verticals))
            self._baseline_head_depth = float(np.median(self._calib_depths))
            self._baseline_tilt = float(np.median(self._calib_tilts))
            
            self._is_calibrated = True
            
            # Clear buffers
            self._calib_shoulder_dists.clear()
            self._calib_verticals.clear()
            self._calib_depths.clear()
            self._calib_tilts.clear()

    def save_calibration(self, path: Optional[Path] = None) -> bool:
        """Persist calibration to JSON. Used by Streamlit app across processes."""
        if not self._is_calibrated:
            return False
        p = path or _CALIBRATION_FILE
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({
                    "baseline_shoulder_distance": self._baseline_shoulder_distance,
                    "baseline_vertical_ear_shoulder": self._baseline_vertical_ear_shoulder,
                    "baseline_head_depth": self._baseline_head_depth,
                    "baseline_tilt": self._baseline_tilt,
                }, f, indent=2)
            return True
        except Exception:
            return False
    
    def load_calibration(self, path: Optional[Path] = None) -> bool:
        """Load calibration from JSON. Returns True if loaded."""
        p = path or _CALIBRATION_FILE
        if not p.exists():
            return False
        try:
            with open(p) as f:
                d = json.load(f)
            self._baseline_shoulder_distance = d.get("baseline_shoulder_distance")
            self._baseline_vertical_ear_shoulder = d.get("baseline_vertical_ear_shoulder")
            self._baseline_head_depth = d.get("baseline_head_depth")
            self._baseline_tilt = d.get("baseline_tilt")
            self._is_calibrated = bool(self._baseline_vertical_ear_shoulder)
            return self._is_calibrated
        except Exception:
            return False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    @property
    def calibration_progress(self) -> float:
        """Return calibration progress (0-1)."""
        if self._is_calibrated:
            return 1.0
        return len(self._calib_shoulder_dists) / 30.0

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def analyze(
        self,
        nose: Tuple[float, float, float, float],
        left_ear: Tuple[float, float, float, float],
        right_ear: Tuple[float, float, float, float],
        left_shoulder: Tuple[float, float, float, float],
        right_shoulder: Tuple[float, float, float, float]
    ) -> BiomechanicsResult:
        """Run biomechanical analysis on pose landmarks."""
        # Midpoints & distances
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_mid_z = (left_shoulder[2] + right_shoulder[2]) / 2
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2
        ear_mid_y = (left_ear[1] + right_ear[1]) / 2
        ear_mid_z = (left_ear[2] + right_ear[2]) / 2
        shoulder_distance = np.hypot(
            right_shoulder[0] - left_shoulder[0],
            right_shoulder[1] - left_shoulder[1]
        )

        # Depth: shoulder_mid_z - nose[2] (positive = nose forward)
        head_depth_offset = shoulder_mid_z - nose[2]

        # Slouch: max(0, expected_height - nose_y_relative) / shoulder_distance
        nose_y_relative = shoulder_mid_y - nose[1]
        expected_height = shoulder_distance * 0.8
        slouch_indicator = np.clip(
            max(0, expected_height - nose_y_relative) / max(shoulder_distance, 0.001),
            0, 1
        )
        vertical_ear_shoulder = shoulder_mid_y - ear_mid_y

        if self._baseline_vertical_ear_shoulder and self._baseline_vertical_ear_shoulder > 0:
            vertical_ratio = (self._baseline_vertical_ear_shoulder - vertical_ear_shoulder) / self._baseline_vertical_ear_shoulder
        else:
            vertical_ratio = 0.0
        
        if self._baseline_head_depth is not None:
            head_depth_delta = head_depth_offset - self._baseline_head_depth
        else:
            head_depth_delta = 0.0

        # CVA proxy: arctan2(|ear_x - shoulder_x|, shoulder_y - ear_y)
        delta_x = ear_mid_x - shoulder_mid_x
        delta_y = shoulder_mid_y - ear_mid_y
        
        if delta_y <= 0:
            delta_y = 0.001
        
        cva_angle = np.degrees(np.arctan2(abs(delta_x), delta_y))
        if self._baseline_shoulder_distance and self._baseline_shoulder_distance > 0:
            depth_ratio = shoulder_distance / self._baseline_shoulder_distance
            depth_adjustment = np.clip(2 - depth_ratio, 0.8, 1.5)
            cva_angle *= depth_adjustment

        # Shoulder tilt: positive = right shoulder lower = tilted left
        shoulder_x_diff = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_y_diff = right_shoulder[1] - left_shoulder[1]
        
        if shoulder_x_diff > 0.001:
            shoulder_tilt = np.degrees(np.arctan2(shoulder_y_diff, shoulder_x_diff))
        else:
            shoulder_tilt = 0.0
        tilt_delta = shoulder_tilt - self._baseline_tilt if self._baseline_tilt is not None else shoulder_tilt
        abs_tilt = abs(tilt_delta)
        tilt_threshold = 7.0 if self._baseline_tilt is not None else 10.0
        is_significantly_tilted = abs_tilt > tilt_threshold

        # Distance check (normalized shoulder width)
        if shoulder_distance > 0.6:
            cva_class = PostureClassification.TOO_CLOSE
        elif shoulder_distance < 0.15:
            cva_class = PostureClassification.TOO_FAR
        else:
            forward_score = 0.0

            if self._baseline_vertical_ear_shoulder and self._baseline_vertical_ear_shoulder > 0:
                if head_depth_delta > 0.15:
                    forward_score += min(head_depth_delta * 1.3, 0.5)
                if not is_significantly_tilted and vertical_ratio > 0.02:
                    forward_score += min(vertical_ratio * 3.2, 0.55)
            else:
                if head_depth_offset > 0.25:
                    forward_score += min(head_depth_offset * 1.0, 0.35)
                if not is_significantly_tilted:
                    vertical_abs_ratio = vertical_ear_shoulder / max(shoulder_distance, 0.001)
                    if vertical_abs_ratio < 0.48:
                        forward_score += min((0.48 - vertical_abs_ratio) * 1.8, 0.5)

            if not is_significantly_tilted:
                if self._baseline_vertical_ear_shoulder and self._baseline_vertical_ear_shoulder > 0:
                    if cva_angle >= 12.0:
                        forward_score += min((cva_angle - 12) / 28.0, 0.3)
                else:
                    if cva_angle >= 10.0:
                        forward_score += min((cva_angle - 10) / 25.0, 0.35)
            if slouch_indicator > 0.45:
                forward_score += (slouch_indicator - 0.45) * 0.25

            if forward_score < 0.25:
                cva_class = PostureClassification.OPTIMAL
            elif forward_score < 0.65:
                cva_class = PostureClassification.MILD_FORWARD
            else:
                cva_class = PostureClassification.SEVERE_FORWARD

            if is_significantly_tilted and forward_score < 0.45:
                cva_class = PostureClassification.TILTED_LEFT if tilt_delta > 0 else PostureClassification.TILTED_RIGHT

        head_lateral_offset = np.clip((nose[0] - shoulder_mid_x) * 2, -1, 1)
        min_visibility = min(nose[3], left_ear[3], right_ear[3], left_shoulder[3], right_shoulder[3])
        
        return BiomechanicsResult(
            cva_angle=round(cva_angle, 2),
            cva_classification=cva_class,
            shoulder_tilt_angle=round(shoulder_tilt, 2),
            shoulder_distance_ratio=round(shoulder_distance, 4),
            head_lateral_offset=round(head_lateral_offset, 3),
            head_depth_offset=round(head_depth_offset, 3),
            slouch_indicator=round(slouch_indicator, 3),
            vertical_ear_shoulder_ratio=round(vertical_ratio, 3),
            head_depth_delta=round(head_depth_delta, 3),
            overall_confidence=round(min_visibility, 3)
        )


# -----------------------------------------------------------------------------
# Postural Sway
# -----------------------------------------------------------------------------

@dataclass
class SwayMetrics:
    """Postural sway analysis metrics."""
    amplitude: float
    path_length: float
    velocity_rms: float
    frequency: float
    fatigue_index: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'amplitude': self.amplitude,
            'path_length': self.path_length,
            'velocity_rms': self.velocity_rms,
            'frequency': self.frequency,
            'fatigue_index': self.fatigue_index,
            'sample_count': self.sample_count
        }


class PosturalSwayAnalyzer:
    """
    Postural sway from nose position. Tracks amplitude, path length, velocity,
    and frequency. Used as fatigue marker (Paillard 2012).
    """
    def __init__(self, window_seconds: float = 30.0, sample_rate: float = 30.0):
        self.window_size = int(window_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.position_history: deque = deque(maxlen=self.window_size)
        
    def add_sample(self, nose_x: float, nose_y: float, timestamp: float):
        """Add a nose position sample."""
        self.position_history.append((nose_x, nose_y, timestamp))
    
    def analyze(self) -> Optional[SwayMetrics]:
        """Calculate sway metrics from position history."""
        min_samples = int(self.sample_rate * 5)
        
        if len(self.position_history) < min_samples:
            return None
        
        positions = np.array([(p[0], p[1]) for p in self.position_history])
        timestamps = np.array([p[2] for p in self.position_history])
        
        center = positions.mean(axis=0)
        displacements = np.linalg.norm(positions - center, axis=1)
        amplitude = np.percentile(displacements, 95)
        
        diffs = np.diff(positions, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        
        if len(timestamps) > 1:
            dt = np.diff(timestamps) / 1000
            dt[dt <= 0] = 1 / self.sample_rate
            velocities = np.linalg.norm(diffs, axis=1) / dt
            velocity_rms = np.sqrt(np.mean(velocities ** 2))
        else:
            velocity_rms = 0.0
        
        frequency = self._estimate_frequency(displacements)
        fatigue_index = self._calculate_fatigue_index(amplitude, frequency, velocity_rms)
        
        return SwayMetrics(
            amplitude=round(amplitude, 6),
            path_length=round(path_length, 4),
            velocity_rms=round(velocity_rms, 6),
            frequency=round(frequency, 3),
            fatigue_index=round(fatigue_index, 1),
            sample_count=len(self.position_history)
        )
    
    def _estimate_frequency(self, signal: np.ndarray) -> float:
        """Estimate dominant frequency using zero-crossing method."""
        mean_val = np.mean(signal)
        crossings = np.sum(np.diff(np.sign(signal - mean_val)) != 0)
        duration_sec = len(self.position_history) / self.sample_rate
        return crossings / (2 * duration_sec) if duration_sec > 0 else 0.0
    
    def _calculate_fatigue_index(self, amplitude: float, frequency: float, velocity_rms: float) -> float:
        """Calculate fatigue index based on sway characteristics."""
        amp_score = min(amplitude * 1000, 50)
        freq_score = max(0, 25 - frequency * 12.5)
        vel_score = min(velocity_rms * 500, 25)
        return np.clip(amp_score + freq_score + vel_score, 0, 100)
    
    def reset(self):
        """Clear the position history."""
        self.position_history.clear()
