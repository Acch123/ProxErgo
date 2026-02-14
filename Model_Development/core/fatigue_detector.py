"""
Fatigue Detector Module

PERCLOS-based fatigue detection using Eye Aspect Ratio (EAR).
EAR = (vertical_a + vertical_b) / (2 * horizontal). Eyes closed when EAR < threshold.

References:
- Wierwille et al. (1994): PERCLOS
- Soukupová & Čech (2016): Eye blink detection
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
from enum import Enum


# -----------------------------------------------------------------------------
# Enums & Data Classes
# -----------------------------------------------------------------------------

class FatigueLevel(Enum):
    """Fatigue level classification."""
    ALERT = "alert"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class FatigueMetrics:
    """Comprehensive fatigue analysis results."""
    perclos: float
    blink_rate: float
    avg_blink_duration: float
    fatigue_level: FatigueLevel
    ear_current: float
    
    @property
    def needs_break(self) -> bool:
        return self.fatigue_level in [FatigueLevel.MODERATE, FatigueLevel.SEVERE]
    
    @property
    def severity_score(self) -> int:
        return {FatigueLevel.ALERT: 0, FatigueLevel.MILD: 1, 
                FatigueLevel.MODERATE: 2, FatigueLevel.SEVERE: 3}.get(self.fatigue_level, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'perclos': self.perclos,
            'blink_rate': self.blink_rate,
            'avg_blink_duration': self.avg_blink_duration,
            'fatigue_level': self.fatigue_level.value,
            'ear_current': self.ear_current,
            'needs_break': self.needs_break,
            'severity_score': self.severity_score
        }


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------

class FatigueDetector:
    """PERCLOS + blink-based fatigue detection. 60s window by default."""
    DEFAULT_EAR_THRESHOLD = 0.2
    BLINK_MIN_DURATION_MS = 100
    BLINK_MAX_DURATION_MS = 400
    
    def __init__(self, window_seconds: float = 60.0, ear_threshold: float = 0.2):
        self.window_ms = window_seconds * 1000
        self.ear_threshold = ear_threshold
        self.eye_state_history: deque = deque()
        self.blinks: List[Tuple[float, float]] = []
        self._current_blink_start: Optional[float] = None
        self._last_ear = 1.0
        self._ear_baseline: Optional[float] = None
        self._calibration_samples: List[float] = []
        
    @staticmethod
    def calculate_ear(eye_points: List[Tuple[float, float]]) -> float:
        """EAR = (vertical_a + vertical_b) / (2 * horizontal)."""
        if len(eye_points) < 6:
            return 1.0
        
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_a = np.hypot(p2[0] - p6[0], p2[1] - p6[1])
        vertical_b = np.hypot(p3[0] - p5[0], p3[1] - p5[1])
        horizontal = np.hypot(p1[0] - p4[0], p1[1] - p4[1])
        
        return (vertical_a + vertical_b) / (2.0 * horizontal) if horizontal != 0 else 1.0
    
    def calibrate(self, ear_value: float, num_samples: int = 30):
        """Calibrate EAR baseline for individual user."""
        self._calibration_samples.append(ear_value)
        if len(self._calibration_samples) >= num_samples:
            self._ear_baseline = np.percentile(self._calibration_samples, 25)
            self.ear_threshold = self._ear_baseline * 0.6
            self._calibration_samples = []
    
    @property
    def is_calibrated(self) -> bool:
        return self._ear_baseline is not None
    
    def add_frame(self, left_eye_points: List[Tuple[float, float]], 
                  right_eye_points: List[Tuple[float, float]], timestamp: float):
        """Process a frame's eye data."""
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2
        
        self._last_ear = avg_ear
        eyes_closed = avg_ear < self.ear_threshold
        
        self.eye_state_history.append({'closed': eyes_closed, 'ear': avg_ear, 'timestamp': timestamp})
        if eyes_closed and self._current_blink_start is None:
            self._current_blink_start = timestamp
        elif not eyes_closed and self._current_blink_start is not None:
            duration = timestamp - self._current_blink_start
            if self.BLINK_MIN_DURATION_MS <= duration <= self.BLINK_MAX_DURATION_MS:
                self.blinks.append((self._current_blink_start, timestamp))
            self._current_blink_start = None
        cutoff = timestamp - self.window_ms
        while self.eye_state_history and self.eye_state_history[0]['timestamp'] < cutoff:
            self.eye_state_history.popleft()
        self.blinks = [(s, e) for s, e in self.blinks if e > cutoff]
    
    def analyze(self) -> Optional[FatigueMetrics]:
        """Calculate fatigue metrics."""
        if len(self.eye_state_history) < 100:
            return None
        
        closed_count = sum(1 for s in self.eye_state_history if s['closed'])
        perclos = (closed_count / len(self.eye_state_history)) * 100
        
        window_duration_min = (self.eye_state_history[-1]['timestamp'] - 
                               self.eye_state_history[0]['timestamp']) / 60000 if len(self.eye_state_history) > 1 else 1.0
        
        blink_rate = len(self.blinks) / max(window_duration_min, 0.1)
        avg_blink_duration = np.mean([e - s for s, e in self.blinks]) if self.blinks else 0.0
        
        fatigue_level = self._classify_fatigue(perclos, blink_rate, avg_blink_duration)
        
        return FatigueMetrics(
            perclos=round(perclos, 2),
            blink_rate=round(blink_rate, 1),
            avg_blink_duration=round(avg_blink_duration, 1),
            fatigue_level=fatigue_level,
            ear_current=round(self._last_ear, 3)
        )
    
    def _classify_fatigue(self, perclos: float, blink_rate: float, avg_blink_duration: float) -> FatigueLevel:
        """Score from PERCLOS, blink rate, and blink duration."""
        score = 0
        
        if perclos > 40: score += 3
        elif perclos > 25: score += 2
        elif perclos > 15: score += 1
        
        if blink_rate < 8 or blink_rate > 30: score += 2
        elif blink_rate < 12 or blink_rate > 25: score += 1
        
        if avg_blink_duration > 300: score += 2
        elif avg_blink_duration > 200: score += 1
        
        if score >= 5: return FatigueLevel.SEVERE
        elif score >= 3: return FatigueLevel.MODERATE
        elif score >= 1: return FatigueLevel.MILD
        return FatigueLevel.ALERT
    
    def reset(self):
        """Clear all history."""
        self.eye_state_history.clear()
        self.blinks = []
        self._current_blink_start = None
