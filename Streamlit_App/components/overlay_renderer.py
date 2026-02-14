"""
Overlay Renderer

Draws skeleton, CVA line, metrics panel, calibration progress, and fatigue
on video frames. Uses OpenCV.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Model_Development.core.biomechanics import BiomechanicsResult, PostureClassification


class OverlayRenderer:
    """OpenCV overlay renderer for posture visualization."""
    
    COLORS = {
        'optimal': (136, 255, 0), 'warning': (0, 165, 255), 'danger': (0, 0, 255),
        'skeleton': (255, 212, 0), 'reference': (0, 255, 0), 'text_bg': (30, 30, 30),
        'text': (255, 255, 255), 'privacy': (136, 255, 0), 'accent': (255, 0, 212)
    }
    
    def __init__(self, show_skeleton: bool = True, show_metrics: bool = True):
        self.show_skeleton = show_skeleton
        self.show_metrics = show_metrics
        
    def render(self, frame: np.ndarray, landmarks: Dict, bio_result: BiomechanicsResult,
               fatigue_metrics: Optional[Dict] = None, calibration_progress: Optional[float] = None,
               is_calibrated: bool = False) -> np.ndarray:
        output = frame.copy()
        h, w = output.shape[:2]
        
        if calibration_progress is not None and calibration_progress < 1.0:
            return self._draw_calibration(output, calibration_progress, w, h)
        
        if self.show_skeleton:
            output = self._draw_skeleton(output, landmarks, bio_result, w, h)
        if self.show_metrics:
            output = self._draw_metrics(output, bio_result, fatigue_metrics, w, h, is_calibrated)
        cv2.putText(output, "LANDMARKS ONLY", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['privacy'], 1)
        return output
    
    def _draw_skeleton(self, frame, landmarks, bio_result, w, h):
        def to_px(p): return (int(p[0] * w), int(p[1] * h))
        
        nose = to_px(landmarks.get('nose', (0.5, 0.3, 0, 0)))
        l_shoulder = to_px(landmarks.get('left_shoulder', (0.3, 0.5, 0, 0)))
        r_shoulder = to_px(landmarks.get('right_shoulder', (0.7, 0.5, 0, 0)))
        l_ear = to_px(landmarks.get('left_ear', (0.35, 0.25, 0, 0)))
        r_ear = to_px(landmarks.get('right_ear', (0.65, 0.25, 0, 0)))
        
        color = self._get_color(bio_result.cva_classification)
        cv2.line(frame, l_shoulder, r_shoulder, self.COLORS['skeleton'], 3)
        shoulder_mid = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
        cv2.line(frame, shoulder_mid, (shoulder_mid[0], shoulder_mid[1] - 100), self.COLORS['reference'], 1)
        ear_mid = ((l_ear[0] + r_ear[0]) // 2, (l_ear[1] + r_ear[1]) // 2)
        cv2.line(frame, shoulder_mid, ear_mid, color, 3)
        for p in [nose, l_shoulder, r_shoulder, l_ear, r_ear]:
            cv2.circle(frame, p, 6, self.COLORS['skeleton'], -1)
            cv2.circle(frame, p, 8, color, 2)
        cv2.putText(frame, f"{bio_result.cva_angle:.1f}°", (shoulder_mid[0] + 50, shoulder_mid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def _draw_metrics(self, frame, bio_result, fatigue, w, h, is_calibrated: bool = False):
        panel_w, panel_h = 200, 120
        x, y = w - panel_w - 10, 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), self.COLORS['text_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), self.COLORS['accent'], 1)
        
        color = self._get_color(bio_result.cva_classification)
        cv2.putText(frame, f"CVA: {bio_result.cva_angle:.1f}°", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, bio_result.cva_classification.value.replace('_', ' ').title(), 
                    (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if is_calibrated:
            cv2.putText(frame, "Calibrated", (x + 10, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS['optimal'], 1)
        
        if fatigue:
            f_color = self.COLORS['optimal'] if fatigue['fatigue_level'] == 'alert' else self.COLORS['warning']
            cv2.putText(frame, f"Fatigue: {fatigue['fatigue_level'].upper()}", (x + 10, y + 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_color, 1)
        return frame
    
    def _draw_calibration(self, frame, progress, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, "CALIBRATING...", (w//2 - 100, h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['accent'], 2)
        
        bar_w, bar_h = 300, 20
        bar_x, bar_y = (w - bar_w) // 2, h // 2
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLORS['text_bg'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), self.COLORS['optimal'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLORS['accent'], 1)
        
        cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w + 10, bar_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
        return frame
    
    def _get_color(self, classification):
        if classification == PostureClassification.OPTIMAL:
            return self.COLORS['optimal']
        elif classification in [PostureClassification.MILD_FORWARD, PostureClassification.TILTED_LEFT, 
                                PostureClassification.TILTED_RIGHT]:
            return self.COLORS['warning']
        return self.COLORS['danger']
