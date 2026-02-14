"""
Posture Classifier Module

XGBoost-based posture classification with rule-based fallback. Uses CVA, depth,
vertical, and slouch features. 7 classes: optimal, mild/severe forward head,
tilted left/right, too close/far.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class PostureFeatures:
    """Feature vector for posture classification (10 features)."""
    cva_angle: float
    shoulder_tilt: float
    head_lateral_offset: float
    shoulder_distance: float
    nose_y_relative: float
    ear_shoulder_ratio: float
    nose_depth_relative: float
    ear_depth_relative: float
    vertical_ratio: float
    head_drop: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.cva_angle, self.shoulder_tilt, self.head_lateral_offset,
            self.shoulder_distance, self.nose_y_relative, self.ear_shoulder_ratio,
            self.nose_depth_relative, self.ear_depth_relative,
            self.vertical_ratio, self.head_drop
        ])
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cva_angle': self.cva_angle, 
            'shoulder_tilt': self.shoulder_tilt,
            'head_lateral_offset': self.head_lateral_offset,
            'shoulder_distance': self.shoulder_distance,
            'nose_y_relative': self.nose_y_relative, 
            'ear_shoulder_ratio': self.ear_shoulder_ratio,
            'nose_depth_relative': self.nose_depth_relative,
            'ear_depth_relative': self.ear_depth_relative,
            'vertical_ratio': self.vertical_ratio,
            'head_drop': self.head_drop
        }


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------

class PostureClassifierML:
    """XGBoost posture classifier with rule-based fallback when untrained."""
    LABELS = ['optimal', 'forward_head_mild', 'forward_head_severe',
              'tilted_left', 'tilted_right', 'too_close', 'too_far']
    
    FEATURE_NAMES = [
        'cva_angle', 'shoulder_tilt', 'head_lateral_offset',
        'shoulder_distance', 'nose_y_relative', 'ear_shoulder_ratio',
        'nose_depth_relative', 'ear_depth_relative',
        'vertical_ratio', 'head_drop'
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self._is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ))
        ])
    
    def extract_features(self, nose, left_ear, right_ear, left_shoulder, right_shoulder) -> PostureFeatures:
        """Extract 10 features from pose landmarks. Landmarks: (x, y, z, visibility)."""
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_mid_z = (left_shoulder[2] + right_shoulder[2]) / 2
        
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2
        ear_mid_y = (left_ear[1] + right_ear[1]) / 2
        ear_mid_z = (left_ear[2] + right_ear[2]) / 2

        delta_x = ear_mid_x - shoulder_mid_x
        delta_y = shoulder_mid_y - ear_mid_y if shoulder_mid_y > ear_mid_y else 0.001
        cva_angle = np.degrees(np.arctan2(abs(delta_x), delta_y))
        shoulder_x_diff = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_y_diff = right_shoulder[1] - left_shoulder[1]
        shoulder_tilt = np.degrees(np.arctan2(shoulder_y_diff, shoulder_x_diff)) if shoulder_x_diff > 0.001 else 0.0
        head_lateral_offset = np.clip((nose[0] - shoulder_mid_x) * 2, -1, 1)
        shoulder_distance = np.hypot(
            right_shoulder[0] - left_shoulder[0], 
            right_shoulder[1] - left_shoulder[1]
        )
        ear_distance = np.hypot(
            right_ear[0] - left_ear[0], 
            right_ear[1] - left_ear[1]
        )
        nose_y_relative = shoulder_mid_y - nose[1]
        ear_shoulder_ratio = ear_distance / max(shoulder_distance, 0.001)
        nose_depth_relative = nose[2] - shoulder_mid_z
        ear_depth_relative = ear_mid_z - shoulder_mid_z
        vertical_ratio = nose_y_relative / max(shoulder_distance, 0.001)
        expected_head_height = shoulder_distance * 0.8
        head_drop = max(0, expected_head_height - nose_y_relative) / max(shoulder_distance, 0.001)
        
        return PostureFeatures(
            cva_angle=cva_angle,
            shoulder_tilt=shoulder_tilt,
            head_lateral_offset=head_lateral_offset,
            shoulder_distance=shoulder_distance,
            nose_y_relative=nose_y_relative,
            ear_shoulder_ratio=ear_shoulder_ratio,
            nose_depth_relative=nose_depth_relative,
            ear_depth_relative=ear_depth_relative,
            vertical_ratio=vertical_ratio,
            head_drop=head_drop
        )
    
    def predict(self, features: PostureFeatures) -> Tuple[str, float, Dict[str, float]]:
        """Predict posture class with confidence."""
        if not self._is_trained:
            return self._rule_based_predict(features)
        
        X = features.to_array().reshape(1, -1)
        prediction_idx = self.pipeline.predict(X)[0]
        probabilities = self.pipeline.predict_proba(X)[0]
        
        return self.LABELS[prediction_idx], float(np.max(probabilities)), \
               {label: prob for label, prob in zip(self.LABELS, probabilities)}
    
    def _rule_based_predict(self, features: PostureFeatures) -> Tuple[str, float, Dict[str, float]]:
        """Rule-based fallback when model is not trained."""
        probs = {label: 0.0 for label in self.LABELS}
        if features.shoulder_distance > 0.5:
            return 'too_close', 0.8, {**probs, 'too_close': 0.8}
        elif features.shoulder_distance < 0.12:
            return 'too_far', 0.8, {**probs, 'too_far': 0.8}
        abs_tilt = abs(features.shoulder_tilt)
        is_significantly_tilted = abs_tilt > 10.0
        forward_score = 0.0
        forward_nose = -features.nose_depth_relative
        if forward_nose > 0.25:
            forward_score += min(forward_nose * 1.0, 0.4)
        forward_ear = -features.ear_depth_relative
        if forward_ear > 0.12:
            forward_score += min(forward_ear * 0.5, 0.15)
        if not is_significantly_tilted and features.cva_angle >= 12:
            forward_score += min((features.cva_angle - 12) / 28.0, 0.25)
        if not is_significantly_tilted:
            if features.vertical_ratio < 0.6:
                forward_score += min((0.6 - features.vertical_ratio) * 0.85, 0.35)
            if features.head_drop > 0.08:
                forward_score += min(features.head_drop * 0.5, 0.25)
        if is_significantly_tilted and forward_score < 0.45:
            label = 'tilted_left' if features.shoulder_tilt > 0 else 'tilted_right'
            return label, 0.7, {**probs, label: 0.7}
        if forward_score < 0.25:
            return 'optimal', 0.85, {**probs, 'optimal': 0.85}
        elif forward_score < 0.65:
            return 'forward_head_mild', 0.75, {**probs, 'forward_head_mild': 0.75}
        else:
            return 'forward_head_severe', 0.8, {**probs, 'forward_head_severe': 0.8}
    
    def train(self, features: List[PostureFeatures], labels: List[str], validate: bool = True) -> Dict[str, Any]:
        """Train the classifier."""
        X = np.array([f.to_array() for f in features])
        y = np.array([self.LABELS.index(l) for l in labels])
        
        self.pipeline.fit(X, y)
        self._is_trained = True
        
        metrics = {'num_samples': len(features), 'num_classes': len(set(labels))}
        if validate and len(features) >= 10:
            cv_scores = cross_val_score(self.pipeline, X, y, cv=min(5, len(features)))
            metrics['cv_accuracy_mean'] = float(np.mean(cv_scores))
        
        return metrics
    
    def save_model(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'is_trained': self._is_trained}, path)
    
    def load_model(self, path: Path):
        data = joblib.load(path)
        self.pipeline = data['pipeline']
        self._is_trained = data.get('is_trained', True)
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
