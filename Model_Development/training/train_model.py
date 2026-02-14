#!/usr/bin/env python3
"""
Model Trainer - Train the posture classification model.

Usage:
    python -m Model_Development.training.train_model

This script:
1. Loads cleaned training data
2. Extracts features from landmarks
3. Trains an XGBoost classifier
4. Evaluates with cross-validation
5. Saves the trained model to models/posture_classifier.pkl
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

from Model_Development.core.posture_classifier import PostureClassifierML, PostureFeatures


class ModelTrainer:
    """
    Train and evaluate the posture classification model.
    
    Uses the existing PostureClassifierML but with actual training data
    instead of rule-based predictions.
    """
    
    # Simplified 7-class system (default)
    DEFAULT_LABELS = [
        'optimal', 'forward_head_mild', 'forward_head_severe',
        'tilted_left', 'tilted_right', 'too_close', 'too_far'
    ]
    
    # Preset label mappings for further simplification
    LABEL_PRESETS = {
        'full': {},  # No remapping (use all 7 classes)
        'minimal': {
            # Merge all forward head into one
            'forward_head_mild': 'poor_posture',
            'forward_head_severe': 'poor_posture',
            'tilted_left': 'poor_posture',
            'tilted_right': 'poor_posture',
        }
    }
    
    def __init__(self, 
                 data_path: str = "training_data/cleaned_data.json",
                 model_output: str = "models/posture_classifier.pkl",
                 label_preset: str = 'full',
                 custom_mapping: Dict[str, str] = None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to cleaned training data
            model_output: Path to save trained model
            label_preset: 'full', 'simplified', or 'minimal'
            custom_mapping: Custom label remapping dict (overrides preset)
        """
        self.data_path = Path(data_path)
        self.model_output = Path(model_output)
        self.model_output.parent.mkdir(exist_ok=True)
        
        # Set up label mapping
        if custom_mapping:
            self.label_mapping = custom_mapping
        else:
            self.label_mapping = self.LABEL_PRESETS.get(label_preset, {})
        
        # Compute effective labels after remapping
        self.LABELS = self._compute_effective_labels()
        
        # Will hold the classifier
        self.classifier = PostureClassifierML()
        
        # Training results
        self.results = {}
    
    def _compute_effective_labels(self) -> List[str]:
        """Compute the final label set after applying remapping."""
        # Start with default labels
        final_labels = set()
        for label in self.DEFAULT_LABELS:
            remapped = self.label_mapping.get(label, label)
            final_labels.add(remapped)
        
        # Sort to maintain consistent ordering
        return sorted(list(final_labels))
    
    def _remap_label(self, label: str) -> str:
        """Apply label mapping to a single label."""
        return self.label_mapping.get(label, label)
    
    def load_data(self) -> Tuple[List[PostureFeatures], List[str]]:
        """
        Load and prepare training data.
        
        Returns:
            Tuple of (features, labels)
        """
        if not self.data_path.exists():
            print(f"❌ Data file not found: {self.data_path}")
            print("   Run the data cleaner first:")
            print("   python -m Model_Development.training.data_cleaner")
            return [], []
        
        print(f"📂 Loading data from {self.data_path}")
        
        with open(self.data_path) as f:
            data = json.load(f)
        
        samples = data.get('samples', data)  # Handle both formats
        
        print(f"   Loaded {len(samples)} samples")
        
        # Show label mapping if active
        if self.label_mapping:
            print(f"\n🏷️  Label Remapping Active:")
            for old, new in self.label_mapping.items():
                print(f"     {old} → {new}")
            print(f"   Final labels: {self.LABELS}")
        
        # Extract features
        features = []
        labels = []
        skipped = 0
        
        for sample in samples:
            try:
                lm = sample['landmarks']
                
                # Use the classifier's feature extraction
                feat = self.classifier.extract_features(
                    nose=tuple(lm['nose']),
                    left_ear=tuple(lm['left_ear']),
                    right_ear=tuple(lm['right_ear']),
                    left_shoulder=tuple(lm['left_shoulder']),
                    right_shoulder=tuple(lm['right_shoulder'])
                )
                
                # Apply label remapping
                original_label = sample['label']
                remapped_label = self._remap_label(original_label)
                
                features.append(feat)
                labels.append(remapped_label)
                
            except Exception as e:
                skipped += 1
        
        if skipped > 0:
            print(f"   ⚠️  Skipped {skipped} samples due to errors")
        
        print(f"   Extracted features for {len(features)} samples")
        
        return features, labels
    
    def analyze_data(self, features: List[PostureFeatures], labels: List[str]):
        """
        Analyze the training data distribution.
        
        Args:
            features: List of extracted features
            labels: List of labels
        """
        print("\n📊 Data Analysis")
        print("="*50)
        
        # Label distribution
        from collections import Counter
        label_counts = Counter(labels)
        
        print("\nLabel Distribution:")
        for label in self.LABELS:
            count = label_counts.get(label, 0)
            bar = "█" * (count // 2)
            print(f"  {label:25s} {count:4d} {bar}")
        
        # Feature statistics
        print("\nFeature Statistics:")
        feature_arrays = np.array([f.to_array() for f in features])
        feature_names = PostureClassifierML.FEATURE_NAMES
        
        for i, name in enumerate(feature_names):
            if i < feature_arrays.shape[1]:
                values = feature_arrays[:, i]
                print(f"  {name:20s} mean={np.mean(values):7.3f}  "
                      f"std={np.std(values):7.3f}  "
                      f"range=[{np.min(values):7.3f}, {np.max(values):7.3f}]")
    
    def train(self, test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the model with cross-validation.
        
        Args:
            test_size: Fraction of data to hold out for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of training results
        """
        # Load data
        features, labels = self.load_data()
        
        if not features:
            return {}
        
        # Analyze data
        self.analyze_data(features, labels)
        
        # Convert to arrays
        X = np.array([f.to_array() for f in features])
        
        # Map labels to indices
        label_to_idx = {label: i for i, label in enumerate(self.LABELS)}
        y = np.array([label_to_idx.get(label, -1) for label in labels])
        
        # Remove unknown labels
        valid_mask = y >= 0
        X = X[valid_mask]
        y = y[valid_mask]
        labels = [l for l, v in zip(labels, valid_mask) if v]
        
        print(f"\n🏋️ Training Model")
        print("="*50)
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Test size: {test_size:.0%}")
        print(f"   CV folds: {cv_folds}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Create pipeline with XGBoost
        pipeline = Pipeline([
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
        
        # Cross-validation
        print("\n   Running cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"   CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Train on full training set (with sample weights for class imbalance)
        print("\n   Training final model...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Test Accuracy: {test_accuracy:.3f}")
        
        # Update the classifier's pipeline
        self.classifier.pipeline = pipeline
        self.classifier._is_trained = True
        
        # Store results
        self.results = {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'test_accuracy': float(test_accuracy),
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'trained_at': datetime.now().isoformat()
        }
        
        # Detailed evaluation
        self._detailed_evaluation(y_test, y_pred, pipeline)
        
        return self.results
    
    def _detailed_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, pipeline):
        """Print detailed evaluation metrics."""
        print("\n📈 Detailed Evaluation")
        print("="*50)
        
        # Classification report
        # Filter to only labels present in the data
        present_labels = sorted(set(y_true) | set(y_pred))
        present_label_names = [self.LABELS[i] for i in present_labels]
        
        print("\nClassification Report:")
        report = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=present_label_names,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        
        # Print header
        header = "Predicted →"
        print(f"\n{'':<20} ", end="")
        for name in present_label_names:
            print(f"{name[:8]:>10}", end="")
        print("\nActual ↓")
        
        for i, row in enumerate(cm):
            print(f"{present_label_names[i]:<20} ", end="")
            for val in row:
                print(f"{val:>10}", end="")
            print()
        
        # Feature importance
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            print("\nFeature Importance:")
            importances = pipeline.named_steps['classifier'].feature_importances_
            feature_names = PostureClassifierML.FEATURE_NAMES
            
            sorted_idx = np.argsort(importances)[::-1]
            for idx in sorted_idx:
                if idx < len(feature_names):
                    bar = "█" * int(importances[idx] * 30)
                    print(f"  {feature_names[idx]:<22} {importances[idx]:.3f} {bar}")
    
    def save_model(self):
        """Save the trained model."""
        if not self.classifier._is_trained:
            print("❌ No trained model to save!")
            return
        
        # Save using the classifier's save method
        self.classifier.save_model(self.model_output)
        
        print(f"\n✅ Model saved to {self.model_output}")
        
        # Also save training results
        results_path = self.model_output.parent / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   Training results saved to {results_path}")
    
    def run(self):
        """Run the complete training pipeline."""
        print("\n" + "="*60)
        print("  🧊 FROSTBYTE MODEL TRAINER")
        print("="*60)
        
        results = self.train()
        
        if results:
            self.save_model()
            
            print("\n" + "="*60)
            print("  ✅ TRAINING COMPLETE!")
            print("="*60)
            print(f"\n  CV Accuracy:   {results['cv_accuracy_mean']:.1%} (±{results['cv_accuracy_std']:.1%})")
            print(f"  Test Accuracy: {results['test_accuracy']:.1%}")
            print(f"\n  Model saved to: {self.model_output}")
            print("\n  To use the trained model in the app:")
            print("  1. The app will automatically load models/posture_classifier.pkl")
            print("  2. Or specify: PostureClassifierML(model_path='models/posture_classifier.pkl')")
        else:
            print("\n❌ Training failed! Check the data and try again.")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the posture classifier")
    parser.add_argument("--data", "-d", default="training_data/cleaned_data.json",
                        help="Path to cleaned training data")
    parser.add_argument("--output", "-o", default="models/posture_classifier.pkl",
                        help="Path to save trained model")
    parser.add_argument("--test-size", "-t", type=float, default=0.2,
                        help="Test set size (0-1)")
    parser.add_argument("--cv-folds", "-k", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--labels", "-l", default="full",
                        choices=['full', 'minimal'],
                        help="Label preset: 'full' (7 classes), "
                             "'minimal' (4 classes: optimal, poor_posture, too_close, too_far)")
    args = parser.parse_args()
    
    trainer = ModelTrainer(data_path=args.data, model_output=args.output, 
                          label_preset=args.labels)
    trainer.run()


if __name__ == "__main__":
    main()
