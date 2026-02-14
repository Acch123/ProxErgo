#!/usr/bin/env python3
"""
Model Evaluator - Detailed performance analysis for each class.

Usage:
    python -m Model_Development.training.evaluate_model
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib

from Model_Development.core.posture_classifier import PostureClassifierML, PostureFeatures


def load_data_and_model(data_path: str = "training_data/cleaned_data.json",
                       model_path: str = "models/posture_classifier.pkl"):
    """Load training data and trained model."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return None, None, None
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    # Load data
    with open(data_path) as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    
    # Load model
    classifier = PostureClassifierML(model_path=model_path)
    
    # Extract features
    features = []
    labels = []
    
    for sample in samples:
        try:
            lm = sample['landmarks']
            feat = classifier.extract_features(
                nose=tuple(lm['nose']),
                left_ear=tuple(lm['left_ear']),
                right_ear=tuple(lm['right_ear']),
                left_shoulder=tuple(lm['left_shoulder']),
                right_shoulder=tuple(lm['right_shoulder'])
            )
            features.append(feat)
            labels.append(sample['label'])
        except Exception as e:
            continue
    
    return features, labels, classifier


def evaluate_per_class_performance(features, labels, classifier, test_size=0.2):
    """Evaluate model and return detailed per-class metrics."""
    # Convert to arrays
    X = np.array([f.to_array() for f in features])
    
    # Map labels to indices
    label_to_idx = {label: i for i, label in enumerate(classifier.LABELS)}
    y = np.array([label_to_idx.get(label, -1) for label in labels])
    
    # Remove unknown labels
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    labels = [l for l, v in zip(labels, valid_mask) if v]
    
    # Split data (same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Predict on test set
    y_pred = classifier.pipeline.predict(X_test)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classifier.LABELS)), zero_division=0
    )
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(classifier.LABELS):
        per_class_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classifier.LABELS)))
    
    return {
        'overall_accuracy': float(accuracy),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'labels': classifier.LABELS,
        'test_samples': len(y_test),
        'train_samples': len(y_train)
    }


def print_detailed_report(results: Dict[str, Any]):
    """Print a detailed, formatted performance report."""
    print("\n" + "="*70)
    print("  📊 DETAILED MODEL PERFORMANCE REPORT")
    print("="*70)
    
    print(f"\n📈 Overall Performance:")
    print(f"   Test Accuracy: {results['overall_accuracy']:.1%}")
    print(f"   Test Samples:  {results['test_samples']}")
    print(f"   Train Samples: {results['train_samples']}")
    
    print(f"\n{'='*70}")
    print("  📋 Per-Class Performance Metrics")
    print(f"{'='*70}\n")
    
    # Table header
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10} {'Status'}")
    print("-" * 70)
    
    # Sort by F1-score (descending)
    sorted_classes = sorted(
        results['per_class'].items(),
        key=lambda x: x[1]['f1_score'],
        reverse=True
    )
    
    for label, metrics in sorted_classes:
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        support = metrics['support']
        
        # Status indicator
        if f1 >= 0.9:
            status = "✅ Excellent"
        elif f1 >= 0.75:
            status = "✅ Good"
        elif f1 >= 0.6:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"
        
        # Format numbers
        print(f"{label:<25} {precision:>10.1%}  {recall:>10.1%}  {f1:>10.1%}  {support:>8}  {status}")
    
    print("\n" + "="*70)
    print("  🔍 Confusion Matrix")
    print("="*70)
    
    cm = np.array(results['confusion_matrix'])
    labels = results['labels']
    
    # Print header
    print(f"\n{'Actual ↓':<20} ", end="")
    for label in labels:
        print(f"{label[:10]:>12}", end="")
    print()
    print("-" * (20 + 12 * len(labels)))
    
    # Print rows
    for i, label in enumerate(labels):
        print(f"{label:<20} ", end="")
        for j in range(len(labels)):
            val = cm[i, j]
            # Highlight correct predictions (diagonal)
            if i == j and val > 0:
                print(f"{val:>12}", end="")
            elif val > 0:
                print(f"{val:>12}", end="")
            else:
                print(f"{'·':>12}", end="")
        print()
    
    # Summary statistics
    print("\n" + "="*70)
    print("  📊 Summary Statistics")
    print("="*70)
    
    f1_scores = [m['f1_score'] for m in results['per_class'].values()]
    precisions = [m['precision'] for m in results['per_class'].values()]
    recalls = [m['recall'] for m in results['per_class'].values()]
    
    print(f"\n   Average F1-Score:    {np.mean(f1_scores):.1%}")
    print(f"   Average Precision:   {np.mean(precisions):.1%}")
    print(f"   Average Recall:      {np.mean(recalls):.1%}")
    print(f"   Min F1-Score:        {np.min(f1_scores):.1%}")
    print(f"   Max F1-Score:        {np.max(f1_scores):.1%}")
    
    # Classes needing improvement
    poor_classes = [label for label, m in results['per_class'].items() 
                   if m['f1_score'] < 0.7]
    if poor_classes:
        print(f"\n   ⚠️  Classes needing improvement:")
        for label in poor_classes:
            f1 = results['per_class'][label]['f1_score']
            print(f"      • {label}: F1={f1:.1%}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model performance")
    parser.add_argument("--data", "-d", default="training_data/cleaned_data.json",
                        help="Path to cleaned training data")
    parser.add_argument("--model", "-m", default="models/posture_classifier.pkl",
                        help="Path to trained model")
    args = parser.parse_args()
    
    print("\n🔍 Loading model and data...")
    features, labels, classifier = load_data_and_model(args.data, args.model)
    
    if features is None:
        return
    
    print(f"   Loaded {len(features)} samples")
    print(f"   Model has {len(classifier.LABELS)} classes: {', '.join(classifier.LABELS)}")
    
    print("\n📊 Evaluating model...")
    results = evaluate_per_class_performance(features, labels, classifier)
    
    print_detailed_report(results)
    
    # Save results
    output_path = Path(args.model).parent / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
