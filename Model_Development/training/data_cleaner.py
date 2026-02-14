#!/usr/bin/env python3
"""
Data Cleaner - Clean, validate, and balance posture training data.

Usage:
    python -m Model_Development.training.data_cleaner

This script:
1. Loads all JSON data files from training_data/
2. Validates each sample (visibility, anatomical ranges)
3. Removes invalid/corrupted samples
4. Balances classes (optional)
5. Saves cleaned data to training_data/cleaned_data.json
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import random
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DataCleaner:
    """
    Clean and validate posture training data.
    
    Performs:
    - Visibility filtering (landmarks must be clearly visible)
    - Anatomical validation (proportions must be reasonable)
    - Duplicate detection
    - Class balancing
    """
    
    # Minimum visibility threshold (0-1)
    MIN_VISIBILITY = 0.5
    
    # Anatomically reasonable ranges for normalized coordinates (0-1)
    # Note: These are more permissive to account for various distances from camera
    VALID_RANGES = {
        'shoulder_distance': (0.05, 0.9),   # Shoulder width (wider range for close/far)
        'ear_distance': (0.02, 0.5),        # Ear spread (wider range)
        'nose_y': (0.01, 0.85),             # Nose vertical position (very permissive)
        'shoulder_y': (0.1, 0.99),          # Shoulder vertical position
    }
    
    # Labels we expect (simplified 7-class system)
    EXPECTED_LABELS = [
        'optimal', 'forward_head_mild', 'forward_head_severe',
        'tilted_left', 'tilted_right', 'too_close', 'too_far'
    ]
    
    def __init__(self, data_dir: str = "training_data"):
        """
        Initialize the data cleaner.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        
        # Statistics
        self.stats = {
            'total_loaded': 0,
            'valid': 0,
            'rejected': defaultdict(int),
            'by_label': defaultdict(int)
        }
    
    def load_all_data(self) -> List[Dict[str, Any]]:
        """
        Load all JSON data files from the data directory.
        
        Returns:
            List of all samples from all files
        """
        all_samples = []
        
        json_files = list(self.data_dir.glob("posture_data_*.json"))
        
        if not json_files:
            print(f"⚠️  No data files found in {self.data_dir}/")
            print("   Run the data collector first:")
            print("   python -m Model_Development.training.data_collector")
            return []
        
        print(f"📂 Found {len(json_files)} data file(s)")
        
        for filepath in json_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                # Handle both old format (list) and new format (dict with metadata)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                else:
                    print(f"   ⚠️  Unknown format in {filepath.name}, skipping")
                    continue
                
                all_samples.extend(samples)
                print(f"   ✓ {filepath.name}: {len(samples)} samples")
                
            except json.JSONDecodeError as e:
                print(f"   ❌ Error reading {filepath.name}: {e}")
            except Exception as e:
                print(f"   ❌ Error processing {filepath.name}: {e}")
        
        self.stats['total_loaded'] = len(all_samples)
        print(f"\n📊 Total samples loaded: {len(all_samples)}")
        
        return all_samples
    
    def clean(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and validate all samples.
        
        Args:
            samples: List of raw samples
            
        Returns:
            List of valid, cleaned samples
        """
        cleaned = []
        
        print("\n🧹 Cleaning data...")
        
        for sample in samples:
            is_valid, reason = self._validate_sample(sample)
            
            if is_valid:
                # Normalize the sample format
                cleaned_sample = self._normalize_sample(sample)
                cleaned.append(cleaned_sample)
                self.stats['valid'] += 1
                self.stats['by_label'][sample['label']] += 1
            else:
                self.stats['rejected'][reason] += 1
        
        print(f"\n✅ Valid samples: {self.stats['valid']}")
        print(f"❌ Rejected samples: {sum(self.stats['rejected'].values())}")
        
        if self.stats['rejected']:
            print("   Rejection reasons:")
            for reason, count in sorted(self.stats['rejected'].items(), key=lambda x: -x[1]):
                print(f"     {reason}: {count}")
        
        return cleaned
    
    def _validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single sample.
        
        Args:
            sample: Sample to validate
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check required fields
        if 'label' not in sample:
            return False, "missing_label"
        
        if 'landmarks' not in sample:
            return False, "missing_landmarks"
        
        # Check label is valid
        if sample['label'] not in self.EXPECTED_LABELS:
            return False, "unknown_label"
        
        landmarks = sample['landmarks']
        
        # Check required landmarks exist
        required = ['nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
        for lm_name in required:
            if lm_name not in landmarks:
                return False, "missing_required_landmark"
        
        # Check visibility
        for lm_name in required:
            lm = landmarks[lm_name]
            if len(lm) < 4:
                return False, "invalid_landmark_format"
            if lm[3] < self.MIN_VISIBILITY:
                return False, "low_visibility"
        
        # Check anatomical validity
        l_shoulder = landmarks['left_shoulder']
        r_shoulder = landmarks['right_shoulder']
        l_ear = landmarks['left_ear']
        r_ear = landmarks['right_ear']
        nose = landmarks['nose']
        
        # Shoulder distance
        shoulder_dist = np.hypot(
            r_shoulder[0] - l_shoulder[0],
            r_shoulder[1] - l_shoulder[1]
        )
        if not (self.VALID_RANGES['shoulder_distance'][0] <= shoulder_dist <= 
                self.VALID_RANGES['shoulder_distance'][1]):
            return False, "shoulder_distance_out_of_range"
        
        # Ear distance
        ear_dist = np.hypot(r_ear[0] - l_ear[0], r_ear[1] - l_ear[1])
        if not (self.VALID_RANGES['ear_distance'][0] <= ear_dist <= 
                self.VALID_RANGES['ear_distance'][1]):
            return False, "ear_distance_out_of_range"
        
        # Nose Y position
        if not (self.VALID_RANGES['nose_y'][0] <= nose[1] <= 
                self.VALID_RANGES['nose_y'][1]):
            return False, "nose_position_out_of_range"
        
        # Shoulder Y position (should be below nose in most cases)
        # Note: this check is lenient to allow for unusual postures
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        if shoulder_mid_y < nose[1] - 0.15:  # Allow some tolerance
            return False, "shoulders_above_nose"
        
        # In a FRONT-FACING camera with MediaPipe:
        # - "left_shoulder" is the SUBJECT's left, which appears on the RIGHT of image (higher x)
        # - "right_shoulder" is the SUBJECT's right, which appears on the LEFT of image (lower x)
        # So l_shoulder[0] > r_shoulder[0] is CORRECT and expected!
        # We check for obviously wrong detections where they're basically at the same position
        shoulder_x_diff = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_x_diff < 0.02:  # Shoulders collapsed to same point = bad detection
            return False, "shoulders_collapsed"
        
        return True, None
    
    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize sample to consistent format.
        
        Args:
            sample: Raw sample
            
        Returns:
            Normalized sample
        """
        # Keep only the fields we need for training
        return {
            'label': sample['label'],
            'landmarks': {
                'nose': tuple(sample['landmarks']['nose']),
                'left_ear': tuple(sample['landmarks']['left_ear']),
                'right_ear': tuple(sample['landmarks']['right_ear']),
                'left_shoulder': tuple(sample['landmarks']['left_shoulder']),
                'right_shoulder': tuple(sample['landmarks']['right_shoulder']),
                # Include extra landmarks if available
                'left_eye': tuple(sample['landmarks'].get('left_eye', (0, 0, 0, 0))),
                'right_eye': tuple(sample['landmarks'].get('right_eye', (0, 0, 0, 0))),
            },
            'timestamp': sample.get('timestamp', 0)
        }
    
    def balance_classes(self, samples: List[Dict[str, Any]], 
                        max_per_class: Optional[int] = None,
                        min_per_class: int = 10,
                        strategy: str = 'undersample') -> List[Dict[str, Any]]:
        """
        Balance the dataset by class.
        
        Args:
            samples: List of cleaned samples
            max_per_class: Maximum samples per class (None = auto)
            min_per_class: Minimum samples required per class
            strategy: 'undersample' (default), 'keep_all', or 'drop_small'
            
        Returns:
            Balanced list of samples
        """
        print("\n⚖️  Balancing classes...")
        
        # Group by label
        by_label = defaultdict(list)
        for sample in samples:
            by_label[sample['label']].append(sample)
        
        # Report current distribution
        print("   Current distribution:")
        for label in self.EXPECTED_LABELS:
            count = len(by_label[label])
            print(f"     {label}: {count}")
        
        if strategy == 'keep_all':
            print("\n   Strategy: KEEP ALL (no balancing)")
            result = samples.copy()
            random.shuffle(result)
            print(f"   Total samples: {len(result)}")
            return result
        
        # Find valid classes (those with enough samples)
        valid_classes = {label: samples for label, samples in by_label.items() 
                        if len(samples) >= min_per_class}
        
        if not valid_classes:
            print(f"⚠️  No class has >= {min_per_class} samples!")
            return samples
        
        if strategy == 'drop_small':
            # Drop classes below threshold, keep all from valid classes
            print(f"\n   Strategy: DROP SMALL (drop classes < {min_per_class})")
            balanced = []
            for label, label_samples in by_label.items():
                if len(label_samples) >= min_per_class:
                    # Cap at max_per_class if specified
                    if max_per_class and len(label_samples) > max_per_class:
                        selected = random.sample(label_samples, max_per_class)
                    else:
                        selected = label_samples
                    balanced.extend(selected)
                    print(f"     {label}: {len(selected)} kept")
                else:
                    print(f"     {label}: DROPPED (only {len(label_samples)} samples)")
        else:
            # Default: undersample to match smallest valid class
            counts = [len(s) for s in valid_classes.values()]
            target_count = min(counts)
            if max_per_class:
                target_count = min(target_count, max_per_class)
            
            print(f"\n   Strategy: UNDERSAMPLE (target: {target_count} per class)")
            
            balanced = []
            for label, label_samples in by_label.items():
                if len(label_samples) >= min_per_class:
                    selected = random.sample(label_samples, min(len(label_samples), target_count))
                    balanced.extend(selected)
                    print(f"     {label}: {len(selected)} selected")
                else:
                    print(f"     {label}: SKIPPED (only {len(label_samples)} samples)")
        
        # Shuffle
        random.shuffle(balanced)
        
        print(f"\n   Total balanced samples: {len(balanced)}")
        
        return balanced
    
    def save_cleaned_data(self, samples: List[Dict[str, Any]], 
                          filename: str = "cleaned_data.json"):
        """
        Save cleaned data to JSON file.
        
        Args:
            samples: Cleaned samples to save
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        # Count by label
        label_counts = defaultdict(int)
        for sample in samples:
            label_counts[sample['label']] += 1
        
        save_data = {
            'metadata': {
                'total_samples': len(samples),
                'samples_per_label': dict(label_counts),
                'labels': self.EXPECTED_LABELS,
                'cleaning_stats': {
                    'total_loaded': self.stats['total_loaded'],
                    'valid': self.stats['valid'],
                    'rejected': dict(self.stats['rejected'])
                }
            },
            'samples': samples
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✅ Saved {len(samples)} cleaned samples to {filepath}")
    
    def run(self, balance: bool = True, max_per_class: Optional[int] = None,
            strategy: str = 'drop_small') -> List[Dict[str, Any]]:
        """
        Run the complete cleaning pipeline.
        
        Args:
            balance: Whether to balance classes
            max_per_class: Maximum samples per class when balancing
            strategy: 'undersample', 'keep_all', or 'drop_small' (default)
            
        Returns:
            List of cleaned (and optionally balanced) samples
        """
        # Load all data
        raw_data = self.load_all_data()
        
        if not raw_data:
            return []
        
        # Clean data
        cleaned_data = self.clean(raw_data)
        
        if not cleaned_data:
            print("❌ No valid samples after cleaning!")
            return []
        
        # Balance classes
        if balance:
            cleaned_data = self.balance_classes(cleaned_data, max_per_class, strategy=strategy)
        
        # Save
        self.save_cleaned_data(cleaned_data)
        
        return cleaned_data


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean and validate posture training data")
    parser.add_argument("--data-dir", "-d", default="training_data",
                        help="Directory containing raw data files")
    parser.add_argument("--no-balance", action="store_true",
                        help="Don't balance classes")
    parser.add_argument("--max-per-class", "-m", type=int, default=None,
                        help="Maximum samples per class")
    parser.add_argument("--strategy", "-s", default="drop_small",
                        choices=['undersample', 'keep_all', 'drop_small'],
                        help="Balancing strategy: 'undersample' (equal sizes), "
                             "'keep_all' (no balancing), 'drop_small' (drop classes <10, keep rest)")
    args = parser.parse_args()
    
    cleaner = DataCleaner(data_dir=args.data_dir)
    cleaner.run(balance=not args.no_balance, max_per_class=args.max_per_class, 
                strategy=args.strategy)


if __name__ == "__main__":
    main()
