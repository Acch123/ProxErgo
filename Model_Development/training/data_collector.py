#!/usr/bin/env python3
"""
Data Collector - Interactive tool to collect labeled posture data for ML training.

Usage:
    python -m Model_Development.training.data_collector

Controls:
    1-7    : Set current label (see list below)
    SPACE  : Record current frame with the set label
    S      : Save all collected data to JSON
    R      : Reset (clear all collected data)
    Q      : Quit

Labels (Simplified 7-class system):
    1: optimal              - Perfect upright posture
    2: forward_head_mild    - Slight forward head / mild slouching
    3: forward_head_severe  - Significant forward head / severe slouching
    4: tilted_left          - Head/shoulders tilted left
    5: tilted_right         - Head/shoulders tilted right
    6: too_close            - Too close to camera
    7: too_far              - Too far from camera
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Model_Development.detectors.pose_detector import PoseDetector


class DataCollector:
    """
    Interactive tool to collect labeled posture data.
    
    This tool opens your webcam and lets you:
    1. Assume different postures
    2. Label them with key presses (1-7)
    3. Record landmark data with SPACE
    4. Save to JSON for training
    """
    
    LABELS = [
        'optimal',               # 1 - Perfect upright posture
        'forward_head_mild',     # 2 - Slight forward head / mild slouching
        'forward_head_severe',   # 3 - Significant forward head / severe slouching
        'tilted_left',           # 4 - Head/shoulders tilted left
        'tilted_right',          # 5 - Head/shoulders tilted right
        'too_close',             # 6 - Too close to camera
        'too_far'                # 7 - Too far from camera
    ]
    
    # Colors for UI
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (128, 128, 128)
    
    def __init__(self, output_dir: str = "training_data", camera_id: int = 0):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            camera_id: Camera device ID
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.camera_id = camera_id
        self.pose_detector = PoseDetector(model_complexity=1)
        
        # Collection state
        self.data: List[Dict[str, Any]] = []
        self.current_label: Optional[str] = None
        self.current_label_idx: Optional[int] = None
        
        # Stats
        self.samples_per_label: Dict[str, int] = {label: 0 for label in self.LABELS}
        
    def collect(self):
        """Run the interactive data collection interface."""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        self._print_instructions()
        
        cv2.namedWindow("ProxErgo Data Collector", cv2.WINDOW_NORMAL)
        
        last_record_time = 0
        record_cooldown = 0.2  # 200ms between records to prevent duplicates
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                timestamp = time.time() * 1000
                
                # Detect pose
                pose = self.pose_detector.detect(frame, timestamp)
                
                # Create display frame
                display = self._draw_ui(frame, pose, timestamp)
                
                cv2.imshow("ProxErgo Data Collector", display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Number keys 1-9 set label
                if ord('1') <= key <= ord('7'):
                    idx = key - ord('1')
                    if idx < len(self.LABELS):
                        self.current_label = self.LABELS[idx]
                        self.current_label_idx = idx
                        print(f"📌 Label set to: [{idx+1}] {self.current_label}")
                
                # SPACE records data
                elif key == ord(' '):
                    current_time = time.time()
                    if current_time - last_record_time > record_cooldown:
                        self._try_record(pose, timestamp)
                        last_record_time = current_time
                
                # S saves data
                elif key == ord('s') or key == ord('S'):
                    self._save_data()
                
                # R resets data
                elif key == ord('r') or key == ord('R'):
                    self._reset_data()
                
                # Q quits
                elif key == ord('q') or key == ord('Q'):
                    if self.data:
                        print("\n⚠️  You have unsaved data!")
                        print("Press 'S' to save, or 'Q' again to quit without saving")
                        confirm_key = cv2.waitKey(0) & 0xFF
                        if confirm_key == ord('s') or confirm_key == ord('S'):
                            self._save_data()
                        elif confirm_key != ord('q') and confirm_key != ord('Q'):
                            continue
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
            print("\n👋 Data collector closed")
    
    def _draw_ui(self, frame: np.ndarray, pose, timestamp: float) -> np.ndarray:
        """Draw the collection UI overlay."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Semi-transparent overlay for text areas
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        # Title
        cv2.putText(display, "FROSTBYTE DATA COLLECTOR", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_CYAN, 2)
        
        # Current label
        if self.current_label:
            label_text = f"Label: [{self.current_label_idx+1}] {self.current_label}"
            cv2.putText(display, label_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GREEN, 2)
        else:
            cv2.putText(display, "Label: NOT SET (press 1-7)", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_RED, 2)
        
        # Sample counts
        cv2.putText(display, f"Total samples: {len(self.data)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        # Pose detection status and skeleton
        if pose and pose.is_valid:
            # Draw skeleton
            self._draw_skeleton(display, pose, w, h)
            
            # Status indicator
            cv2.circle(display, (w-30, 30), 15, self.COLOR_GREEN, -1)
            cv2.putText(display, "POSE OK", (w-120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GREEN, 1)
            
            # Visibility score
            vis = pose.average_visibility
            cv2.putText(display, f"Visibility: {vis:.0%}", (w-150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        else:
            cv2.circle(display, (w-30, 30), 15, self.COLOR_RED, -1)
            cv2.putText(display, "NO POSE", (w-120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_RED, 1)
        
        # Bottom panel - instructions and stats
        y_start = h - 140
        
        cv2.putText(display, "Controls: SPACE=Record | S=Save | R=Reset | Q=Quit",
                    (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_YELLOW, 1)
        
        # Label counts (2 columns)
        col1_x, col2_x = 10, w//2 + 10
        for i, label in enumerate(self.LABELS):
            count = self.samples_per_label[label]
            
            # Highlight current label
            if label == self.current_label:
                color = self.COLOR_GREEN
                prefix = "▶ "
            else:
                color = self.COLOR_GRAY
                prefix = "  "
            
            text = f"{prefix}[{i+1}] {label}: {count}"
            
            if i < 5:
                y = y_start + 25 + (i * 20)
                cv2.putText(display, text, (col1_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                y = y_start + 25 + ((i-5) * 20)
                cv2.putText(display, text, (col2_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return display
    
    def _draw_skeleton(self, frame: np.ndarray, pose, w: int, h: int):
        """Draw pose skeleton on frame."""
        def to_px(pt):
            return (int(pt[0] * w), int(pt[1] * h))
        
        # Get key points
        nose = to_px(pose.nose)
        l_shoulder = to_px(pose.left_shoulder)
        r_shoulder = to_px(pose.right_shoulder)
        l_ear = to_px(pose.left_ear)
        r_ear = to_px(pose.right_ear)
        l_eye = to_px(pose.left_eye)
        r_eye = to_px(pose.right_eye)
        
        # Draw connections
        cv2.line(frame, l_shoulder, r_shoulder, self.COLOR_CYAN, 2)
        cv2.line(frame, l_ear, r_ear, self.COLOR_CYAN, 1)
        cv2.line(frame, l_eye, r_eye, self.COLOR_CYAN, 1)
        
        # Shoulder midpoint to ear midpoint (posture line)
        shoulder_mid = ((l_shoulder[0] + r_shoulder[0])//2, (l_shoulder[1] + r_shoulder[1])//2)
        ear_mid = ((l_ear[0] + r_ear[0])//2, (l_ear[1] + r_ear[1])//2)
        cv2.line(frame, shoulder_mid, ear_mid, self.COLOR_YELLOW, 2)
        
        # Draw points
        for pt in [nose, l_shoulder, r_shoulder, l_ear, r_ear]:
            cv2.circle(frame, pt, 5, self.COLOR_GREEN, -1)
            cv2.circle(frame, pt, 7, self.COLOR_CYAN, 1)
    
    def _try_record(self, pose, timestamp: float):
        """Attempt to record the current frame."""
        if not self.current_label:
            print("⚠️  Set a label first! Press 1-7 to select a posture type.")
            return
        
        if not pose or not pose.is_valid:
            print("⚠️  No valid pose detected. Make sure you're visible to the camera.")
            return
        
        if pose.average_visibility < 0.5:
            print(f"⚠️  Low visibility ({pose.average_visibility:.0%}). Adjust your position.")
            return
        
        # Record the sample
        sample = {
            "timestamp": timestamp,
            "recorded_at": datetime.now().isoformat(),
            "label": self.current_label,
            "landmarks": {
                "nose": pose.nose,
                "left_eye": pose.left_eye,
                "right_eye": pose.right_eye,
                "left_eye_inner": pose.left_eye_inner,
                "left_eye_outer": pose.left_eye_outer,
                "right_eye_inner": pose.right_eye_inner,
                "right_eye_outer": pose.right_eye_outer,
                "left_ear": pose.left_ear,
                "right_ear": pose.right_ear,
                "left_shoulder": pose.left_shoulder,
                "right_shoulder": pose.right_shoulder,
                "mouth_left": pose.mouth_left,
                "mouth_right": pose.mouth_right
            },
            "visibility": pose.average_visibility
        }
        
        self.data.append(sample)
        self.samples_per_label[self.current_label] += 1
        
        print(f"✅ Recorded #{len(self.data)}: {self.current_label} "
              f"(total for this label: {self.samples_per_label[self.current_label]})")
    
    def _save_data(self):
        """Save collected data to JSON file."""
        if not self.data:
            print("⚠️  No data to save!")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"posture_data_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Save with metadata
        save_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(self.data),
                "samples_per_label": self.samples_per_label,
                "labels": self.LABELS
            },
            "samples": self.data
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✅ Saved {len(self.data)} samples to {filepath}")
        print("   Breakdown:")
        for label, count in self.samples_per_label.items():
            if count > 0:
                print(f"     {label}: {count}")
    
    def _reset_data(self):
        """Clear all collected data."""
        if not self.data:
            print("ℹ️  No data to reset")
            return
        
        count = len(self.data)
        self.data = []
        self.samples_per_label = {label: 0 for label in self.LABELS}
        print(f"🔄 Reset! Cleared {count} samples.")
    
    def _print_instructions(self):
        """Print usage instructions to console."""
        print("\n" + "="*60)
        print("  🧊 FROSTBYTE DATA COLLECTOR")
        print("="*60)
        print("\n📖 Instructions:")
        print("   1. Assume a posture (good, slouching, tilted, etc.)")
        print("   2. Press 1-7 to select the label for that posture")
        print("   3. Press SPACE to record the current frame")
        print("   4. Repeat for different postures")
        print("   5. Press S to save all data to JSON")
        print("\n🏷️  Labels:")
        for i, label in enumerate(self.LABELS, 1):
            print(f"   [{i}] {label}")
        print("\n⌨️  Controls:")
        print("   SPACE  = Record current frame")
        print("   S      = Save data to file")
        print("   R      = Reset (clear all data)")
        print("   Q      = Quit")
        print("\n💡 Tips:")
        print("   • Collect at least 50 samples per label for good results")
        print("   • Vary your position slightly within each posture type")
        print("   • Make sure lighting is consistent")
        print("   • Green skeleton = good pose detection")
        print("="*60 + "\n")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect labeled posture data")
    parser.add_argument("--output", "-o", default="training_data",
                        help="Output directory for data files")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera device ID")
    args = parser.parse_args()
    
    collector = DataCollector(output_dir=args.output, camera_id=args.camera)
    collector.collect()


if __name__ == "__main__":
    main()
