#!/usr/bin/env python3
"""
ProxErgo - Standalone OpenCV Demo
Run this to test posture detection without Streamlit.

Usage: python demo_opencv.py

Controls:
    q - Quit
    c - Start calibration
    s - Toggle skeleton
    m - Toggle metrics panel
    r - Reset session
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from dataclasses import replace

from Model_Development.detectors.pose_detector import PoseDetector
from Model_Development.detectors.face_mesh_detector import FaceMeshDetector
from Model_Development.core.biomechanics import BiomechanicsAnalyzer, PosturalSwayAnalyzer, PostureClassification
from Model_Development.core.fatigue_detector import FatigueDetector
from Model_Development.core.posture_classifier import PostureClassifierML
from Streamlit_App.components.overlay_renderer import OverlayRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="ProxErgo OpenCV Demo")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--width", "-w", type=int, default=640, help="Width")
    parser.add_argument("--height", "-H", type=int, default=480, help="Height")
    parser.add_argument("--model", type=int, choices=[0, 1, 2], default=1, help="Model complexity")
    return parser.parse_args()


class ProxErgoDemo:
    def __init__(self, camera_id=0, width=640, height=480, model_complexity=1):
        print("ProxErgo - Initializing...")
        
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("   Loading detectors...")
        self.pose_detector = PoseDetector(model_complexity=model_complexity)
        self.face_detector = FaceMeshDetector()
        
        print("   Initializing analyzers...")
        self.biomechanics = BiomechanicsAnalyzer()
        self.sway_analyzer = PosturalSwayAnalyzer()
        self.fatigue_detector = FatigueDetector()
        self.overlay = OverlayRenderer()
        
        model_path = Path(__file__).parent / "models" / "posture_classifier.pkl"
        self.ml_classifier = PostureClassifierML(model_path=model_path)
        if self.ml_classifier.is_trained:
            print("   ✓ ML model loaded")
        else:
            print("   ✓ Using rule-based biomechanics")
        
        self.is_calibrating = False
        self.calibration_frames = 0
        self.fps_history = []
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        print("✅ Ready! Press 'q' to quit, 'c' to calibrate")
        
    def run(self):
        cv2.namedWindow("ProxErgo - Posture & Fatigue", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = time.time() * 1000
            output = self.process_frame(frame, timestamp)
            
            # FPS
            self.frame_count += 1
            if time.time() - self.last_fps_time >= 1.0:
                fps = self.frame_count / (time.time() - self.last_fps_time)
                self.fps_history.append(fps)
                self.frame_count = 0
                self.last_fps_time = time.time()
            
            if self.fps_history:
                cv2.putText(output, f"FPS: {self.fps_history[-1]:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("ProxErgo - Posture & Fatigue", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_calibration()
            elif key == ord('s'):
                self.overlay.show_skeleton = not self.overlay.show_skeleton
            elif key == ord('m'):
                self.overlay.show_metrics = not self.overlay.show_metrics
            elif key == ord('r'):
                self.reset()
        
        self.cleanup()
    
    def process_frame(self, frame, timestamp):
        output = frame.copy()
        pose = self.pose_detector.detect(frame, timestamp)
        
        if pose and pose.is_valid:
            if self.is_calibrating:
                # Use full landmarks to build a personalized baseline
                self.biomechanics.calibrate(
                    pose.nose,
                    pose.left_ear, pose.right_ear,
                    pose.left_shoulder, pose.right_shoulder
                )
                self.calibration_frames += 1
                
                if self.calibration_frames >= 30:
                    self.is_calibrating = False
                    self.calibration_frames = 0
                    print("✅ Calibration complete!")
                
                progress = self.calibration_frames / 30
                bio = self.biomechanics.analyze(pose.nose, pose.left_ear, pose.right_ear,
                                                pose.left_shoulder, pose.right_shoulder)
                output = self.overlay.render(output, pose.to_dict(), bio, calibration_progress=progress)
            else:
                bio = self.biomechanics.analyze(pose.nose, pose.left_ear, pose.right_ear,
                                                pose.left_shoulder, pose.right_shoulder)
                
                # Use ML only when NOT calibrated - calibration is personalized, ML is generic
                if self.ml_classifier.is_trained and not self.biomechanics.is_calibrated:
                    features = self.ml_classifier.extract_features(
                        pose.nose, pose.left_ear, pose.right_ear,
                        pose.left_shoulder, pose.right_shoulder
                    )
                    ml_label, _conf, _probs = self.ml_classifier.predict(features)
                    bio = replace(bio, cva_classification=PostureClassification(ml_label))
                
                self.sway_analyzer.add_sample(pose.nose[0], pose.nose[1], timestamp)
                
                fatigue_metrics = None
                eyes = self.face_detector.detect(frame, timestamp)
                if eyes and eyes.is_valid:
                    self.fatigue_detector.add_frame(eyes.left_eye, eyes.right_eye, timestamp)
                    fatigue = self.fatigue_detector.analyze()
                    if fatigue:
                        fatigue_metrics = fatigue.to_dict()
                
                output = self.overlay.render(output, pose.to_dict(), bio, fatigue_metrics,
                                            is_calibrated=self.biomechanics.is_calibrated)
        else:
            cv2.putText(output, "No pose detected - face the camera", (50, output.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output
    
    def start_calibration(self):
        print("📏 Calibrating - sit in optimal posture...")
        self.is_calibrating = True
        self.calibration_frames = 0
    
    def reset(self):
        print("🔄 Resetting...")
        self.sway_analyzer.reset()
        self.fatigue_detector.reset()
    
    def cleanup(self):
        self.cap.release()
        self.pose_detector.close()
        self.face_detector.close()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")


def main():
    args = parse_args()
    print("\n" + "=" * 50)
    print("  ProxErgo - Posture & Fatigue Monitoring")
    print("  Local processing • Front-facing camera")
    print("=" * 50 + "\n")
    
    demo = ProxErgoDemo(args.camera, args.width, args.height, args.model)
    demo.run()


if __name__ == "__main__":
    main()
