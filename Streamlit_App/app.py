"""
ProxErgo - Posture & Fatigue Monitoring
Main Streamlit Dashboard Application

Run with: streamlit run Streamlit_App/app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from Model_Development.detectors.pose_detector import PoseDetector
from Model_Development.detectors.face_mesh_detector import FaceMeshDetector
from Model_Development.core.biomechanics import BiomechanicsAnalyzer, PosturalSwayAnalyzer, PostureClassification
from Model_Development.core.fatigue_detector import FatigueDetector
from Model_Development.core.posture_classifier import PostureClassifierML
from Streamlit_App.components.overlay_renderer import OverlayRenderer
from dataclasses import replace

# File-based calibration request (works across WebRTC process/thread boundaries)
_CALIBRATION_REQUEST_FILE = Path(__file__).parent.parent / "data" / ".calibrate_requested"
from Streamlit_App.components.charts import create_posture_gauge, create_trend_chart, create_fatigue_chart

# Page config
st.set_page_config(
    page_title="ProxErgo - Posture & Fatigue Monitoring",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - clean, minimal UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    .stApp { background: #f8fafc; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #0f172a !important; }
    
    .metric-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-value { font-family: 'DM Sans', sans-serif; font-size: 2rem; font-weight: 600; }
    .status-optimal { color: #059669 !important; }
    .status-warning { color: #d97706 !important; }
    .status-danger { color: #dc2626 !important; }
    
    .privacy-badge {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] { background: #f1f5f9; }
    [data-testid="stSidebar"] .stMarkdown { color: #334155; }
</style>
""", unsafe_allow_html=True)


class PostureVideoProcessor(VideoProcessorBase):
    """WebRTC video processor for real-time posture analysis."""
    
    def __init__(self):
        self.pose_detector = PoseDetector(model_complexity=1)
        self.face_detector = FaceMeshDetector()
        self.biomechanics = BiomechanicsAnalyzer()
        self.sway_analyzer = PosturalSwayAnalyzer()
        self.fatigue_detector = FatigueDetector()
        self.overlay_renderer = OverlayRenderer()
        
        # Load persisted calibration (survives processor recreation)
        self.biomechanics.load_calibration()
        
        # Load ML model if available (uses biomechanics rule-based when not trained)
        model_path = Path(__file__).parent.parent / "models" / "posture_classifier.pkl"
        self.ml_classifier = PostureClassifierML(model_path=model_path)
        
        self.latest_metrics = {}
        self.cva_history = deque(maxlen=300)
        self.fatigue_history = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        
        self.is_calibrating = False
        self.calibration_frames = 0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # File-based calibration request (works across process/thread boundaries)
        if _CALIBRATION_REQUEST_FILE.exists():
            try:
                _CALIBRATION_REQUEST_FILE.unlink()
            except OSError:
                pass
            self.is_calibrating = True
        
        img = frame.to_ndarray(format="bgr24")
        timestamp = time.time() * 1000
        
        pose = self.pose_detector.detect(img, timestamp)
        
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
                    self.biomechanics.save_calibration()
                
                progress = self.calibration_frames / 30
                bio_result = self.biomechanics.analyze(
                    pose.nose, pose.left_ear, pose.right_ear,
                    pose.left_shoulder, pose.right_shoulder
                )
                img = self.overlay_renderer.render(img, pose.to_dict(), bio_result, calibration_progress=progress)
            else:
                bio_result = self.biomechanics.analyze(
                    pose.nose, pose.left_ear, pose.right_ear,
                    pose.left_shoulder, pose.right_shoulder
                )
                
                # Use ML only when NOT calibrated - calibration is personalized, ML is generic
                if self.ml_classifier.is_trained and not self.biomechanics.is_calibrated:
                    features = self.ml_classifier.extract_features(
                        pose.nose, pose.left_ear, pose.right_ear,
                        pose.left_shoulder, pose.right_shoulder
                    )
                    ml_label, _conf, _probs = self.ml_classifier.predict(features)
                    bio_result = replace(bio_result, cva_classification=PostureClassification(ml_label))
                
                self.sway_analyzer.add_sample(pose.nose[0], pose.nose[1], timestamp)
                sway = self.sway_analyzer.analyze()
                
                self.cva_history.append(bio_result.cva_angle)
                self.timestamps.append(timestamp / 1000)
                
                fatigue_metrics = None
                eyes = self.face_detector.detect(img, timestamp)
                if eyes and eyes.is_valid:
                    self.fatigue_detector.add_frame(eyes.left_eye, eyes.right_eye, timestamp)
                    fatigue = self.fatigue_detector.analyze()
                    if fatigue:
                        fatigue_metrics = fatigue.to_dict()
                        self.fatigue_history.append(fatigue.perclos)
                
                self.latest_metrics = {
                    'cva_angle': bio_result.cva_angle,
                    'cva_class': bio_result.cva_classification.value,
                    'shoulder_tilt': bio_result.shoulder_tilt_angle,
                    'confidence': bio_result.overall_confidence,
                    'sway': sway.to_dict() if sway else None,
                    'fatigue': fatigue_metrics,
                    'ml_used': self.ml_classifier.is_trained,
                    'calibrated': self.biomechanics.is_calibrated
                }
                
                img = self.overlay_renderer.render(img, pose.to_dict(), bio_result, fatigue_metrics,
                                                   is_calibrated=self.biomechanics.is_calibrated)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# ProxErgo\n*Posture & Fatigue Monitoring*")
    with col2:
        st.markdown('<div style="text-align:right;padding-top:24px;"><span class="privacy-badge">🔒 Local processing only</span></div>', 
                    unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        if 'session_start' not in st.session_state:
            st.session_state.session_start = time.time()
        
        elapsed = time.time() - st.session_state.session_start
        st.metric("Session", f"{int(elapsed//60):02d}:{int(elapsed%60):02d}")
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.session_start = time.time()
            st.rerun()
        
        st.divider()
        if st.button("📏 Calibrate Posture", use_container_width=True):
            try:
                _CALIBRATION_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
                _CALIBRATION_REQUEST_FILE.touch()
            except OSError:
                pass
            st.success("Calibration started! Sit in optimal posture when video starts.")
    
    # Main content
    col_video, col_charts = st.columns([1.2, 1])
    
    ctx = None
    with col_video:
        st.markdown("### 📹 Live Monitor")
        
        if WEBRTC_AVAILABLE:
            ctx = webrtc_streamer(
                key="proxergo",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=PostureVideoProcessor,
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                async_processing=True
            )
            
            if ctx.video_processor and ctx.video_processor.latest_metrics:
                metrics = ctx.video_processor.latest_metrics
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    cva = metrics.get('cva_angle', 0)
                    status = "status-optimal" if cva < 5 else "status-warning" if cva < 12 else "status-danger"
                    st.markdown(f'<div class="metric-card"><div class="metric-value {status}">{cva:.1f}°</div><small>CVA Angle</small></div>', 
                                unsafe_allow_html=True)
                with c2:
                    if metrics.get('fatigue'):
                        level = metrics['fatigue'].get('fatigue_level', 'unknown')
                        status = "status-optimal" if level == 'alert' else "status-warning" if level == 'mild' else "status-danger"
                        st.markdown(f'<div class="metric-card"><div class="metric-value {status}">{level.upper()}</div><small>Fatigue</small></div>', 
                                    unsafe_allow_html=True)
                with c3:
                    if metrics.get('sway'):
                        stability = 100 - metrics['sway'].get('fatigue_index', 0)
                        status = "status-optimal" if stability > 70 else "status-warning" if stability > 40 else "status-danger"
                        st.markdown(f'<div class="metric-card"><div class="metric-value {status}">{stability:.0f}%</div><small>Stability</small></div>', 
                                    unsafe_allow_html=True)
                if metrics.get('calibrated'):
                    st.caption("✅ Calibrated – using your baseline")
                elif metrics.get('ml_used'):
                    st.caption("🤖 Using trained ML model")
                else:
                    st.caption("📐 Using rule-based biomechanics")
        else:
            st.warning("Install streamlit-webrtc: `pip install streamlit-webrtc`")
    
    with col_charts:
        st.markdown("### 📈 Analytics")
        
        if ctx and ctx.video_processor:
            proc = ctx.video_processor
            metrics = proc.latest_metrics or {}
            cva_angle = metrics.get('cva_angle', 0)
            cva_class = metrics.get('cva_class', 'optimal')
            
            # CVA Gauge
            gauge_fig = create_posture_gauge(cva_angle, cva_class)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Trend chart (CVA + Fatigue over time)
            cva_hist = list(proc.cva_history) if hasattr(proc, 'cva_history') else []
            ts_hist = list(proc.timestamps) if hasattr(proc, 'timestamps') else []
            fatigue_hist = list(proc.fatigue_history) if hasattr(proc, 'fatigue_history') else []
            
            if len(cva_hist) >= 2 and len(ts_hist) >= 2:
                trend_fig = create_trend_chart(ts_hist, cva_hist, fatigue_hist if fatigue_hist else None)
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.caption("Trend chart will appear after ~10 seconds of monitoring")
            
            # Fatigue chart (when fatigue data available)
            if metrics.get('fatigue'):
                f = metrics['fatigue']
                fatigue_fig = create_fatigue_chart(
                    f.get('perclos', 0), f.get('blink_rate', 0), f.get('fatigue_level', 'alert')
                )
                st.plotly_chart(fatigue_fig, use_container_width=True)
        else:
            st.info("📊 Start the video monitor above to see live charts")
    
    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">ProxErgo • All processing happens locally</div>', 
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
