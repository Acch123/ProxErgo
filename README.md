# ProxErgo

## Posture & Fatigue Monitoring

<p align="center">
  <strong>A Privacy-First Posture & Fatigue Monitoring System</strong><br>
  <em>Ergonomic assessment on any laptop with a front-facing camera</em>
</p>

---

## Overview

**ProxErgo** monitors posture and fatigue using your front-facing webcam. Built on clinically-validated biomechanics and privacy-by-design: only landmark coordinates are processed, never raw video.

### Key Features

| Traditional Approach | ProxErgo |
|---------------------|----------|
| Requires side-view camera setup | Works with any front-facing webcam |
| Expensive clinical equipment | Runs on any laptop |
| Data sent to cloud servers | 100% local processing |
| Generic "slouch detection" | CVA proxy + depth + vertical metrics |
| Single metric | Multi-factor posture + fatigue + sway |

---

## Project Structure

```
ProxErgo/
├── Clinical_Research/          # Evidence-based thresholds
│   ├── clinical_thresholds.py
│   └── references.md
│
├── Model_Development/          # Core analysis
│   ├── core/
│   │   ├── biomechanics.py      # CVA, depth, vertical, tilt, sway
│   │   ├── fatigue_detector.py # PERCLOS & blink detection
│   │   └── posture_classifier.py  # XGBoost + rule-based fallback
│   ├── detectors/              # MediaPipe pose & face mesh
│   └── training/               # Data collection & model training
│
├── Streamlit_App/              # Web dashboard
│   ├── app.py
│   └── components/
│
├── config.yaml                 # Thresholds & settings
├── demo_opencv.py              # Standalone OpenCV demo
├── environment.yml
└── requirements.txt
```

---

## Quick Start

### 1. Set Up Environment

```bash
conda env create -f environment.yml
conda activate proxergo
# or: pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run Streamlit_App/app.py
```

### 3. Or Run the Standalone Demo

```bash
python demo_opencv.py
```

**Demo controls:** `q` Quit · `c` Calibrate · `s` Skeleton · `m` Metrics · `r` Reset

---

## Calibration

Calibration personalizes the baseline for your posture and setup:

1. Press **`c`** (demo) or click **Calibrate Posture** (app)
2. Sit in your best upright posture
3. Hold still for ~30 frames (~1 second)

Calibration captures your neutral depth, vertical ear–shoulder distance, and shoulder tilt. When calibrated, the system uses your baseline instead of generic thresholds. Recalibrate when you change clothing or camera position.

---

## Posture Classification

7-class system: **optimal**, **forward_head_mild**, **forward_head_severe**, **tilted_left**, **tilted_right**, **too_close**, **too_far**.

A forward-head score combines depth, vertical ear–shoulder distance, CVA, and slouch. When tilted, vertical/CVA are excluded (they invert).

| Score | Classification |
|-------|----------------|
| < 0.25 | Optimal |
| 0.25 – 0.65 | Mild forward |
| ≥ 0.65 | Severe forward |

Tilt overrides forward head when shoulder tilt deviation exceeds 7° (calibrated) or 10° (uncalibrated).

---

## Key Metrics

### Posture
- **CVA proxy:** arctan2(|ear_x − shoulder_x|, shoulder_y − ear_y)
- **Depth:** nose vs shoulder midpoint (positive = forward)
- **Vertical:** shoulder_mid_y − ear_mid_y (head drop)
- **Slouch:** max(0, expected_height − nose_y) / shoulder_distance

### Fatigue (PERCLOS)
| Level | PERCLOS | Action |
|-------|---------|--------|
| Alert | < 15% | Normal |
| Mild | 15–25% | Monitor |
| Moderate | 25–40% | Take a break |
| Severe | > 40% | Rest needed |

### Postural Sway
Tracks nose position over time. Amplitude, path length, velocity, and frequency indicate neuromuscular fatigue.

---

## Privacy by Design

- Raw video frames are **never** saved
- Only skeleton coordinates are processed
- All inference runs locally
- No network transmission of posture/fatigue data
- Calibration stored in `data/calibration.json` (local)

---

## Report

See `Clinical_Research/Fatigue Tracker ProxErgo Report.pdf` for the report.

---

<p align="center">
  <strong>ProxErgo</strong> · An expert exists locally on your device.
</p>
