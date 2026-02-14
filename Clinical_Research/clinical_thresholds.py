"""
Clinical Thresholds & Reference Values
Based on peer-reviewed medical literature for posture and fatigue assessment.

This module defines evidence-based thresholds used throughout ProxErgo.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CVAThresholds:
    """
    Craniovertebral Angle (CVA) Thresholds
    
    True CVA is measured as the angle between a horizontal line through C7
    and a line from C7 to the tragus of the ear. Normal range: 48-50°.
    
    Our front-view proxy measures deviation from vertical alignment.
    Thresholds are adapted from:
    - Yip et al. (2008): CVA correlation with forward head posture
    - Silva et al. (2017): Photogrammetric posture assessment
    - Nejati et al. (2015): CVA and neck pain correlation
    
    Classification (Front-View Proxy):
    """
    OPTIMAL: float = 5.0       # < 5° deviation = excellent posture
    MILD: float = 12.0         # 5-12° = mild forward head posture
    MODERATE: float = 20.0     # 12-20° = moderate, intervention recommended
    SEVERE: float = 30.0       # > 20° = severe, clinical attention needed
    
    # Clinical notes
    CLINICAL_NOTES: str = """
    Forward Head Posture (FHP) is associated with:
    - Increased cervical spine loading (every inch forward = ~10lbs additional load)
    - Neck pain and tension headaches
    - Reduced respiratory capacity
    - Temporomandibular joint (TMJ) dysfunction
    
    Intervention recommended when CVA proxy > 12° for sustained periods.
    """


@dataclass(frozen=True)
class PERCLOSThresholds:
    """
    PERCLOS (Percentage of Eye Closure) Thresholds
    
    PERCLOS is a validated drowsiness measure used in transportation safety.
    Measures the percentage of time eyelids cover >80% of pupil.
    
    Based on:
    - Wierwille et al. (1994): Original PERCLOS validation
    - Dinges et al. (1998): PERCLOS as drowsiness predictor
    - Caffier et al. (2003): Blink patterns and fatigue
    """
    ALERT: float = 8.0         # < 8% = fully alert
    MILD_FATIGUE: float = 15.0 # 8-15% = mild drowsiness
    MODERATE_FATIGUE: float = 25.0  # 15-25% = significant drowsiness
    SEVERE_FATIGUE: float = 40.0    # > 40% = critical drowsiness
    
    # Eye Aspect Ratio threshold
    EAR_CLOSED_THRESHOLD: float = 0.2  # EAR < 0.2 = eyes closed
    
    CLINICAL_NOTES: str = """
    PERCLOS > 40% correlates with:
    - 4x increased accident risk in driving studies
    - Significant cognitive impairment
    - Reaction time delays > 500ms
    
    Blink rate norms (adults):
    - Relaxed: 15-20 blinks/minute
    - Reading/screen work: 3-4 blinks/minute (reduced)
    - Fatigue: Variable, often < 10 or > 25/min
    """


@dataclass(frozen=True)
class BlinkThresholds:
    """
    Blink Rate & Duration Thresholds
    
    Based on:
    - Stern et al. (1994): Blink rate and cognitive load
    - Caffier et al. (2003): Blink duration and drowsiness
    - Schleicher et al. (2008): Blink patterns in fatigue
    """
    # Normal blink rate range (blinks per minute)
    NORMAL_RATE_LOW: float = 12.0
    NORMAL_RATE_HIGH: float = 20.0
    
    # Abnormal ranges
    VERY_LOW_RATE: float = 8.0    # Possible concentration or dry eyes
    HIGH_RATE: float = 25.0       # Possible stress or fatigue
    VERY_HIGH_RATE: float = 30.0  # Significant fatigue indicator
    
    # Blink duration (milliseconds)
    NORMAL_DURATION_MIN: float = 100.0
    NORMAL_DURATION_MAX: float = 200.0
    PROLONGED_BLINK: float = 300.0  # Fatigue indicator
    MICROSLEEP_THRESHOLD: float = 500.0  # Potential microsleep


@dataclass(frozen=True)
class PosturalSwayThresholds:
    """
    Postural Sway Thresholds for Fatigue Detection
    
    Based on:
    - Paillard (2012): Postural control and fatigue
    - Donath et al. (2015): Balance and neuromuscular fatigue
    - Vuillerme et al. (2002): Sway frequency and stability
    """
    # Sway amplitude thresholds (normalized 0-1 scale)
    STABLE_AMPLITUDE: float = 0.01     # < 1% sway = excellent stability
    NORMAL_AMPLITUDE: float = 0.025    # 1-2.5% = normal
    ELEVATED_AMPLITUDE: float = 0.04   # 2.5-4% = fatigue indication
    HIGH_AMPLITUDE: float = 0.06       # > 6% = significant instability
    
    # Sway frequency (Hz) - lower frequency indicates fatigue
    NORMAL_FREQUENCY_LOW: float = 0.3
    NORMAL_FREQUENCY_HIGH: float = 2.0
    FATIGUE_FREQUENCY: float = 0.2  # < 0.2 Hz suggests fatigue
    
    CLINICAL_NOTES: str = """
    Postural sway increases with:
    - Neuromuscular fatigue
    - Sleep deprivation
    - Cognitive load
    - Visual fatigue
    
    Sway pattern changes:
    - Increased amplitude = reduced motor control
    - Decreased frequency = slowed postural corrections
    - Increased velocity variability = compensation effort
    """


@dataclass(frozen=True)
class ShoulderTiltThresholds:
    """
    Shoulder Alignment Thresholds
    
    Based on:
    - Kendall et al. (2005): Muscle Testing and Function
    - Sahrmann (2002): Movement Impairment Syndromes
    """
    OPTIMAL: float = 2.0      # < 2° = excellent alignment
    ACCEPTABLE: float = 5.0   # 2-5° = acceptable variation
    ELEVATED: float = 8.0     # 5-8° = noticeable tilt
    SIGNIFICANT: float = 12.0 # > 12° = significant asymmetry
    
    CLINICAL_NOTES: str = """
    Lateral shoulder tilt may indicate:
    - Scoliosis or spinal curvature
    - Habitual posture patterns
    - Workstation ergonomic issues
    - Muscle imbalance (upper trapezius, levator scapulae)
    
    Sustained tilt > 5° may benefit from ergonomic intervention.
    """


# Aggregate all thresholds
CLINICAL_THRESHOLDS = {
    'cva': CVAThresholds(),
    'perclos': PERCLOSThresholds(),
    'blink': BlinkThresholds(),
    'sway': PosturalSwayThresholds(),
    'shoulder': ShoulderTiltThresholds()
}


def get_posture_risk_level(cva_angle: float) -> Tuple[str, str, str]:
    """
    Get risk level and recommendations based on CVA angle.
    
    Args:
        cva_angle: Measured CVA angle in degrees
        
    Returns:
        Tuple of (risk_level, description, recommendation)
    """
    thresholds = CLINICAL_THRESHOLDS['cva']
    
    if cva_angle < thresholds.OPTIMAL:
        return (
            "LOW",
            "Excellent posture alignment",
            "Maintain current posture habits"
        )
    elif cva_angle < thresholds.MILD:
        return (
            "MILD",
            "Slight forward head position detected",
            "Consider chin tuck exercises during breaks"
        )
    elif cva_angle < thresholds.MODERATE:
        return (
            "MODERATE", 
            "Moderate forward head posture",
            "Adjust monitor height, take regular posture breaks"
        )
    else:
        return (
            "HIGH",
            "Significant forward head posture",
            "Consider ergonomic assessment and corrective exercises"
        )


def get_fatigue_risk_level(perclos: float, blink_rate: float) -> Tuple[str, str, str]:
    """
    Get fatigue risk level based on PERCLOS and blink rate.
    
    Args:
        perclos: PERCLOS percentage (0-100)
        blink_rate: Blinks per minute
        
    Returns:
        Tuple of (risk_level, description, recommendation)
    """
    thresholds = CLINICAL_THRESHOLDS['perclos']
    blink_thresh = CLINICAL_THRESHOLDS['blink']
    
    # Combined risk assessment
    perclos_risk = 0
    if perclos >= thresholds.SEVERE_FATIGUE:
        perclos_risk = 3
    elif perclos >= thresholds.MODERATE_FATIGUE:
        perclos_risk = 2
    elif perclos >= thresholds.MILD_FATIGUE:
        perclos_risk = 1
    
    blink_risk = 0
    if blink_rate < blink_thresh.VERY_LOW_RATE or blink_rate > blink_thresh.VERY_HIGH_RATE:
        blink_risk = 2
    elif blink_rate < blink_thresh.NORMAL_RATE_LOW or blink_rate > blink_thresh.HIGH_RATE:
        blink_risk = 1
    
    total_risk = perclos_risk + blink_risk
    
    if total_risk == 0:
        return ("LOW", "Alert and focused", "Continue working")
    elif total_risk <= 2:
        return ("MILD", "Early signs of fatigue", "Consider a short break soon")
    elif total_risk <= 4:
        return ("MODERATE", "Moderate fatigue detected", "Take a 5-10 minute break")
    else:
        return ("HIGH", "Significant fatigue", "Take an immediate break, consider rest")
