"""Kalman Filter for landmark smoothing."""

import numpy as np


class KalmanFilter1D:
    """Simple 1D Kalman filter for noise reduction."""
    
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-2, initial_value: float = 0.0):
        self.Q, self.R = process_variance, measurement_variance
        self.x, self.P, self.K = initial_value, 1.0, 0.0
        
    def update(self, measurement: float) -> float:
        self.P = self.P + self.Q
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * (measurement - self.x)
        self.P = (1 - self.K) * self.P
        return self.x
    
    def filter(self, measurement: float) -> float:
        return self.update(measurement)
    
    @property
    def value(self) -> float:
        return self.x
    
    def reset(self, value: float = 0.0):
        self.x, self.P, self.K = value, 1.0, 0.0


class KalmanFilter2D:
    """2D Kalman filter for (x, y) coordinates."""
    
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-2,
                 initial_x: float = 0.0, initial_y: float = 0.0):
        self.filter_x = KalmanFilter1D(process_variance, measurement_variance, initial_x)
        self.filter_y = KalmanFilter1D(process_variance, measurement_variance, initial_y)
    
    def update(self, x: float, y: float) -> tuple:
        return (self.filter_x.update(x), self.filter_y.update(y))
    
    def filter(self, x: float, y: float) -> tuple:
        return self.update(x, y)
    
    @property
    def value(self) -> tuple:
        return (self.filter_x.value, self.filter_y.value)
    
    def reset(self, x: float = 0.0, y: float = 0.0):
        self.filter_x.reset(x)
        self.filter_y.reset(y)
