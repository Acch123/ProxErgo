"""Rolling window statistics."""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class StatsSummary:
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int


class RollingStatistics:
    """Efficient rolling window statistics using Welford's algorithm."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._data = deque(maxlen=window_size)
        self._count, self._mean, self._M2 = 0, 0.0, 0.0
        self._min, self._max = float('inf'), float('-inf')
        
    def add(self, value: float):
        if len(self._data) == self.window_size:
            self._remove_from_stats(self._data[0])
        self._data.append(value)
        self._add_to_stats(value)
        
        if len(self._data) == self.window_size:
            self._min, self._max = min(self._data), max(self._data)
        else:
            self._min, self._max = min(self._min, value), max(self._max, value)
    
    def _add_to_stats(self, value: float):
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (value - self._mean)
    
    def _remove_from_stats(self, value: float):
        if self._count <= 1:
            self._count, self._mean, self._M2 = 0, 0.0, 0.0
            return
        self._count -= 1
        delta = value - self._mean
        self._mean -= delta / self._count
        self._M2 = max(0, self._M2 - delta * (value - self._mean))
    
    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else 0.0
    
    @property
    def std(self) -> float:
        return np.sqrt(self._M2 / (self._count - 1)) if self._count > 1 else 0.0
    
    def summary(self) -> StatsSummary:
        return StatsSummary(self.mean, self.std, self._min if self._count > 0 else 0, 
                           self._max if self._count > 0 else 0, 
                           float(np.median(list(self._data))) if self._data else 0, len(self._data))
    
    def reset(self):
        self._data.clear()
        self._count, self._mean, self._M2 = 0, 0.0, 0.0
        self._min, self._max = float('inf'), float('-inf')


class RollingTimeSeries:
    """Rolling time series with timestamp support."""
    
    def __init__(self, window_seconds: float = 60.0):
        self.window_ms = window_seconds * 1000
        self._data: deque = deque()
        
    def add(self, value: float, timestamp: float):
        self._data.append((timestamp, value))
        cutoff = timestamp - self.window_ms
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()
    
    def get_values(self) -> List[float]:
        return [v for _, v in self._data]
    
    @property
    def count(self) -> int:
        return len(self._data)
    
    def reset(self):
        self._data.clear()
