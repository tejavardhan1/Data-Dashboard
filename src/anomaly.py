"""Anomaly detection for time-series data."""

from typing import Optional

import numpy as np


def zscore_anomaly(values: list[float], threshold: float = 2.0) -> list[bool]:
    """Flag values that deviate > threshold standard deviations from mean."""
    if not values or len(values) < 3:
        return [False] * len(values)
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return [False] * len(values)
    z = np.abs((arr - mean) / std)
    return (z > threshold).tolist()


def detect_spike(values: list[float], threshold_pct: float = 10.0) -> Optional[int]:
    """Return index of last value that spiked > threshold_pct from previous."""
    if not values or len(values) < 2:
        return None
    for i in range(len(values) - 1, 0, -1):
        prev = values[i - 1]
        if prev and prev != 0:
            ch = abs(values[i] - prev) / prev * 100
            if ch >= threshold_pct:
                return i
    return None


def stress_score(volatility: float, sentiment: float, crash_flag: bool) -> float:
    """
    Market stress 0-100. Higher = more stress.
    volatility: 0-50+ (typical stock vol %)
    sentiment: -1 to 1
    crash_flag: True if recent crash detected
    """
    vol_component = min(50, volatility)
    sent_component = max(0, 25 - sentiment * 25)
    crash_component = 25 if crash_flag else 0
    return min(100, vol_component + sent_component + crash_component)
