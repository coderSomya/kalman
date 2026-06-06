"""
Kalman Filter Library
=====================
A modular, clean implementation of Kalman Filter variants.

Modules:
    linear      - Standard Linear Kalman Filter
    extended    - Extended Kalman Filter (EKF)
    unscented   - Unscented Kalman Filter (UKF)
    utils       - Matrix helpers and validation
    noise       - Noise covariance estimation utilities
"""

from .linear import KalmanFilter
from .extended import ExtendedKalmanFilter
from .unscented import UnscentedKalmanFilter
from .utils import validate_matrix, make_identity, make_zero
from .noise import estimate_noise_covariances

__all__ = [
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "validate_matrix",
    "make_identity",
    "make_zero",
    "estimate_noise_covariances",
]

__version__ = "1.0.0"
