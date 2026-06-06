"""
noise.py — Utilities for estimating process and measurement noise covariances
           from empirical data.
"""

import numpy as np


def estimate_noise_covariances(
    observations: np.ndarray,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate Q (process noise) and R (measurement noise) from raw observations.

    Uses a simple heuristic:
      - R  ≈ variance of successive observation differences (sensor noise).
      - Q  ≈ fraction of R scaled to state_dim (conservative prior).

    Parameters
    ----------
    observations : ndarray, shape (T, obs_dim)
        Time-series of raw sensor readings.
    state_dim : int
        Dimensionality of the state vector.

    Returns
    -------
    Q : ndarray, shape (state_dim, state_dim)
    R : ndarray, shape (obs_dim, obs_dim)
    """
    if observations.ndim == 1:
        observations = observations[:, np.newaxis]

    obs_dim = observations.shape[1]
    diffs = np.diff(observations, axis=0)
    R = np.diag(np.var(diffs, axis=0)) / 2          # measurement noise estimate
    q_scale = np.mean(np.diag(R)) * 0.01            # process noise ≈ 1% of R
    Q = np.eye(state_dim) * q_scale
    return Q, R


def sample_covariance(data: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased sample covariance matrix of row-wise data.

    Parameters
    ----------
    data : ndarray, shape (T, n)

    Returns
    -------
    cov : ndarray, shape (n, n)
    """
    return np.cov(data, rowvar=False, bias=False)
