"""
unscented.py — Unscented Kalman Filter (UKF).

Uses the Unscented Transform to propagate a set of deterministic sigma
points through nonlinear functions — no Jacobians required.

    x_k  = f(x_{k-1}) + w_k    (w_k ~ N(0, Q))
    z_k  = h(x_k)      + v_k    (v_k ~ N(0, R))
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from .linear import KalmanState
from .utils import symmetrize, safe_invert, validate_matrix, nearest_positive_definite


# ──────────────────────────────────────────────
# Sigma point generation (Merwe scaled)
# ──────────────────────────────────────────────

def compute_sigma_points(
    x: np.ndarray,
    P: np.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2n+1 sigma points and their weights (Van der Merwe scaling).

    Parameters
    ----------
    x     : (n,)    State mean
    P     : (n, n)  State covariance
    alpha : spread parameter (1e-3 typical)
    beta  : distribution parameter (2 for Gaussian)
    kappa : secondary scaling (0 typical)

    Returns
    -------
    sigma_pts : (2n+1, n)
    Wm        : (2n+1,)   mean weights
    Wc        : (2n+1,)   covariance weights
    """
    n      = x.shape[0]
    lam    = alpha ** 2 * (n + kappa) - n

    # Cholesky of scaled covariance — repair if not PD
    try:
        L = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky((n + lam) * nearest_positive_definite(P))

    sigma_pts        = np.zeros((2 * n + 1, n))
    sigma_pts[0]     = x
    sigma_pts[1:n+1] = x + L.T
    sigma_pts[n+1:]  = x - L.T

    Wm    = np.full(2 * n + 1, 0.5 / (n + lam))
    Wm[0] = lam / (n + lam)

    Wc    = Wm.copy()
    Wc[0] = Wm[0] + (1 - alpha ** 2 + beta)

    return sigma_pts, Wm, Wc


def unscented_transform(
    sigma_pts: np.ndarray,
    Wm: np.ndarray,
    Wc: np.ndarray,
    noise_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and covariance after propagating sigma points.

    Parameters
    ----------
    sigma_pts : (2n+1, k)
    Wm        : (2n+1,)
    Wc        : (2n+1,)
    noise_cov : (k, k)

    Returns
    -------
    mean : (k,)
    cov  : (k, k)
    """
    mean = Wm @ sigma_pts

    diff = sigma_pts - mean                          # (2n+1, k)
    cov  = (Wc[:, None] * diff).T @ diff + noise_cov
    cov  = symmetrize(cov)

    return mean, cov


# ──────────────────────────────────────────────
# UKF
# ──────────────────────────────────────────────

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter — derivative-free nonlinear estimation.

    Parameters
    ----------
    f     : callable (n,) → (n,)   State transition function
    h     : callable (n,) → (m,)   Observation function
    Q     : (n, n)                  Process noise covariance
    R     : (m, m)                  Measurement noise covariance
    n     : int                     State dimension
    m     : int                     Observation dimension
    alpha : float                   Sigma spread    (default 1e-3)
    beta  : float                   Prior knowledge (default 2, Gaussian)
    kappa : float                   Secondary scale (default 0)
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        n: int,
        m: int,
        alpha: float = 1e-3,
        beta:  float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.f     = f
        self.h     = h
        self.Q     = Q
        self.R     = R
        self.n     = n
        self.m     = m
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

        self._state: KalmanState | None = None

    # ── Setup ─────────────────────────────────

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Set initial state mean (n,) and covariance (n, n)."""
        validate_matrix(x0, "x0", (self.n,))
        validate_matrix(P0, "P0", (self.n, self.n))
        self._state = KalmanState(x=x0.copy(), P=P0.copy())

    # ── Core steps ────────────────────────────

    def predict(self) -> KalmanState:
        """
        UKF predict step via Unscented Transform through f.
        """
        self._check_initialized()
        x, P = self._state.x, self._state.P

        sigma_pts, Wm, Wc = compute_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        # Propagate each sigma point through f
        f_sigma = np.array([self.f(s) for s in sigma_pts])

        x_pred, P_pred = unscented_transform(f_sigma, Wm, Wc, self.Q)

        self._state = KalmanState(x=x_pred, P=P_pred)
        return KalmanState(x=x_pred.copy(), P=P_pred.copy())

    def update(self, z: np.ndarray) -> KalmanState:
        """
        UKF update step via Unscented Transform through h.

        Parameters
        ----------
        z : (m,) Observation.
        """
        self._check_initialized()
        validate_matrix(z, "z", (self.m,))

        x, P = self._state.x, self._state.P

        sigma_pts, Wm, Wc = compute_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        # Propagate sigma points through h
        h_sigma = np.array([self.h(s) for s in sigma_pts])

        z_pred, S = unscented_transform(h_sigma, Wm, Wc, self.R)

        # Cross-covariance P_xz
        dx = sigma_pts - x                      # (2n+1, n)
        dz = h_sigma   - z_pred                 # (2n+1, m)
        P_xz = (Wc[:, None] * dx).T @ dz       # (n, m)

        K          = P_xz @ safe_invert(S)      # Kalman gain  (n, m)
        innovation = z - z_pred

        x_new = x + K @ innovation
        P_new = P - K @ S @ K.T
        P_new = symmetrize(P_new)

        self._state = KalmanState(x=x_new, P=P_new)
        return KalmanState(x=x_new.copy(), P=P_new.copy())

    def step(self, z: np.ndarray) -> tuple[KalmanState, KalmanState]:
        """Predict then update. Returns (prior, posterior)."""
        prior     = self.predict()
        posterior = self.update(z)
        return prior, posterior

    def filter_sequence(
        self,
        observations: np.ndarray,
    ) -> list[KalmanState]:
        """
        Run UKF over a full observation sequence.

        Parameters
        ----------
        observations : (T, m)

        Returns
        -------
        List of T posterior KalmanState objects.
        """
        self._check_initialized()
        posteriors = []
        for z in observations:
            _, posterior = self.step(z)
            posteriors.append(posterior)
        return posteriors

    # ── Properties ────────────────────────────

    @property
    def state(self) -> KalmanState:
        self._check_initialized()
        return self._state

    # ── Internals ─────────────────────────────

    def _check_initialized(self) -> None:
        if self._state is None:
            raise RuntimeError("Call initialize() before using the filter.")
