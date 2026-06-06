"""
linear.py — Standard Linear Kalman Filter.

Discrete-time model:
    x_k  = F · x_{k-1} + B · u_k  + w_k    (w_k ~ N(0, Q))
    z_k  = H · x_k     + v_k               (v_k ~ N(0, R))
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .utils import validate_matrix, validate_positive_definite, symmetrize, safe_invert


# ──────────────────────────────────────────────
# State snapshot
# ──────────────────────────────────────────────

@dataclass
class KalmanState:
    """Holds the current mean and covariance estimate."""
    x: np.ndarray            # state mean,       shape (n,)
    P: np.ndarray            # state covariance, shape (n, n)


# ──────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────

class KalmanFilter:
    """
    Linear (Gaussian) Kalman Filter.

    Parameters
    ----------
    F : (n, n)   State transition matrix
    H : (m, n)   Observation matrix
    Q : (n, n)   Process noise covariance
    R : (m, m)   Measurement noise covariance
    B : (n, c)   Control-input matrix (optional; None = no control)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        B: np.ndarray | None = None,
    ) -> None:
        self.n = F.shape[0]   # state dimension
        self.m = H.shape[0]   # observation dimension

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B

        self._validate_matrices()

        # Will be set by initialize()
        self._state: KalmanState | None = None

    # ── Setup ─────────────────────────────────

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """
        Set the initial state estimate.

        Parameters
        ----------
        x0 : (n,)   Initial state mean
        P0 : (n, n) Initial state covariance
        """
        validate_matrix(x0, "x0", (self.n,))
        validate_matrix(P0, "P0", (self.n, self.n))
        self._state = KalmanState(x=x0.copy(), P=P0.copy())

    # ── Core steps ────────────────────────────

    def predict(self, u: np.ndarray | None = None) -> KalmanState:
        """
        Time-update (predict) step.

        Parameters
        ----------
        u : (c,) Control input vector (optional).

        Returns
        -------
        Predicted KalmanState (prior).
        """
        self._check_initialized()
        x, P = self._state.x, self._state.P

        # State prediction
        x_pred = self.F @ x
        if u is not None and self.B is not None:
            x_pred = x_pred + self.B @ u

        # Covariance prediction
        P_pred = self.F @ P @ self.F.T + self.Q
        P_pred = symmetrize(P_pred)

        self._state = KalmanState(x=x_pred, P=P_pred)
        return KalmanState(x=x_pred.copy(), P=P_pred.copy())

    def update(self, z: np.ndarray) -> KalmanState:
        """
        Measurement-update (correct) step.

        Parameters
        ----------
        z : (m,) Observation vector.

        Returns
        -------
        Updated KalmanState (posterior).
        """
        self._check_initialized()
        validate_matrix(z, "z", (self.m,))

        x, P = self._state.x, self._state.P

        # Innovation and its covariance
        innovation   = z - self.H @ x
        S            = self.H @ P @ self.H.T + self.R

        # Kalman gain
        K = P @ self.H.T @ safe_invert(S)

        # Posterior state
        x_new = x + K @ innovation
        P_new = (np.eye(self.n) - K @ self.H) @ P
        P_new = symmetrize(P_new)

        self._state = KalmanState(x=x_new, P=P_new)
        return KalmanState(x=x_new.copy(), P=P_new.copy())

    def step(
        self, z: np.ndarray, u: np.ndarray | None = None
    ) -> tuple[KalmanState, KalmanState]:
        """
        Convenience: run predict → update in one call.

        Returns
        -------
        (prior, posterior) as a tuple of KalmanState.
        """
        prior     = self.predict(u=u)
        posterior = self.update(z)
        return prior, posterior

    # ── Batch processing ──────────────────────

    def filter_sequence(
        self,
        observations: np.ndarray,
        controls: np.ndarray | None = None,
    ) -> list[KalmanState]:
        """
        Run the filter over a full time-series.

        Parameters
        ----------
        observations : (T, m)  Sequence of observations.
        controls     : (T, c)  Sequence of control inputs (optional).

        Returns
        -------
        List of T posterior KalmanState objects.
        """
        self._check_initialized()
        T = observations.shape[0]
        posteriors: list[KalmanState] = []

        for t in range(T):
            u = controls[t] if controls is not None else None
            _, posterior = self.step(observations[t], u=u)
            posteriors.append(posterior)

        return posteriors

    # ── Smoothing ─────────────────────────────

    def rts_smooth(
        self, posteriors: list[KalmanState]
    ) -> list[KalmanState]:
        """
        Rauch-Tung-Striebel (RTS) smoother pass.

        Runs backward over the list of filter posteriors produced by
        filter_sequence() and returns improved smoothed estimates.

        Parameters
        ----------
        posteriors : list of KalmanState from filter_sequence()

        Returns
        -------
        List of smoothed KalmanState objects (same length).
        """
        T = len(posteriors)
        smoothed = [None] * T
        smoothed[-1] = posteriors[-1]

        for t in range(T - 2, -1, -1):
            x_f   = posteriors[t].x
            P_f   = posteriors[t].P
            P_pred = self.F @ P_f @ self.F.T + self.Q

            G = P_f @ self.F.T @ safe_invert(P_pred)       # smoother gain

            x_s   = x_f + G @ (smoothed[t + 1].x - self.F @ x_f)
            P_s   = P_f + G @ (smoothed[t + 1].P - P_pred) @ G.T
            P_s   = symmetrize(P_s)

            smoothed[t] = KalmanState(x=x_s, P=P_s)

        return smoothed

    # ── Properties ────────────────────────────

    @property
    def state(self) -> KalmanState:
        self._check_initialized()
        return self._state

    # ── Internals ─────────────────────────────

    def _validate_matrices(self) -> None:
        validate_matrix(self.F, "F", (self.n, self.n))
        validate_matrix(self.H, "H", (self.m, self.n))
        validate_matrix(self.Q, "Q", (self.n, self.n))
        validate_matrix(self.R, "R", (self.m, self.m))
        validate_positive_definite(self.Q, "Q")
        validate_positive_definite(self.R, "R")

    def _check_initialized(self) -> None:
        if self._state is None:
            raise RuntimeError("Call initialize() before using the filter.")
