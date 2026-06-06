"""
extended.py — Extended Kalman Filter (EKF).

Handles nonlinear systems by linearising around the current estimate:
    x_k  = f(x_{k-1}, u_k) + w_k    (w_k ~ N(0, Q))
    z_k  = h(x_k)           + v_k    (v_k ~ N(0, R))

The user supplies:
    f  — state transition function
    h  — observation function
    Jf — Jacobian of f w.r.t. x  (or use numerical approximation)
    Jh — Jacobian of h w.r.t. x  (or use numerical approximation)
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from .linear import KalmanState
from .utils import symmetrize, safe_invert, validate_matrix


# ──────────────────────────────────────────────
# Jacobian utilities
# ──────────────────────────────────────────────

def numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute the Jacobian of func at x via central finite differences.

    Parameters
    ----------
    func : callable  f: R^n → R^m
    x    : (n,)      Point at which to evaluate
    eps  : float     Step size

    Returns
    -------
    J : (m, n) Jacobian matrix
    """
    f0 = func(x)
    m, n = f0.shape[0], x.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (func(x + dx) - func(x - dx)) / (2 * eps)
    return J


# ──────────────────────────────────────────────
# EKF
# ──────────────────────────────────────────────

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.

    Parameters
    ----------
    f    : callable (n,) → (n,)          State transition function
    h    : callable (n,) → (m,)          Observation function
    Q    : (n, n)                         Process noise covariance
    R    : (m, m)                         Measurement noise covariance
    Jf   : callable (n,) → (n, n) | None Jacobian of f (auto if None)
    Jh   : callable (n,) → (m, n) | None Jacobian of h (auto if None)
    n    : int                            State dimension
    m    : int                            Observation dimension
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        n: int,
        m: int,
        Jf: Callable[[np.ndarray], np.ndarray] | None = None,
        Jh: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.f  = f
        self.h  = h
        self.Q  = Q
        self.R  = R
        self.n  = n
        self.m  = m

        # Use supplied Jacobians or fall back to numerical differentiation
        self._Jf = Jf if Jf is not None else (lambda x: numerical_jacobian(f, x))
        self._Jh = Jh if Jh is not None else (lambda x: numerical_jacobian(h, x))

        self._state: KalmanState | None = None

    # ── Setup ─────────────────────────────────

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Set initial state mean (n,) and covariance (n, n)."""
        validate_matrix(x0, "x0", (self.n,))
        validate_matrix(P0, "P0", (self.n, self.n))
        self._state = KalmanState(x=x0.copy(), P=P0.copy())

    # ── Core steps ────────────────────────────

    def predict(self, u: np.ndarray | None = None) -> KalmanState:
        """
        EKF predict step: linearise f around the current estimate.

        Parameters
        ----------
        u : control input (currently passed to f if needed via closure)
        """
        self._check_initialized()
        x, P = self._state.x, self._state.P

        F = self._Jf(x)                         # linearised transition
        x_pred = self.f(x)                      # nonlinear propagation
        P_pred = F @ P @ F.T + self.Q
        P_pred = symmetrize(P_pred)

        self._state = KalmanState(x=x_pred, P=P_pred)
        return KalmanState(x=x_pred.copy(), P=P_pred.copy())

    def update(self, z: np.ndarray) -> KalmanState:
        """
        EKF update step: linearise h around the predicted state.

        Parameters
        ----------
        z : (m,) Observation.
        """
        self._check_initialized()
        validate_matrix(z, "z", (self.m,))

        x, P = self._state.x, self._state.P

        H          = self._Jh(x)                # linearised observation
        innovation = z - self.h(x)
        S          = H @ P @ H.T + self.R
        K          = P @ H.T @ safe_invert(S)   # Kalman gain

        x_new = x + K @ innovation
        P_new = (np.eye(self.n) - K @ H) @ P
        P_new = symmetrize(P_new)

        self._state = KalmanState(x=x_new, P=P_new)
        return KalmanState(x=x_new.copy(), P=P_new.copy())

    def step(self, z: np.ndarray, u: np.ndarray | None = None) -> tuple[KalmanState, KalmanState]:
        """Predict then update. Returns (prior, posterior)."""
        prior     = self.predict(u=u)
        posterior = self.update(z)
        return prior, posterior

    def filter_sequence(
        self,
        observations: np.ndarray,
    ) -> list[KalmanState]:
        """
        Run EKF over a full observation sequence.

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
