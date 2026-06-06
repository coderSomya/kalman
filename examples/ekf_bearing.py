"""
examples/ekf_bearing.py
------------------------
Bearing-only target tracking with the Extended Kalman Filter.

State  : [x, y, vx, vy]   (2-D position + velocity)
Measure: [bearing angle]   (nonlinear: arctan2(y, x))
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalman import ExtendedKalmanFilter

dt = 0.5   # s

# ── Nonlinear functions ───────────────────────

def f(x: np.ndarray) -> np.ndarray:
    """Constant-velocity motion model."""
    return np.array([
        x[0] + dt * x[2],
        x[1] + dt * x[3],
        x[2],
        x[3],
    ])

def h(x: np.ndarray) -> np.ndarray:
    """Bearing measurement: angle from origin to target."""
    return np.array([np.arctan2(x[1], x[0])])

# Analytical Jacobians (optional — EKF can compute numerically)
def Jf(x: np.ndarray) -> np.ndarray:
    return np.array([
        [1, 0, dt, 0],
        [0, 1,  0, dt],
        [0, 0,  1, 0],
        [0, 0,  0, 1],
    ])

def Jh(x: np.ndarray) -> np.ndarray:
    r2 = x[0]**2 + x[1]**2 + 1e-9
    return np.array([[-x[1]/r2, x[0]/r2, 0, 0]])

# ── Noise ─────────────────────────────────────
Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
R = np.array([[np.deg2rad(2.0)**2]])          # ≈ 2° bearing noise

# ── Initialise ────────────────────────────────
ekf = ExtendedKalmanFilter(f=f, h=h, Q=Q, R=R, n=4, m=1, Jf=Jf, Jh=Jh)
ekf.initialize(
    x0=np.array([10.0, 5.0, -0.5, 0.2]),
    P0=np.diag([1.0, 1.0, 0.1, 0.1]),
)

# ── Simulate & filter ─────────────────────────
rng = np.random.default_rng(0)
true_states, estimates = [], []
state = np.array([10.0, 5.0, -0.5, 0.2])

for _ in range(40):
    state = f(state) + rng.multivariate_normal(np.zeros(4), Q)
    bearing = h(state) + rng.multivariate_normal(np.zeros(1), R)
    true_states.append(state.copy())

    _, post = ekf.step(bearing)
    estimates.append(post.x.copy())

true_states = np.array(true_states)
estimates   = np.array(estimates)
pos_errors  = np.linalg.norm(true_states[:, :2] - estimates[:, :2], axis=1)

print(f"EKF bearing | Mean 2-D position error: {pos_errors.mean():.4f} m")
print(f"              Max  2-D position error: {pos_errors.max():.4f} m")
