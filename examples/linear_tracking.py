"""
examples/linear_tracking.py
----------------------------
1-D constant-velocity tracking with the Linear Kalman Filter.

State  : [position, velocity]
Measure: [position]
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalman import KalmanFilter

# ── Model ─────────────────────────────────────
dt = 0.1   # time step (s)

F = np.array([[1, dt],   # state transition: pos += vel*dt
              [0,  1]])

H = np.array([[1, 0]])   # we only observe position

Q = np.array([[1e-4, 0  ],   # small process noise
              [0,    1e-4]])

R = np.array([[0.5]])        # position measurement noise (std ≈ 0.7 m)

# ── Initialise ────────────────────────────────
kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf.initialize(
    x0=np.array([0.0, 1.0]),          # start at 0 m, moving at 1 m/s
    P0=np.eye(2) * 1.0,
)

# ── Simulate & filter ─────────────────────────
rng  = np.random.default_rng(42)
true_pos, estimates = [], []
pos, vel = 0.0, 1.0

for _ in range(50):
    pos += vel * dt
    noisy_obs = np.array([pos + rng.normal(0, 0.5)])
    true_pos.append(pos)

    _, posterior = kf.step(noisy_obs)
    estimates.append(posterior.x[0])

# ── Results ───────────────────────────────────
errors = np.abs(np.array(true_pos) - np.array(estimates))
print(f"Linear KF | Mean abs. position error: {errors.mean():.4f} m")
print(f"           Max  abs. position error: {errors.max():.4f} m")
