## Implementation of the kalman filter algorithm

[Paper](https://www.unitedthc.com/DSP/Kalman1960.pdf)

## usage

```python
import numpy as np
from kalman import KalmanFilter

dt = 0.1
F  = np.array([[1, dt], [0, 1]])   # constant velocity
H  = np.array([[1, 0]])             # observe position only
Q  = np.eye(2) * 1e-4              # process noise
R  = np.array([[0.5]])              # measurement noise

kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf.initialize(x0=np.array([0.0, 1.0]), P0=np.eye(2))

# Single step
prior, posterior = kf.step(z=np.array([0.12]))
print(posterior.x)   # estimated [position, velocity]

# Full sequence
observations = np.random.randn(100, 1)
posteriors   = kf.filter_sequence(observations)

# RTS smoother pass
smoothed = kf.rts_smooth(posteriors)
```
