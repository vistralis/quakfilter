# Q/R Tuning Guide

Adjusting the process noise covariance **Q** and measurement noise covariance **R** is the most important step in getting good filter performance. This guide provides practical tuning strategies.

## Quick Reference

| Matrix | Dimension | What It Represents |
|--------|-----------|-------------------|
| **Q** (Process Noise) | 15×15 | How much the state can change unpredictably per timestep |
| **R** (Measurement Noise) | Sensor-dependent | How noisy the sensor readings are |

## Process Noise (Q)

The 15×15 diagonal of Q maps to the error-state components:

| Index | Component | Typical Range | Meaning |
|-------|-----------|---------------|---------|
| 0–2 | Position (m) | 0.001 – 0.1 | How much position drifts without measurement |
| 3–5 | Velocity (m/s) | 0.01 – 1.0 | How much velocity can change unexpectedly |
| 6–8 | Acceleration (m/s²) | 0.01 – 1.0 | How much acceleration drifts (CA/Singer models) |
| 9–11 | Orientation (rad) | 0.001 – 0.05 | How much orientation drifts |
| 12–14 | Angular velocity (rad/s) | 0.01 – 0.5 | How much angular velocity can change |

### Tuning Strategy

| Symptom | Cause | Fix |
|---------|-------|-----|
| Filter lags behind fast maneuvers | Q too low | Increase velocity/angular velocity Q components |
| Output is noisy/jittery | Q too high | Decrease Q — filter trusts model less |
| Filter diverges after occlusion | Q too low for position | Increase position Q to allow drift during missed measurements |

## Measurement Noise (R)

R dimensions depend on the sensor type:

| Sensor | R Dimension | Components |
|--------|-------------|------------|
| `PoseSensor` | 6×6 | Position (3) + Orientation tangent (3) |
| `VelocitySensor` | 3×3 | Linear velocity (3) |
| `ProjectiveBBoxSensor` | 4×4 | Bounding box [u, v, w, h] |

### Tuning Strategy

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output chases measurement noise | R too low | Increase R — filter trusts measurement less |
| Output drifts from ground truth | R too high | Decrease R — filter ignores valid measurements |
| Covariance collapses to zero | R much smaller than actual noise | Set R to measured sensor variance |

## Bandwidth Trade-off

The ratio **Q/R** determines filter bandwidth:

```
High Q / Low R  →  High bandwidth  →  Fast response, more noise
Low Q / High R  →  Low bandwidth   →  Slow response, smooth output
```

### Practical Recipe

1. **Measure R:** Record sensor data with a static target. Compute variance per axis. Use as baseline R diagonal.
2. **Start Q small:** Set all Q diagonals to `0.01 * R_diagonal`.
3. **Increase velocity Q** until the filter tracks fast maneuvers without lag.
4. **Increase position Q** if the filter must survive measurement gaps.
5. **Fine-tune** by observing the innovation sequence — it should be zero-mean with covariance ≈ `Pᵥᵥ`.

## Multi-Sensor Setup

When using multiple sensors, each has its own R matrix:

```python
from quak import PoseSensor, VelocitySensor

pose_sensor = PoseSensor(R_pose)    # 6×6 R
vel_sensor  = VelocitySensor(R_vel) # 3×3 R

ukf.predict(dt=0.01)
ukf.correct_with_sensor(z_pose, pose_sensor)
ukf.correct_with_sensor(z_vel, vel_sensor)
```

Each sensor's R should reflect that sensor's actual noise characteristics.

## NoiseModel Helpers

The library provides convenience constructors:

```python
import quak

# Diagonal from per-axis standard deviations
Q = quak.NoiseModel.Diagonal(np.array([0.01]*15))

# Isotropic (same sigma for all axes)
R = quak.NoiseModel.Isotropic(6, 0.05)

# From full covariance matrix
R = quak.NoiseModel.Gaussian(custom_cov_matrix)
```

## Innovation-Based Diagnostics

After `correct()`, the innovation covariance `Pᵥᵥ` is available:

```python
ukf.correct(z)
S = ukf.get_innovation_covariance()  # Pᵥᵥ
```

Use it for:
- **Gating:** Reject outlier measurements via Mahalanobis distance: `d² = νᵀ Pᵥᵥ⁻¹ ν`
- **Consistency check:** The normalized innovation squared should follow a χ² distribution with `dim(z)` degrees of freedom
