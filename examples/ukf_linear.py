#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tutorial 01 — UKF with Constant Velocity Model (Linear Motion).

Simulates an object moving in a straight line at constant velocity.
Measurements are noisy 6-DoF pose observations (position + quaternion).
The UKF tracks the object and recovers its position and velocity.

State vector (16D):
    [px, py, pz, vx, vy, vz, ax, ay, az, qw, qx, qy, qz, wx, wy, wz]

Measurement (7D):
    [px, py, pz, qw, qx, qy, qz]
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import UKF, RigidBodyCVModel

# ── Step 1: Configure the filter ─────────────────────────────────────────

model = RigidBodyCVModel()

# Process noise — how much we trust the model vs measurements.
# Smaller Q = more trust in the constant-velocity assumption.
Q = np.eye(15) * 0.01

# Measurement noise — how noisy our sensor readings are.
# Smaller R = more trust in measurements.
R = np.eye(6) * 0.1

# Initial state error covariance — our initial uncertainty.
P0 = np.eye(15) * 1.0

# Initial state: object at origin, identity quaternion, at rest.
x0 = np.zeros(16)
x0[9] = 1.0  # qw = 1 → identity quaternion

ukf = UKF(model, Q, R, P0, x0)

# ── Step 2: Simulate ground truth ────────────────────────────────────────

dt = 0.05  # 20 Hz
num_steps = 100
true_velocity = np.array([2.0, 1.0, 0.5])  # m/s
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

np.random.seed(42)

# ── Step 3: Run the predict-correct loop ─────────────────────────────────

position_errors = []

for step in range(num_steps):
    time = (step + 1) * dt
    true_position = true_velocity * time

    # Simulate noisy measurement: true pose + Gaussian noise
    noise_position = np.random.randn(3) * 0.1
    noise_orientation = np.random.randn(3) * 0.01  # small orientation noise
    measurement = np.concatenate(
        [
            true_position + noise_position,
            identity_quat,
        ]
    )
    # Perturb quaternion slightly and renormalise
    measurement[3:7] += np.concatenate([[0], noise_orientation])
    measurement[3:7] /= np.linalg.norm(measurement[3:7])

    # Predict: advance the internal model by dt
    ukf.predict(dt)

    # Correct: incorporate the measurement
    ukf.correct(measurement)

    # Record tracking error (use .state property)
    position_error = np.linalg.norm(ukf.state[:3] - true_position)
    position_errors.append(position_error)

# ── Step 4: Validate results ─────────────────────────────────────────────

state = ukf.state
final_true_position = true_velocity * num_steps * dt

# Position should converge close to ground truth
position_error = np.linalg.norm(state[:3] - final_true_position)
assert position_error < 0.5, f"Position error too large: {position_error:.3f} m"

# Velocity estimate should converge to the true velocity
velocity_error = np.linalg.norm(state[3:6] - true_velocity)
assert velocity_error < 1.0, f"Velocity error too large: {velocity_error:.3f} m/s"

# Acceleration should be near zero for constant velocity
acceleration = np.linalg.norm(state[6:9])
assert acceleration < 1.0, f"Acceleration should be near zero: {acceleration:.3f} m/s²"

# Error should decrease over time (filter converges)
early_error = np.mean(position_errors[:10])
late_error = np.mean(position_errors[-10:])
assert late_error < early_error, "Filter should improve over time"

print("Tutorial 01 — UKF Linear Motion")
print(f"  Final position:  {state[:3]}  (true: {final_true_position})")
print(f"  Position error:  {position_error:.4f} m")
print(f"  Velocity:        {state[3:6]}  (true: {true_velocity})")
print(f"  Velocity error:  {velocity_error:.4f} m/s")
print(f"  Early avg error: {early_error:.4f} m")
print(f"  Late avg error:  {late_error:.4f} m")
print("  ✓ All checks passed")
