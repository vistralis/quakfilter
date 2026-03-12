#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tutorial 03 — IMMEstimator with a Single Model.

Demonstrates that IMMEstimator works correctly even with just one model.
This is useful as a baseline — when only one model is given, the IMM
automatically uses a fast single-model code path (no mixing step).

This example tracks the same straight-line motion as Tutorial 01,
but uses IMMEstimator instead of raw UKF.
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import IMMEstimator, RigidBodyCVModel

# ── Step 1: Configure the IMM with a single model ───────────────────────

model = RigidBodyCVModel()

Q = np.eye(15) * 0.01
R = np.eye(6) * 0.1
P0 = np.eye(15) * 1.0

x0 = np.zeros(16)
x0[9] = 1.0  # identity quaternion

# IMMEstimator takes lists of models and Q matrices.
# With a single model, it collapses to a plain UKF internally.
imm = IMMEstimator(
    models=[model],
    process_noises=[Q],
    measurement_noise=R,
    initial_covariance=P0,
    initial_state=x0,
)

# ── Step 2: Simulate constant velocity motion ───────────────────────────

dt = 0.05
num_steps = 100
true_velocity = np.array([2.0, 1.0, 0.5])
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

np.random.seed(42)

# ── Step 3: Run predict-correct loop ─────────────────────────────────────

for step in range(num_steps):
    time = (step + 1) * dt
    true_position = true_velocity * time

    noise = np.random.randn(3) * 0.1
    measurement = np.concatenate([true_position + noise, identity_quat])

    imm.predict(dt)
    imm.correct(measurement)

# ── Step 4: Validate ─────────────────────────────────────────────────────

state = imm.state
final_true = true_velocity * num_steps * dt

# Position
position_error = np.linalg.norm(state[:3] - final_true)
assert position_error < 0.5, f"Position error: {position_error:.3f} m"

# Velocity
velocity_error = np.linalg.norm(state[3:6] - true_velocity)
assert velocity_error < 1.0, f"Velocity error: {velocity_error:.3f} m/s"

# Model probabilities: with one model, probability must be 1.0
probabilities = imm.model_probabilities
assert len(probabilities) == 1, "Should have exactly one model"
assert abs(probabilities[0] - 1.0) < 1e-10, "Single model probability must be 1.0"

# Covariance should be positive definite
P = imm.covariance
eigenvalues = np.linalg.eigvalsh(P)
assert np.all(eigenvalues > 0), "Covariance must be positive definite"

print("Tutorial 03 — IMMEstimator (Single Model)")
print(f"  Final position:     {state[:3]}")
print(f"  Position error:     {position_error:.4f} m")
print(f"  Velocity error:     {velocity_error:.4f} m/s")
print(f"  Model probability:  {probabilities[0]:.6f}")
print(f"  Min eigenvalue(P):  {eigenvalues.min():.6e}")
print("  ✓ All checks passed")
