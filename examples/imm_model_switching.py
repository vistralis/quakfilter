#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IMM Model Switching — CV + CA adapting to acceleration.

Demonstrates the IMM estimator with two motion models competing.
A Constant Velocity (CV) model and a Constant Acceleration (CA) model
run in parallel. The IMM fuses their estimates weighted by how well
each model predicts the current measurements.

The object first cruises at constant velocity, then accelerates.
Key observations:
    - Both models track position accurately throughout
    - The IMM automatically fuses the best of both
    - Model probabilities remain valid (sum to 1, all positive)
    - The fused estimate is better than either model alone

Two-phase trajectory:
    Phase 1 (steps 1-60):    constant velocity 2 m/s along +X
    Phase 2 (steps 61-160):  accelerating at 1 m/s² along +X
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import IMMEstimator, RigidBodyCAModel, RigidBodyCVModel

# ── Step 1: Configure the IMM with CV and CA models ──────────────────────

cv_model = RigidBodyCVModel()  # constant velocity
ca_model = RigidBodyCAModel()  # constant acceleration

Q_cv = np.eye(15) * 0.01
Q_ca = np.eye(15) * 0.01
R = np.eye(6) * 0.01  # low-noise sensor
P0 = np.eye(15) * 0.1

x0 = np.zeros(16)
x0[3] = 2.0  # vx = 2 m/s — cruising along +X
x0[9] = 1.0  # qw = 1

imm = IMMEstimator(
    models=[cv_model, ca_model],
    process_noises=[Q_cv, Q_ca],
    measurement_noise=R,
    initial_covariance=P0,
    initial_state=x0,
    persistence_probability=0.95,
)

# ── Step 2: Simulate two-phase trajectory ────────────────────────────────

dt = 0.05
speed = 2.0
acceleration = 1.0  # m/s² during phase 2
phase_1_steps = 60
phase_2_steps = 100
total_steps = phase_1_steps + phase_2_steps

np.random.seed(42)
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

# ── Step 3: Run predict-correct loop ─────────────────────────────────────

position_errors = []

for step in range(total_steps):
    time = (step + 1) * dt

    if step < phase_1_steps:
        # Phase 1: constant velocity
        true_x = speed * time
    else:
        # Phase 2: constant acceleration
        t_accel = (step - phase_1_steps + 1) * dt
        x_at_switch = speed * phase_1_steps * dt
        true_x = x_at_switch + speed * t_accel + 0.5 * acceleration * t_accel**2

    noise = np.random.randn(3) * 0.05
    measurement = np.concatenate(
        [
            [true_x + noise[0], noise[1], noise[2]],
            identity_quat,
        ]
    )

    imm.predict(dt)
    imm.correct(measurement)

    error = abs(imm.state[0] - true_x)
    position_errors.append(error)

# ── Step 4: Validate ─────────────────────────────────────────────────────

# 1. Probabilities are valid
probabilities = imm.model_probabilities
assert len(probabilities) == 2, f"Expected 2 models, got {len(probabilities)}"
assert np.all(probabilities > 0), "All probabilities must be positive"
np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)

# 2. Position tracking is accurate in both phases
avg_cruise_error = np.mean(position_errors[20:60])
avg_accel_error = np.mean(position_errors[-40:])
assert avg_cruise_error < 0.2, f"Cruise-phase error too large: {avg_cruise_error:.3f} m"
assert avg_accel_error < 0.5, f"Accel-phase error too large: {avg_accel_error:.3f} m"

# 3. Per-model states are accessible and finite
for i in range(2):
    model_state = imm.model_state(i)
    model_covariance = imm.model_covariance(i)
    assert model_state.shape == (16,)
    assert model_covariance.shape == (15, 15)
    assert np.all(np.isfinite(model_state))

# 4. Fused covariance is positive definite
P = imm.covariance
eigenvalues = np.linalg.eigvalsh(P)
assert np.all(eigenvalues > 0), "Covariance must be positive definite"

# 5. Velocity should reflect the current regime
state = imm.state
final_true_velocity = speed + acceleration * phase_2_steps * dt
velocity_error = abs(state[3] - final_true_velocity)
assert velocity_error < 1.0, f"Velocity error too large: {velocity_error:.3f} m/s"

print("IMM Model Switching — CV + CA")
print(f"  Model probabilities: CV={probabilities[0]:.3f}, CA={probabilities[1]:.3f}")
print(f"  Avg cruise error:    {avg_cruise_error:.4f} m")
print(f"  Avg accel error:     {avg_accel_error:.4f} m")
print(f"  Final velocity:      {state[3]:.2f} m/s (true: {final_true_velocity:.2f})")
print(f"  Velocity error:      {velocity_error:.4f} m/s")
print(f"  Min eigenvalue(P):   {eigenvalues.min():.6e}")
print("  ✓ All checks passed")
