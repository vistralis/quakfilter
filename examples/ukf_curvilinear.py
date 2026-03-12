#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tutorial 02 — UKF with CTRV Model (Curvilinear Motion).

Simulates an object moving in a circle with constant turn rate and velocity.
The CTRV (Constant Turn-Rate and Velocity) model naturally handles this
motion pattern, which a simple Constant Velocity model would struggle with.

The object traces a circle of radius R = v / ω:
    - Linear speed: 3 m/s
    - Turn rate: 0.5 rad/s (≈ 28.6°/s)
    - Expected radius: 6 m
    - Full circle time: 2π / ω ≈ 12.57 s
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import UKF, RigidBodyCTRVModel

# ── Step 1: Configure the filter ─────────────────────────────────────────

model = RigidBodyCTRVModel()

Q = np.eye(15) * 0.01
R = np.eye(6) * 0.05  # reasonably accurate sensor
P0 = np.eye(15) * 1.0

# Start at (radius, 0, 0), heading along +Y (yaw = 90°)
x0 = np.zeros(16)
radius = 6.0
speed = 3.0
turn_rate = speed / radius  # ω = v / R

x0[0] = radius  # px = R (start on the right of the circle)
x0[4] = speed  # vy = speed (moving along +Y initially)
x0[9] = 1.0  # qw = 1
x0[15] = turn_rate  # wz = ω (turning in the XY plane)

ukf = UKF(model, Q, R, P0, x0)

# ── Step 2: Simulate circular ground truth ───────────────────────────────

dt = 0.05  # 20 Hz
num_steps = 200  # 10 seconds — roughly 80% of the circle

np.random.seed(42)

# ── Step 3: Run the predict-correct loop ─────────────────────────────────

position_errors = []

for step in range(num_steps):
    time = (step + 1) * dt
    angle = turn_rate * time

    # Ground truth: object moves on a circle centred at origin
    true_x = radius * np.cos(angle)
    true_y = radius * np.sin(angle)
    true_z = 0.0

    # Quaternion: rotation about Z by the current heading angle
    heading = np.pi / 2 + angle  # started heading along +Y
    true_quat = np.array([np.cos(heading / 2), 0, 0, np.sin(heading / 2)])

    # Add measurement noise
    position_noise = np.random.randn(3) * 0.1
    noise_quat = np.random.randn(3) * 0.01
    measurement = np.array(
        [
            true_x + position_noise[0],
            true_y + position_noise[1],
            true_z + position_noise[2],
            true_quat[0],
            true_quat[1],
            true_quat[2],
            true_quat[3],
        ]
    )
    measurement[3:7] += np.concatenate([[0], noise_quat])
    measurement[3:7] /= np.linalg.norm(measurement[3:7])

    ukf.predict(dt)
    ukf.correct(measurement)

    error = np.linalg.norm(ukf.state[:2] - np.array([true_x, true_y]))
    position_errors.append(error)

# ── Step 4: Validate results ─────────────────────────────────────────────

state = ukf.state
final_angle = turn_rate * num_steps * dt
final_true = np.array([radius * np.cos(final_angle), radius * np.sin(final_angle)])

# Position tracking
position_error = np.linalg.norm(state[:2] - final_true)
assert position_error < 1.0, f"Position error too large: {position_error:.3f} m"

# Velocity magnitude should be close to the true speed
velocity_magnitude = np.linalg.norm(state[3:6])
assert abs(velocity_magnitude - speed) < 1.5, (
    f"Speed error too large: {velocity_magnitude:.2f} vs {speed:.2f}"
)

# Turn rate (angular velocity around Z) should match
estimated_turn_rate = state[15]
assert abs(estimated_turn_rate - turn_rate) < 0.3, (
    f"Turn rate error: {estimated_turn_rate:.3f} vs {turn_rate:.3f}"
)

# Distance from origin should be close to the radius
distance_from_origin = np.linalg.norm(state[:2])
assert abs(distance_from_origin - radius) < 2.0, (
    f"Radius error: {distance_from_origin:.2f} vs {radius:.2f}"
)

# Filter should converge
early_error = np.mean(position_errors[:20])
late_error = np.mean(position_errors[-20:])
assert late_error < early_error, "Filter should converge"

print("Tutorial 02 — UKF Curvilinear Motion (CTRV)")
print(f"  Final position:  ({state[0]:.2f}, {state[1]:.2f})")
print(f"  True position:   ({final_true[0]:.2f}, {final_true[1]:.2f})")
print(f"  Position error:  {position_error:.4f} m")
print(f"  Speed:           {velocity_magnitude:.2f} m/s (true: {speed:.2f})")
print(f"  Turn rate:       {estimated_turn_rate:.3f} rad/s (true: {turn_rate:.3f})")
print(f"  Orbit radius:    {distance_from_origin:.2f} m (true: {radius:.2f})")
print(f"  Early avg error: {early_error:.4f} m")
print(f"  Late avg error:  {late_error:.4f} m")
print("  ✓ All checks passed")
