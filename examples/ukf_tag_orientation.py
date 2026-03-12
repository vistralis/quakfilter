#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UKF Figure-8 — full 6-DoF tracking with coupled translation and rotation.

Simulates a tag (like an AprilTag on a palm) tracing a figure-8 pattern
while continuously flipping orientation — similar to a person doing
figure-8 hand motions with the palm alternately facing up and down.

Trajectory:
    Position follows a lemniscate (figure-8):
        x(t) = A · sin(ωt)
        y(t) = A · sin(ωt) · cos(ωt) = (A/2) · sin(2ωt)
        z(t) = 0

    Orientation continuously rotates around the local X-axis:
        The tag flips 180° per half-loop, completing one full rotation
        per figure-8 cycle. This couples translation and rotation in
        a way that exercises the manifold-aware quaternion handling.

This example uses the Helical model which naturally handles coordinated
translation and rotation patterns.
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import UKF, RigidBodyHelicalModel


def quaternion_from_axis_angle(axis, angle):
    """Create quaternion [w, x, y, z] from axis and angle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ── Step 1: Configure the filter ─────────────────────────────────────────

model = RigidBodyHelicalModel()

Q = np.eye(15) * 0.01
R = np.eye(6) * 0.05
P0 = np.eye(15) * 1.0

# Figure-8 parameters
amplitude = 0.3  # 30 cm amplitude (a hand-sized motion)
omega = 2 * np.pi / 4.0  # one figure-8 cycle every 4 seconds
flip_rate = omega  # one full rotation per figure-8 cycle

# Compute initial velocity from the lemniscate derivative at t=0
#   dx/dt = A·ω·cos(ωt) → A·ω at t=0
#   dy/dt = A·ω·cos(2ωt) → A·ω at t=0
vx0 = amplitude * omega
vy0 = amplitude * omega

x0 = np.zeros(16)
x0[3] = vx0  # initial vx
x0[4] = vy0  # initial vy
x0[9] = 1.0  # qw = 1 (identity)
x0[13] = flip_rate  # wx — rotation rate around X-axis (palm flip)

ukf = UKF(model, Q, R, P0, x0)

# ── Step 2: Simulate figure-8 with palm flipping ────────────────────────

dt = 0.02  # 50 Hz — faster rate for smooth curves
num_steps = 400  # 8 seconds — two full figure-8 cycles

np.random.seed(42)

# ── Step 3: Predict-correct loop ─────────────────────────────────────────

position_errors = []
orientation_errors = []

for step in range(num_steps):
    time = (step + 1) * dt

    # Ground truth position: lemniscate
    true_x = amplitude * np.sin(omega * time)
    true_y = amplitude * np.sin(omega * time) * np.cos(omega * time)
    true_z = 0.0

    # Ground truth orientation: continuous rotation around X-axis
    flip_angle = flip_rate * time
    true_quat = quaternion_from_axis_angle([1, 0, 0], flip_angle)

    # Noisy measurement
    position_noise = np.random.randn(3) * 0.01  # 1 cm position noise
    noise_rotvec = np.random.randn(3) * 0.02  # small rotation noise
    noise_angle = np.linalg.norm(noise_rotvec)
    if noise_angle > 1e-8:
        noise_quat = quaternion_from_axis_angle(noise_rotvec / noise_angle, noise_angle)
    else:
        noise_quat = np.array([1.0, 0, 0, 0])
    noisy_quat = quaternion_multiply(true_quat, noise_quat)
    noisy_quat = noisy_quat / np.linalg.norm(noisy_quat)

    measurement = np.concatenate(
        [
            [true_x + position_noise[0], true_y + position_noise[1], true_z + position_noise[2]],
            noisy_quat,
        ]
    )

    ukf.predict(dt)
    ukf.correct(measurement)

    # Track errors
    state = ukf.state
    position_error = np.linalg.norm(state[:3] - [true_x, true_y, true_z])
    position_errors.append(position_error)

    # Orientation error: angle between estimated and true quaternion
    q_est = state[9:13]
    q_est = q_est / np.linalg.norm(q_est)  # normalise
    dot = np.clip(abs(np.dot(q_est, true_quat)), 0, 1)
    angle_err = 2 * np.arccos(dot)
    orientation_errors.append(np.degrees(angle_err))

# ── Step 4: Validate results ─────────────────────────────────────────────

state = ukf.state

# Position tracking
avg_pos_error = np.mean(position_errors[-100:])
assert avg_pos_error < 0.05, f"Avg position error too large: {avg_pos_error:.4f} m"

# Orientation tracking
avg_orient_error = np.mean(orientation_errors[-100:])
assert avg_orient_error < 10.0, f"Avg orientation error too large: {avg_orient_error:.2f}°"

# The figure-8 should cross the origin — at t ≈ 2s and t ≈ 4s
# After two full cycles, position should be near origin again
final_time = num_steps * dt
final_true_x = amplitude * np.sin(omega * final_time)
final_true_y = amplitude * np.sin(omega * final_time) * np.cos(omega * final_time)
origin_distance = np.linalg.norm(state[:2] - [final_true_x, final_true_y])
assert origin_distance < 0.1, f"Final position error: {origin_distance:.4f} m"

# Angular velocity around X should be estimated
estimated_wx = state[13]
assert abs(estimated_wx - flip_rate) < 1.0, (
    f"Angular velocity error: {estimated_wx:.3f} vs {flip_rate:.3f} rad/s"
)

# Covariance positive definite
P = ukf.covariance
eigenvalues = np.linalg.eigvalsh(P)
assert np.all(eigenvalues > 0), "Covariance must be positive definite"

# Error should converge
early_pos = np.mean(position_errors[:50])
late_pos = np.mean(position_errors[-50:])

print("UKF Figure-8 — 6-DoF Tracking")
print(f"  Amplitude:          {amplitude:.2f} m")
print(f"  Flip rate:          {flip_rate:.3f} rad/s ({np.degrees(flip_rate):.1f}°/s)")
print(f"  Avg position error: {avg_pos_error:.4f} m (last 2s)")
print(f"  Avg orient. error:  {avg_orient_error:.2f}° (last 2s)")
print(f"  Estimated ωx:       {estimated_wx:.3f} rad/s (true: {flip_rate:.3f})")
print(f"  Early pos error:    {early_pos:.4f} m")
print(f"  Late pos error:     {late_pos:.4f} m")
print(f"  Covariance min λ:   {eigenvalues.min():.6e}")
print("  ✓ All checks passed")
