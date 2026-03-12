# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RigidBodyHelicalModel — constant velocity with correlated rotation."""

import numpy as np

from quak import (
    UKF,
    IMMEstimator,
    RigidBodyHelicalModel,
    SystemModel,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def identity_state():
    """16D state at origin with identity quaternion, at rest."""
    x = np.zeros(16)
    x[9] = 1.0  # qw = 1
    return x


def quaternion_from_axis_angle(axis, angle):
    """Create quaternion [w, x, y, z] from axis and angle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


# ── Construction ─────────────────────────────────────────────────────────


def test_helical_model_instantiation():
    """RigidBodyHelicalModel can be instantiated and is a SystemModel."""
    model = RigidBodyHelicalModel()
    assert isinstance(model, SystemModel)


# ── State Dimensions ─────────────────────────────────────────────────────


def test_helical_model_state_dimensions():
    """Model produces correct state and measurement dimensions inside UKF."""
    model = RigidBodyHelicalModel()
    x0 = identity_state()
    P0 = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01

    ukf = UKF(model, Q, R, P0, x0)
    assert ukf.state.shape == (16,)
    assert ukf.covariance.shape == (15, 15)


# ── Predict: Pure Translation ────────────────────────────────────────────


def test_helical_model_predict_straight_line():
    """Object with linear velocity only should move in a straight line.

    State: vx=2 m/s, no rotation.
    After 1 second: x should be ~2 m.
    """
    model = RigidBodyHelicalModel()
    x0 = identity_state()
    x0[3] = 2.0  # vx = 2 m/s

    P0 = np.eye(15) * 0.1
    Q = np.eye(15) * 0.001
    R = np.eye(6) * 0.01

    ukf = UKF(model, Q, R, P0, x0)
    ukf.predict(1.0)

    state = ukf.state
    assert np.isclose(state[0], 2.0, atol=0.1), f"Expected x≈2.0, got {state[0]:.3f}"
    assert np.isclose(state[1], 0.0, atol=0.1), f"Expected y≈0.0, got {state[1]:.3f}"
    assert np.isclose(state[3], 2.0, atol=0.3), f"Velocity should persist: vx={state[3]:.3f}"


# ── Predict: Pure Rotation ───────────────────────────────────────────────


def test_helical_model_predict_with_rotation():
    """Object with angular velocity only should rotate without translating.

    State: wx=1 rad/s, no linear velocity.
    After 0.5s: should have rotated ~0.5 rad around X.
    """
    model = RigidBodyHelicalModel()
    x0 = identity_state()
    x0[13] = 1.0  # wx = 1 rad/s

    P0 = np.eye(15) * 0.1
    Q = np.eye(15) * 0.001
    R = np.eye(6) * 0.01

    ukf = UKF(model, Q, R, P0, x0)
    ukf.predict(0.5)

    state = ukf.state
    # Position should stay near origin
    assert np.linalg.norm(state[:3]) < 0.1, f"Position should be near origin: {state[:3]}"

    # Quaternion should reflect ~0.5 rad rotation about X
    expected_quat = quaternion_from_axis_angle([1, 0, 0], 0.5)
    q_est = state[9:13] / np.linalg.norm(state[9:13])
    dot = abs(np.dot(q_est, expected_quat))
    angle_error = 2 * np.arccos(np.clip(dot, 0, 1))
    assert angle_error < 0.2, f"Orientation error too large: {np.degrees(angle_error):.1f}°"


# ── Predict-Correct: Trajectory Tracking ─────────────────────────────────


def test_helical_model_trajectory_tracking():
    """UKF with helical model tracks a moving object from noisy measurements."""
    model = RigidBodyHelicalModel()
    x0 = identity_state()

    P0 = np.eye(15) * 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.05

    ukf = UKF(model, Q, R, P0, x0)

    dt = 0.05
    true_velocity = np.array([1.0, 0.5, 0.0])
    np.random.seed(42)

    for step in range(60):
        time = (step + 1) * dt
        true_pos = true_velocity * time
        noise = np.random.randn(3) * 0.05
        measurement = np.concatenate([true_pos + noise, [1, 0, 0, 0]])

        ukf.predict(dt)
        ukf.correct(measurement)

    state = ukf.state
    final_true = true_velocity * 60 * dt
    position_error = np.linalg.norm(state[:3] - final_true)
    assert position_error < 0.5, f"Position error: {position_error:.3f} m"


# ── Coupled Translation + Rotation ───────────────────────────────────────


def test_helical_model_coupled_translation_rotation():
    """Helical model tracks an object with simultaneous linear and angular velocity."""
    model = RigidBodyHelicalModel()

    x0 = identity_state()
    x0[3] = 1.0  # vx = 1 m/s
    x0[13] = 0.5  # wx = 0.5 rad/s

    P0 = np.eye(15) * 0.5
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.02

    ukf = UKF(model, Q, R, P0, x0)

    dt = 0.02
    np.random.seed(123)

    for step in range(100):
        time = (step + 1) * dt
        true_x = 1.0 * time
        rotation_angle = 0.5 * time
        true_quat = quaternion_from_axis_angle([1, 0, 0], rotation_angle)

        position_noise = np.random.randn(3) * 0.01
        measurement = np.concatenate(
            [[true_x + position_noise[0], position_noise[1], position_noise[2]], true_quat]
        )

        ukf.predict(dt)
        ukf.correct(measurement)

    state = ukf.state
    final_time = 100 * dt

    # Position: x should be ~2.0 m
    assert np.isclose(state[0], final_time, atol=0.3), f"x={state[0]:.3f} expected {final_time}"

    # Angular velocity should be estimated
    assert abs(state[13] - 0.5) < 0.5, f"wx={state[13]:.3f} expected ~0.5 rad/s"

    # Covariance should be positive definite
    eigenvalues = np.linalg.eigvalsh(ukf.covariance)
    assert np.all(eigenvalues > 0), "Covariance not positive definite"


# ── IMM Compatibility ────────────────────────────────────────────────────


def test_helical_model_inside_imm():
    """RigidBodyHelicalModel works as a sub-model inside IMMEstimator."""
    model = RigidBodyHelicalModel()

    x0 = identity_state()
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    imm = IMMEstimator(
        models=[model],
        process_noises=[Q],
        measurement_noise=R,
        initial_covariance=P0,
        initial_state=x0,
    )

    imm.predict(0.1)
    measurement = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    imm.correct(measurement)

    assert imm.state.shape == (16,)
    assert np.all(np.isfinite(imm.state))
    assert abs(imm.model_probabilities[0] - 1.0) < 1e-10
