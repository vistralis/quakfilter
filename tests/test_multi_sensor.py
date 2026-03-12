# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-sensor correct() — correcting a single UKF with different sensors.

Verifies:
1. PoseSensor produces same results as built-in correct()
2. VelocitySensor extracts correct state components
3. Multi-sensor fusion: pose + velocity on same filter
4. ProjectiveBBoxSensor standalone (without dynamics wrapper)
"""

import numpy as np

from quak import (
    UKF,
    CameraParams,
    PoseSensor,
    ProjectiveBBoxSensor,
    RigidBodyCVModel,
    VelocitySensor,
)


def make_state(pos=(0, 0, 0), vel=(0, 0, 0), acc=(0, 0, 0), quat=(1, 0, 0, 0), angvel=(0, 0, 0)):
    x = np.zeros(16)
    x[0:3] = pos
    x[3:6] = vel
    x[6:9] = acc
    x[9:13] = quat
    x[13:16] = angvel
    return x


def make_pose_measurement(pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    z = np.zeros(7)
    z[0:3] = pos
    z[3:7] = quat
    return z


# ── PoseSensor equivalence ────────────────────────────────────────────


def test_pose_sensor_matches_builtin_correct():
    """PoseSensor correct_with_sensor should produce same result as built-in correct."""
    model = RigidBodyCVModel()
    x0 = make_state(quat=(1, 0, 0, 0))
    P0 = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001

    # UKF 1: uses built-in correct
    ukf1 = UKF(model, Q, R, P0, x0)
    ukf1.predict(dt=0.1)
    z = make_pose_measurement(pos=(1, 0, 0), quat=(1, 0, 0, 0))
    ukf1.correct(z)

    # UKF 2: uses PoseSensor correct_with_sensor
    ukf2 = UKF(model, Q, R, P0, x0)
    ukf2.predict(dt=0.1)
    pose_sensor = PoseSensor(R)
    ukf2.correct(z, sensor=pose_sensor)

    np.testing.assert_allclose(ukf1.x, ukf2.x, atol=1e-10)
    np.testing.assert_allclose(ukf1.P, ukf2.P, atol=1e-10)


# ── VelocitySensor ────────────────────────────────────────────────────


def test_velocity_sensor_extracts_correct_components():
    """VelocitySensor should extract vel and angvel from state."""
    R_vel = np.eye(6) * 0.01
    vel_sensor = VelocitySensor(R_vel)

    state = make_state(pos=(1, 2, 3), vel=(4, 5, 6), angvel=(0.1, 0.2, 0.3))
    z = vel_sensor.predict(state)

    np.testing.assert_allclose(z[:3], [4, 5, 6])
    np.testing.assert_allclose(z[3:6], [0.1, 0.2, 0.3])


def test_velocity_sensor_dimensions():
    R_vel = np.eye(6) * 0.01
    vel_sensor = VelocitySensor(R_vel)
    assert vel_sensor.raw_dim() == 6
    assert vel_sensor.tangent_dim() == 6


# ── Multi-sensor fusion ──────────────────────────────────────────────


def test_multi_sensor_pose_and_velocity():
    """Fusing pose + velocity measurements should converge faster than pose alone."""
    model = RigidBodyCVModel()
    x0 = make_state(quat=(1, 0, 0, 0))
    P0 = np.eye(15) * 1.0  # Large initial uncertainty
    Q = np.eye(15) * 0.01
    R_default = np.eye(6) * 0.01
    R_pose = np.eye(6) * 0.01
    R_vel = np.eye(6) * 0.01

    pose_sensor = PoseSensor(R_pose)
    vel_sensor = VelocitySensor(R_vel)

    # UKF with pose only
    ukf_pose_only = UKF(model, Q, R_default, P0, x0)

    # UKF with pose + velocity
    ukf_multi = UKF(model, Q, R_default, P0, x0)

    true_pos = np.array([0.0, 0.0, 0.0])
    true_vel = np.array([1.0, 0.5, 0.0])
    rng = np.random.RandomState(42)
    dt = 0.1

    for step in range(50):
        true_pos += true_vel * dt

        # Both predict
        ukf_pose_only.predict(dt=dt)
        ukf_multi.predict(dt=dt)

        # Pose measurement (both get this)
        z_pose = make_pose_measurement(pos=true_pos, quat=(1, 0, 0, 0))
        z_pose[:3] += rng.randn(3) * 0.1
        ukf_pose_only.correct(z_pose)
        ukf_multi.correct(z_pose, sensor=pose_sensor)

        # Velocity measurement (only multi gets this)
        z_vel = np.zeros(6)
        z_vel[:3] = true_vel + rng.randn(3) * 0.1
        ukf_multi.correct(z_vel, sensor=vel_sensor)

    # Multi-sensor should have lower velocity covariance
    # (more information → less uncertainty)
    vel_cov_pose_only = np.trace(ukf_pose_only.P[3:6, 3:6])
    vel_cov_multi = np.trace(ukf_multi.P[3:6, 3:6])

    print(f"\n  Velocity cov trace (pose only): {vel_cov_pose_only:.6f}")
    print(f"  Velocity cov trace (pose+vel):  {vel_cov_multi:.6f}")

    assert vel_cov_multi < vel_cov_pose_only, (
        f"Multi-sensor should have lower velocity uncertainty: "
        f"multi={vel_cov_multi:.6f} vs pose_only={vel_cov_pose_only:.6f}"
    )


# ── ProjectiveBBoxSensor standalone ───────────────────────────────────


def test_projective_bbox_sensor_standalone():
    """ProjectiveBBoxSensor should work as standalone sensor via correct_with_sensor."""
    model = RigidBodyCVModel()
    x0 = make_state(pos=(0, 0, 5), vel=(0.1, 0, 0), quat=(1, 0, 0, 0))
    P0 = np.eye(15) * 0.5
    Q = np.eye(15) * 0.01
    R_default = np.eye(6) * 0.01

    cam = CameraParams()
    cam.K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    cam.R_cw = np.eye(3)
    cam.t_cw = np.zeros(3)
    cam.obj_dims = np.array([0.5, 1.8, 0.5])
    cam.min_depth = 0.1

    R_cam = np.eye(4) * 4.0  # pixel-level noise
    cam_sensor = ProjectiveBBoxSensor(cam, R_cam)

    assert cam_sensor.raw_dim() == 4
    assert cam_sensor.tangent_dim() == 4

    ukf = UKF(model, Q, R_default, P0, x0)

    # Simulate tracking with camera measurements
    true_pos = np.array([0.0, 0.0, 5.0])
    rng = np.random.RandomState(99)
    dt = 0.1

    for step in range(30):
        true_pos[0] += 0.1 * dt

        ukf.predict(dt=dt)

        # Simulate bbox measurement
        z_cam = cam_sensor.predict(make_state(pos=true_pos, quat=(1, 0, 0, 0)))
        z_cam += rng.randn(4) * 2.0  # pixel noise
        ukf.correct(z_cam, sensor=cam_sensor)

    state = ukf.x
    assert np.all(np.isfinite(state)), "State contains NaN/Inf"
    pos_error = np.linalg.norm(state[:3] - true_pos)
    assert pos_error < 2.0, f"Position error too large: {pos_error:.3f}"
