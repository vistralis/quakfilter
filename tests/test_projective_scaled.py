# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TDD: Projective bbox model MUST work with scaled sigma points.

These tests reproduce the divergence bug caused by mAlpha=1e-3 creating
a catastrophic negative W0_cov weight (~-1,000,000 for n=15).

With correct alpha (1.0), all weights are positive and the projective
model should converge normally with scaled_sigma_points=True.
"""

import numpy as np

from quak import (
    UKF,
    CameraParams,
    ProjectiveBBoxSensor,
    RigidBodyCVModel,
)


def make_state(pos=(0, 0, 0), vel=(0, 0, 0), quat=(1, 0, 0, 0)):
    x = np.zeros(16)
    x[0:3] = pos
    x[3:6] = vel
    x[9:13] = quat
    return x


def make_camera():
    cam = CameraParams()
    cam.K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    cam.R_cw = np.eye(3)
    cam.t_cw = np.zeros(3)
    cam.obj_dims = np.array([0.5, 1.8, 0.5])
    cam.min_depth = 0.1
    return cam


# ── RED: These MUST pass — projective tracking with scaled sigma points ──


def test_projective_tracking_converges_with_scaled_sigma_points():
    """Projective bbox tracking should converge with scaled_sigma_points=True.

    This is the core regression test for the mAlpha bug. With alpha=1e-3,
    W0_cov ~ -1,000,000 for n=15 tangent dims, destroying the innovation
    covariance on the first correction step.
    """
    model = RigidBodyCVModel()
    cam = make_camera()
    R_cam = np.eye(4) * 4.0
    cam_sensor = ProjectiveBBoxSensor(cam, R_cam)

    x0 = make_state(pos=(1.0, 0.0, 8.0), vel=(0.5, 0.0, 0.0))
    P0 = np.eye(15) * 1.0
    Q = np.eye(15) * 0.01
    R_default = np.eye(6) * 0.01

    # Explicitly use scaled sigma points — this is the default
    ukf = UKF(model, Q, R_default, P0, x0, scaled_sigma_points=True)

    np.random.seed(42)
    dt = 0.1
    true_pos = np.array([1.0, 0.0, 8.0])
    true_vel = np.array([0.5, 0.0, 0.0])

    for step in range(20):
        true_pos = true_pos + true_vel * dt
        ukf.predict(dt)

        z_true = cam_sensor.predict(make_state(pos=true_pos, quat=(1, 0, 0, 0)))
        z_noisy = z_true + np.random.randn(4) * 2.0
        ukf.correct(z_noisy, sensor=cam_sensor)

    state = ukf.state
    assert np.all(np.isfinite(state)), "State diverged to NaN/Inf"
    pos_error = np.linalg.norm(state[:3] - true_pos)
    assert pos_error < 5.0, f"Position error too large: {pos_error:.3f} m"


def test_projective_covariance_stays_bounded_with_scaled_sigma_points():
    """Covariance should not explode when using scaled sigma points."""
    model = RigidBodyCVModel()
    cam = make_camera()
    R_cam = np.eye(4) * 4.0
    cam_sensor = ProjectiveBBoxSensor(cam, R_cam)

    x0 = make_state(pos=(0.0, 0.0, 5.0), vel=(0.1, 0.0, 0.0))
    P0 = np.eye(15) * 0.5
    Q = np.eye(15) * 0.01
    R_default = np.eye(6) * 0.01

    ukf = UKF(model, Q, R_default, P0, x0, scaled_sigma_points=True)

    np.random.seed(99)
    true_pos = np.array([0.0, 0.0, 5.0])
    dt = 0.1

    for step in range(10):
        true_pos[0] += 0.1 * dt
        ukf.predict(dt)

        z = cam_sensor.predict(make_state(pos=true_pos, quat=(1, 0, 0, 0)))
        z += np.random.randn(4) * 2.0
        ukf.correct(z, sensor=cam_sensor)

    P = ukf.covariance
    P_max = np.max(np.abs(np.diag(P)))
    assert P_max < 100.0, f"Covariance exploded: max diag = {P_max:.1f}"


def test_projective_scaled_and_symmetric_produce_similar_results():
    """Both sigma point schemes should converge to similar estimates."""
    model = RigidBodyCVModel()
    cam = make_camera()
    R_cam = np.eye(4) * 4.0
    cam_sensor = ProjectiveBBoxSensor(cam, R_cam)

    x0 = make_state(pos=(2.0, 0.0, 10.0), vel=(1.0, 0.0, 0.0))
    P0 = np.eye(15) * 1.0
    Q = np.eye(15) * 0.01
    R_default = np.eye(6) * 0.01

    np.random.seed(77)
    true_pos = np.array([2.0, 0.0, 10.0])
    true_vel = np.array([1.0, 0.0, 0.0])
    dt = 0.1

    # Generate measurements once
    measurements = []
    pos_history = []
    for _ in range(30):
        true_pos = true_pos + true_vel * dt
        pos_history.append(true_pos.copy())
        z = cam_sensor.predict(make_state(pos=true_pos, quat=(1, 0, 0, 0)))
        z += np.random.randn(4) * 2.0
        measurements.append(z)

    errors = {}
    for scaled in [False, True]:
        ukf = UKF(model, Q, R_default, P0, x0.copy(), scaled_sigma_points=scaled)
        for z, gt in zip(measurements, pos_history):
            ukf.predict(dt)
            ukf.correct(z, sensor=cam_sensor)
        errors[scaled] = np.linalg.norm(ukf.state[:3] - pos_history[-1])

    # Both should converge
    assert errors[False] < 5.0, f"Symmetric failed: {errors[False]:.3f}"
    assert errors[True] < 5.0, f"Scaled failed: {errors[True]:.3f}"

    # Results should be in the same ballpark (within 10x of each other)
    ratio = max(errors[True], errors[False]) / max(min(errors[True], errors[False]), 1e-10)
    assert ratio < 10.0, f"Schemes diverged: sym={errors[False]:.3f}, scl={errors[True]:.3f}"
