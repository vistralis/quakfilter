# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from quak import (
    UKF,
    RigidBodyCAModel,
    SystemModel,
)


def quat_from_yaw(yaw):
    """Create a quaternion [w, x, y, z] from a yaw angle."""
    hz = yaw / 2.0
    return np.array([np.cos(hz), 0.0, 0.0, np.sin(hz)])


def test_ca_model_construction():
    """Verify RigidBodyCAModel can be instantiated."""
    model = RigidBodyCAModel()
    assert isinstance(model, SystemModel)


def test_ca_model_ukf_integration():
    """Verify RigidBodyCAModel works inside a standard UKF with acceleration."""
    model = RigidBodyCAModel()

    # 16D State: [x, y, z, vx, vy, vz, ax, ay, az, qw, qx, qy, qz, wx, wy, wz]
    x0 = np.zeros(16)
    x0[9] = 1.0  # qw = 1.0

    # Moving at 10m/s in x, accelerating at 2m/s^2 in x
    x0[3] = 10.0  # vx
    x0[6] = 2.0  # ax

    # Tangent space is 15D (Euclidean(9) + tangent of Quat(3) + Euclidean(3))
    P0 = np.eye(15) * 0.1

    process_noise = np.eye(15) * 0.01
    meas_noise = np.eye(6) * 0.1

    # UKF signature: (model, process_noise, measurement_noise, initial_cov, initial_state)
    ukf = UKF(model, process_noise, meas_noise, P0, x0)

    # Predict step (dt = 1.0s)
    u = np.zeros(1)  # dummy control
    ukf.predict(1.0, u)

    state_pred = ukf.state
    assert np.isclose(state_pred[0], 11.0), f"Expected x=11.0, got {state_pred[0]}"
    assert np.isclose(state_pred[3], 12.0), f"Expected vx=12.0, got {state_pred[3]}"
    assert np.isclose(state_pred[6], 2.0), f"Expected ax=2.0, got {state_pred[6]}"

    # Correct step (Measurement is Pos(3) + Ori(4))
    z = np.zeros(7)
    z[0] = 11.0  # x
    z[3] = 1.0  # qw

    ukf.correct(z)

    state_corr = ukf.state
    assert np.isclose(state_corr[0], 11.0), f"Expected x=11.0 after correction, got {state_corr[0]}"

    # Check that covariance remains SPD
    P_corr = ukf.covariance
    assert np.all(np.linalg.eigvals(P_corr) > 0), "Covariance matrix is not positive definite"
    print("RigidBodyCAModel integrated correctly into UKF.")
