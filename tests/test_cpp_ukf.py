# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import quaternion

from quak import UKF, Euclidean, Quaternion, RigidBodyPoseSensorModel


def test_euclidean_manifold():
    manifold = Euclidean(3)

    # 1. Box Plus
    x = np.array([1.0, 2.0, 3.0])
    delta = np.array([0.1, 0.2, 0.3])
    res = manifold.box_plus(x, delta)
    np.testing.assert_allclose(res, x + delta)

    # 2. Box Minus
    x2 = np.array([1.1, 2.2, 3.3])
    diff = manifold.box_minus(x2, x)
    np.testing.assert_allclose(diff, x2 - x)

    # 3. Mean
    samples = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])]
    weights = [0.5, 0.5]
    mean = manifold.mean(samples, weights)
    np.testing.assert_allclose(mean, [1.0, 1.0, 1.0])

    # Weighted Mean
    weights = [0.8, 0.2]
    mean_w = manifold.mean(samples, weights)
    np.testing.assert_allclose(mean_w, [0.4, 0.4, 0.4])


def test_quaternion_manifold():
    manifold = Quaternion()

    # 1. Box Plus
    q_identity = quaternion.one
    state = quaternion.as_float_array(q_identity)  # [1, 0, 0, 0]

    # Perturb around Z axis
    angle = np.pi / 2
    delta = np.array([0.0, 0.0, angle])

    new_state = manifold.box_plus(state, delta)

    # Expected
    expected_q = quaternion.from_rotation_vector(delta)
    expected = quaternion.as_float_array(expected_q)

    # Sign check
    if np.dot(new_state, expected) < 0:
        new_state = -new_state

    np.testing.assert_allclose(new_state, expected, atol=1e-6)

    # 2. Box Minus
    delta_calc = manifold.box_minus(new_state, state)
    np.testing.assert_allclose(delta_calc, delta, atol=1e-6)


def test_ukf_integration():
    dt = 0.1
    model = RigidBodyPoseSensorModel()

    # 16D state: [pos3, vel3, acc3, quat4, angvel3]
    x_init = np.zeros(16)
    x_init[9] = 1.0  # qw=1

    P_init = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001

    ukf = UKF(model, Q, R, P_init, x_init)

    # Set velocity in x
    x_init[3] = 1.0
    ukf.set_state(x_init, P_init)

    ukf.predict(dt)

    current_x = ukf.state
    np.testing.assert_allclose(current_x[0], 0.1, atol=1e-3)

    # Correct
    z_meas = np.array([0.12, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    ukf.correct(z_meas)

    current_x = ukf.x
    assert 0.1 < current_x[0] < 0.13


def test_ukf_control_input():
    dt = 0.1
    model = RigidBodyPoseSensorModel()
    x_init = np.zeros(16)
    x_init[9] = 1.0
    P_init = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001
    ukf = UKF(model, Q, R, P_init, x_init)

    control = np.array([1.0, 2.0])
    ukf.predict(dt, control)  # Should run without error


def test_ukf_getters():
    model = RigidBodyPoseSensorModel()
    x_init = np.zeros(16)
    x_init[9] = 1.0
    P_init = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001

    ukf = UKF(model, Q, R, P_init, x_init)

    np.testing.assert_allclose(ukf.process_noise, Q)
    np.testing.assert_allclose(ukf.measurement_noise, R)
    np.testing.assert_allclose(ukf.covariance, P_init)

    ukf.predict(0.1)
    z_meas = np.array([0.0] * 7)
    ukf.correct(z_meas)

    assert ukf.innovation_covariance is not None
    assert ukf.innovation_covariance.shape == (6, 6)
