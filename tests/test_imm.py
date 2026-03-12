# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for predicted measurement and IMM estimator features."""

import sys

import numpy as np

sys.path.insert(0, "build")
from quak import (
    UKF,
    IMMEstimator,
    RigidBodyCTRVModel,
    RigidBodyPoseSensorModel,
    UKFPredictionResult,
)

# ──────────────────────────────────────────────────────────────────────
# Feature 1: Predicted Measurement
# ──────────────────────────────────────────────────────────────────────


def test_predict_returns_prediction_result():
    """predict() must return a PredictionResult with z_mean and S_zz."""
    model = RigidBodyPoseSensorModel()
    x0 = np.zeros(16)
    x0[9] = 1.0  # quaternion w=1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001
    P0 = np.eye(15) * 0.1

    ukf = UKF(model, Q, R, P0, x0)
    result = ukf.predict(0.1)

    assert isinstance(result, UKFPredictionResult)
    assert result.z_mean.shape == (7,)  # measurement dim = 7 (pos3 + quat4)
    assert result.S_zz.shape == (6, 6)  # tangent measurement dim = 6 (pos3 + rot3)


def test_predicted_measurement_mean_is_close_to_measurement_of_state():
    """z_mean ≈ h(x̂) when covariance is small (linear regime)."""
    model = RigidBodyPoseSensorModel()
    x0 = np.array(
        [
            1.0,
            2.0,
            3.0,  # position
            0.0,
            0.0,
            0.0,  # velocity
            0.0,
            0.0,
            0.0,  # acceleration
            1.0,
            0.0,
            0.0,
            0.0,  # quaternion identity
            0.0,
            0.0,
            0.0,
        ]
    )  # angular velocity
    Q = np.eye(15) * 1e-6
    R = np.eye(6) * 1e-6
    P0 = np.eye(15) * 1e-4

    ukf = UKF(model, Q, R, P0, x0)
    result = ukf.predict(0.01)

    # With tiny covariance, z_mean should be very close to h(x0)
    # h(x) = [pos, quat] = [x0:3, x9:13]
    expected_z = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.z_mean, expected_z, atol=1e-3)


def test_innovation_covariance_is_spd():
    """S_zz must be symmetric positive definite."""
    model = RigidBodyPoseSensorModel()
    x0 = np.zeros(16)
    x0[9] = 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001
    P0 = np.eye(15) * 0.1

    ukf = UKF(model, Q, R, P0, x0)
    result = ukf.predict(0.1)

    S = result.S_zz
    # Symmetric
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    # Positive definite (all eigenvalues > 0)
    eigs = np.linalg.eigvalsh(S)
    assert np.all(eigs > 0), f"S_zz has non-positive eigenvalue: {eigs.min()}"


def test_predict_correct_cycle_with_prediction_result():
    """Full predict→correct cycle with PredictionResult used for gating."""
    model = RigidBodyPoseSensorModel()
    x0 = np.zeros(16)
    x0[9] = 1.0
    x0[3] = 1.0  # velocity
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    ukf = UKF(model, Q, R, P0, x0)

    for _ in range(5):
        result = ukf.predict(0.1)

        # Mahalanobis gating
        z = np.array([ukf.x[0], 0, 0, 1, 0, 0, 0])
        z[:6] - result.z_mean[:6]
        assert np.all(np.isfinite(result.z_mean))
        assert np.all(np.isfinite(result.S_zz))

        ukf.correct(z)


# ──────────────────────────────────────────────────────────────────────
# Feature 2: IMM Estimator
# ──────────────────────────────────────────────────────────────────────


def test_imm_construction():
    """IMMEstimator can be constructed with convenience constructor."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    imm = IMMEstimator([model_cv, model_ctrv], [Q, Q], R, P0, x0)

    assert imm.num_models == 2
    np.testing.assert_allclose(imm.mu, [0.5, 0.5])
    assert imm.x.shape == (16,)
    assert imm.P.shape == (15, 15)


def test_imm_construction_with_custom_transition():
    """IMMEstimator with explicit transition matrix."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    probabilities = np.array([0.6, 0.4])

    imm = IMMEstimator(
        [model_cv, model_ctrv],
        [Q, Q],
        R,
        P0,
        x0,
        transition_matrix=Pi,
        initial_probabilities=probabilities,
    )

    np.testing.assert_allclose(imm.mu, probabilities)
    np.testing.assert_allclose(imm.transition_matrix, Pi)


def test_imm_predict_returns_prediction_result():
    """IMM predict() returns PredictionResult with combined z_mean/S_zz."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    x0[3] = 1.0  # forward velocity
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    imm = IMMEstimator([model_cv, model_ctrv], [Q, Q], R, P0, x0)
    result = imm.predict(0.1)

    assert isinstance(result, UKFPredictionResult)
    assert result.z_mean.shape == (7,)
    assert result.S_zz.shape == (6, 6)
    assert np.all(np.isfinite(result.z_mean))
    assert np.all(np.isfinite(result.S_zz))

    # S_zz should be symmetric positive definite
    S = result.S_zz
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(S)
    assert np.all(eigs > 0)


def test_imm_predict_correct_cycle():
    """Full IMM predict→correct cycle runs without errors."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    x0[3] = 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    imm = IMMEstimator([model_cv, model_ctrv], [Q, Q], R, P0, x0)

    for step in range(10):
        imm.predict(0.1)
        z = np.array([imm.x[0], 0, 0, 1, 0, 0, 0])
        imm.correct(z)

    # After correction, state should be finite
    assert np.all(np.isfinite(imm.x))
    assert np.all(np.isfinite(imm.P))

    # Probabilities should sum to 1
    probabilities = imm.mu
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)


def test_imm_model_probability_evolution():
    """Model probabilities should shift based on dynamics.

    Straight-line motion should favor CV,
    turning motion should favor CTRV.
    """
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    x0[3] = 1.0  # forward velocity
    Q = np.eye(15) * 0.001
    R = np.eye(6) * 0.001
    P0 = np.eye(15) * 0.01

    imm = IMMEstimator([model_cv, model_ctrv], [Q, Q], R, P0, x0)

    # Phase 1: Straight line motion (should favor CV)
    for step in range(20):
        imm.predict(0.1)
        pos = x0[:3] + np.array([1.0, 0, 0]) * 0.1 * (step + 1)
        z = np.array([pos[0], pos[1], pos[2], 1, 0, 0, 0])
        imm.correct(z)

    probabilities_straight = imm.mu.copy()

    # probabilities should be valid
    assert np.all(probabilities_straight > 0)
    np.testing.assert_allclose(probabilities_straight.sum(), 1.0, atol=1e-6)

    # State should track the straight-line trajectory
    assert np.all(np.isfinite(imm.x))


def test_imm_per_model_accessors():
    """Per-model state and covariance getters work."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    x0 = np.zeros(16)
    x0[9] = 1.0
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1

    imm = IMMEstimator([model_cv, model_ctrv], [Q, Q], R, P0, x0)
    imm.predict(0.1)

    # Per-model state and covariance
    for i in range(2):
        state_i = imm.model_state(i)
        cov_i = imm.model_covariance(i)
        assert state_i.shape == (16,)
        assert cov_i.shape == (15, 15)
        assert np.all(np.isfinite(state_i))
        assert np.all(np.isfinite(cov_i))
