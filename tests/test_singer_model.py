# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Singer maneuvering target model.

The Singer model treats acceleration as a first-order Markov process
(Ornstein-Uhlenbeck) that decays toward zero with time constant τ.

Tests verify:
1. Construction with default and custom τ
2. UKF integration (predict/correct cycle converges)
3. Acceleration decay: a → 0 exponentially when no measurements reinforce it
4. Position/velocity integration matches exact Singer formulas
5. IMM: Singer vs CV model switching on maneuvering trajectory
"""

import numpy as np
import pytest

from quak import (
    UKF,
    IMMEstimator,
    RigidBodyCAModel,
    RigidBodyCVModel,
    RigidBodySingerModel,
)


def make_state(pos=(0, 0, 0), vel=(0, 0, 0), acc=(0, 0, 0), quat=(1, 0, 0, 0), angvel=(0, 0, 0)):
    """Build a 16D state vector."""
    x = np.zeros(16)
    x[0:3] = pos
    x[3:6] = vel
    x[6:9] = acc
    x[9:13] = quat
    x[13:16] = angvel
    return x


def make_measurement(pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    """Build a 7D measurement [pos, quat]."""
    z = np.zeros(7)
    z[0:3] = pos
    z[3:7] = quat
    return z


# ── Construction ──────────────────────────────────────────────────────


def test_singer_construction_default_tau():
    model = RigidBodySingerModel()
    assert model.get_tau() == pytest.approx(2.0)


def test_singer_construction_custom_tau():
    model = RigidBodySingerModel(tau=5.0)
    assert model.get_tau() == pytest.approx(5.0)


def test_singer_set_tau():
    model = RigidBodySingerModel()
    model.set_tau(0.5)
    assert model.get_tau() == pytest.approx(0.5)


# ── Acceleration Decay ────────────────────────────────────────────────


def test_singer_acceleration_decays_exponentially():
    """With no control and no measurements, acceleration should decay toward zero."""
    model = RigidBodySingerModel(tau=1.0)
    dt = 0.1

    # Start with acceleration = [1, 0, 0]
    state = make_state(acc=(1.0, 0.0, 0.0))

    for step in range(50):
        state = model.process(state, np.array([]), dt)

    # After 5 seconds at τ=1, acceleration should be ~ exp(-5) ≈ 0.0067
    expected_decay = np.exp(-5.0)
    assert abs(state[6]) == pytest.approx(expected_decay, abs=1e-4)
    assert abs(state[7]) < 1e-10  # y-axis was zero
    assert abs(state[8]) < 1e-10  # z-axis was zero


def test_singer_acceleration_fully_preserved_with_large_tau():
    """With very large τ, Singer should behave like CA (constant acceleration)."""
    tau = 1000.0
    model_singer = RigidBodySingerModel(tau=tau)
    model_ca = RigidBodyCAModel()
    dt = 0.1

    state = make_state(pos=(1, 2, 3), vel=(0.5, 0, 0), acc=(0.1, 0, 0))

    state_singer = model_singer.process(state, np.array([]), dt)
    state_ca = model_ca.process(state, np.array([]), dt)

    # Position and velocity should be nearly identical
    np.testing.assert_allclose(state_singer[:6], state_ca[:6], atol=1e-4)
    # Acceleration should be close (α ≈ 1 for large τ)
    np.testing.assert_allclose(state_singer[6:9], state_ca[6:9], atol=1e-4)


# ── Exact Integration ────────────────────────────────────────────────


def test_singer_exact_integration():
    """Verify position and velocity match the exact Singer discrete formulas."""
    tau = 2.0
    dt = 0.1
    model = RigidBodySingerModel(tau=tau)

    p0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.5, 0.0, 0.0])
    a0 = np.array([0.2, 0.0, 0.0])

    state = make_state(pos=p0, vel=v0, acc=a0)
    state_next = model.process(state, np.array([]), dt)

    alpha = np.exp(-dt / tau)

    # Expected from Singer formulas
    p_expected = p0 + v0 * dt + a0 * (tau * dt - tau**2 * (1 - alpha))
    v_expected = v0 + a0 * tau * (1 - alpha)
    a_expected = a0 * alpha

    np.testing.assert_allclose(state_next[0:3], p_expected, atol=1e-12)
    np.testing.assert_allclose(state_next[3:6], v_expected, atol=1e-12)
    np.testing.assert_allclose(state_next[6:9], a_expected, atol=1e-12)


# ── UKF Integration ──────────────────────────────────────────────────


def test_singer_ukf_predict_correct_converges():
    """Singer model should converge in a standard UKF predict/correct loop."""
    model = RigidBodySingerModel(tau=2.0)

    x0 = make_state(quat=(1, 0, 0, 0))
    P0 = np.eye(15) * 0.1
    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.001

    ukf = UKF(model, Q, R, P0, x0)

    # Simulate a target at constant velocity (1 m/s along x)
    true_pos = np.array([0.0, 0.0, 0.0])
    dt = 0.1

    errors = []
    for step in range(100):
        true_pos[0] += 1.0 * dt

        ukf.predict(dt=dt)
        z = make_measurement(pos=true_pos, quat=(1, 0, 0, 0))
        z[:3] += np.random.RandomState(42 + step).randn(3) * 0.03
        ukf.correct(z)

        state = ukf.state
        errors.append(np.linalg.norm(state[:3] - true_pos))

    # Singer UKF should track well: final errors under 15cm
    final_error = np.mean(errors[-10:])
    assert final_error < 0.15, f"Final error too large: {final_error:.3f}"
    # Filter should remain stable (no divergence)
    assert np.all(np.isfinite(ukf.state)), "State contains NaN/Inf"


# ── IMM: Singer vs CV ────────────────────────────────────────────────


def test_imm_singer_vs_cv_maneuver_detection():
    """IMM with Singer + CV should detect maneuvers via model probability.

    Phase 1 (0-3s):   Constant velocity → CV should dominate
    Phase 2 (3-7s):   Strong acceleration → Singer should rise
    Phase 3 (7-10s):  Back to constant velocity → CV should recover
    """
    model_cv = RigidBodyCVModel()
    model_singer = RigidBodySingerModel(tau=1.0)

    x0 = make_state(quat=(1, 0, 0, 0))
    P0 = np.eye(15) * 0.1
    Q_cv = np.eye(15) * 0.01
    Q_singer = np.eye(15) * 0.05  # Singer gets higher Q (acceleration freedom)
    R = np.eye(6) * 0.01

    imm = IMMEstimator([model_cv, model_singer], [Q_cv, Q_singer], R, P0, x0)

    dt = 0.1
    rng = np.random.RandomState(123)

    probability_history = []
    true_pos = np.array([0.0, 0.0, 0.0])
    true_vel = np.array([1.0, 0.0, 0.0])

    for step in range(100):
        t = step * dt

        # Phase 1: constant velocity
        if t < 3.0:
            acceleration = np.array([0.0, 0.0, 0.0])
        # Phase 2: strong lateral acceleration (maneuver)
        elif t < 7.0:
            acceleration = np.array([0.0, 2.0, 0.0])
        # Phase 3: back to constant velocity
        else:
            acceleration = np.array([0.0, 0.0, 0.0])

        true_vel += acceleration * dt
        true_pos += true_vel * dt

        imm.predict(dt=dt)
        z = make_measurement(pos=true_pos, quat=(1, 0, 0, 0))
        z[:3] += rng.randn(3) * 0.1
        imm.correct(z)

        probabilities = imm.model_probabilities
        probability_history.append(probabilities.copy())

    probability_array = np.array(probability_history)  # (100, 2): [p_cv, p_singer]

    # Phase 1 (const vel): CV should dominate — check late phase 1
    phase1_cv = np.mean(probability_array[20:30, 0])
    phase1_singer = np.mean(probability_array[20:30, 1])

    # Phase 2 (maneuver): Singer should rise — check late phase 2
    phase2_cv = np.mean(probability_array[50:65, 0])
    phase2_singer = np.mean(probability_array[50:65, 1])

    # Phase 3 (recovery): CV should recover
    phase3_cv = np.mean(probability_array[85:95, 0])
    phase3_singer = np.mean(probability_array[85:95, 1])

    print("\n  Singer IMM model probabilities:")
    print(f"  Phase 1 (const vel):  CV={phase1_cv:.3f}  Singer={phase1_singer:.3f}")
    print(f"  Phase 2 (maneuver):   CV={phase2_cv:.3f}  Singer={phase2_singer:.3f}")
    print(f"  Phase 3 (recovery):   CV={phase3_cv:.3f}  Singer={phase3_singer:.3f}")

    # During maneuver, Singer should gain probability
    assert phase2_singer > phase1_singer, (
        f"Singer should increase during maneuver: "
        f"phase1={phase1_singer:.3f} vs phase2={phase2_singer:.3f}"
    )

    # After maneuver, CV should recover somewhat
    assert phase3_cv > phase2_cv, (
        f"CV should recover after maneuver: phase2={phase2_cv:.3f} vs phase3={phase3_cv:.3f}"
    )
