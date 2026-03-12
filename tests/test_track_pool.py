# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrackPool (track pool)."""

import sys

import numpy as np

sys.path.insert(0, "build")
from quak import (
    RigidBodyCTRVModel,
    RigidBodyPoseSensorModel,
    TrackPool,
)


def make_state(pos=None, vel=None):
    """Build a 16D state vector with identity quaternion."""
    x = np.zeros(16)
    x[9] = 1.0  # qw = 1
    if pos is not None:
        x[0:3] = pos
    if vel is not None:
        x[3:6] = vel
    return x


def make_config():
    """Shared models and noise for all tests."""
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()
    models = [model_cv, model_ctrv]

    Q = np.eye(15) * 0.01
    R = np.eye(6) * 0.01
    P0 = np.eye(15) * 0.1
    return models, [Q, Q], R, P0


# ─── Construction ────────────────────────────────────────────────────


def test_empty_pool():
    """Empty pool has size 0."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)
    assert len(pool) == 0
    assert pool.active_count == 0
    assert pool.track_ids == []


def test_add_tracks():
    """Adding tracks grows the pool."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("track-a", make_state(pos=[1, 0, 0]), 0.0)
    pool.add("track-b", make_state(pos=[0, 1, 0]), 0.0)
    pool.add("track-c", make_state(pos=[0, 0, 1]), 0.0)

    assert len(pool) == 3
    assert pool.active_count == 3
    assert "track-a" in pool
    assert "track-b" in pool
    assert "track-c" in pool


def test_remove_track():
    """Removing a track shrinks the pool."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("track-a", make_state(pos=[1, 0, 0]), 0.0)
    pool.add("track-b", make_state(pos=[0, 1, 0]), 0.0)

    pool.remove("track-a")
    assert len(pool) == 1
    assert "track-a" not in pool
    assert "track-b" in pool


# ─── Predict / Correct ──────────────────────────────────────────────


def test_predict_correct_cycle():
    """Full predict → correct cycle runs without errors."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    # Add 5 tracks at different positions
    for i in range(5):
        pool.add(f"track-{i}", make_state(pos=[i * 2, 0, 0], vel=[1, 0, 0]), 0.0)

    for step in range(10):
        pool.predict(0.1)

        # Correct only a subset (tracks 0, 2, 4)
        ids = ["track-0", "track-2", "track-4"]
        z = np.array([[1, 0, 0, 1, 0, 0, 0]] * 3, dtype=np.float64)
        pool.correct(ids, z)

    # All states should be finite
    states = pool.states
    assert states.shape == (5, 16)
    assert np.all(np.isfinite(states))

    # Model probabilities should be valid
    probabilities = pool.model_probabilities
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_selective_correct():
    """Correcting a subset does not affect uncorrected tracks."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("A", make_state(pos=[10, 0, 0]), 0.0)
    pool.add("B", make_state(pos=[20, 0, 0]), 0.0)

    pool.predict(0.1)

    # Snapshot track B state before any correction
    states_before = pool.states.copy()

    # Find B's row index from track_ids
    ids = pool.track_ids
    b_idx = ids.index("B")
    b_state_before = states_before[b_idx].copy()

    # Correct only A
    z = np.array([[10, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
    pool.correct(["A"], z)

    # B's state should be unchanged
    states_after = pool.states
    b_state_after = states_after[ids.index("B")]
    np.testing.assert_array_equal(b_state_before, b_state_after)


# ─── Suspend / Resume ───────────────────────────────────────────────


def test_suspend_resume():
    """Suspended tracks are excluded from predict."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("active-a", make_state(pos=[1, 0, 0], vel=[1, 0, 0]), 0.0)
    pool.add("suspend-b", make_state(pos=[2, 0, 0], vel=[1, 0, 0]), 0.0)

    assert pool.active_count == 2

    pool.suspend("suspend-b")
    assert pool.active_count == 1
    assert len(pool) == 2  # still in pool, just suspended

    # Predict only moves active tracks
    pool.predict(0.1)

    pool.resume("suspend-b")
    assert pool.active_count == 2


# ─── Equivalence ────────────────────────────────────────────────────


def test_equivalence_with_single_imm():
    """A pool with one track should produce identical results to standalone IMMEstimator."""
    from quak import IMMEstimator

    models, Qs, R, P0 = make_config()
    x0 = make_state(pos=[1, 2, 3], vel=[1, 0, 0])

    # Standalone
    imm = IMMEstimator(models, Qs, R, P0, x0)

    # Pool with one track
    pool = TrackPool(models, Qs, R, P0)
    pool.add("solo", x0.copy(), 0.0)

    # Run same predict/correct sequence
    z = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float64)
    for _ in range(5):
        imm.predict(0.1)
        pool.predict(0.1)

        imm.correct(z)
        pool.correct(["solo"], z.reshape(1, -1))

    # States should match exactly
    np.testing.assert_allclose(pool.states[0], imm.state, atol=1e-12)
    np.testing.assert_allclose(pool.model_probabilities[0], imm.model_probabilities, atol=1e-12)


# ─── Bulk Accessors ─────────────────────────────────────────────────


def test_states_shape():
    """States array has correct shape (N, StateDim)."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    for i in range(10):
        pool.add(f"t-{i}", make_state(pos=[i, 0, 0]), 0.0)

    states = pool.states
    assert states.shape == (10, 16)

    probabilities = pool.model_probabilities
    assert probabilities.shape == (10, 2)


# ─── Uncovered paths ────────────────────────────────────────────────


def test_add_with_custom_covariance():
    """Adding a track with an explicit initial covariance."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    custom_P = np.eye(15) * 0.5
    pool.add("cov-track", make_state(pos=[1, 0, 0]), 0.0, covariance=custom_P)

    assert len(pool) == 1
    assert "cov-track" in pool

    # Covariance should be retrievable and finite
    est = pool.get_estimate("cov-track")
    assert np.all(np.isfinite(est.covariance))


def test_correct_with_sensor():
    """Correct with an explicit sensor model."""
    from quak import PoseSensor

    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("s-track", make_state(pos=[1, 0, 0], vel=[1, 0, 0]), 0.0)
    pool.predict(0.1)

    sensor = PoseSensor(R)
    z = np.array([[1, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
    pool.correct(["s-track"], z, sensor=sensor)

    states = pool.states
    assert np.all(np.isfinite(states))


def test_active_track_ids():
    """Active track IDs list excludes suspended tracks."""
    models, Qs, R, P0 = make_config()
    pool = TrackPool(models, Qs, R, P0)

    pool.add("a", make_state(), 0.0)
    pool.add("b", make_state(), 0.0)
    pool.add("c", make_state(), 0.0)

    assert set(pool.active_track_ids) == {"a", "b", "c"}

    pool.suspend("b")
    assert set(pool.active_track_ids) == {"a", "c"}
    assert len(pool) == 3  # total unchanged
