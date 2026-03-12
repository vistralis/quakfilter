# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for spatiotemporal state architecture.

TDD: Red → Green → Refactor for each test.
"""

import numpy as np
import pytest

import quak

# ── Shared helpers ──────────────────────────────────────────────────────

STATE_DIM = 16  # RigidBodyCVModel: pos(3) + vel(3) + ang_vel(3) + quat(4) + dims(3)
TANGENT_DIM = 15  # Tangent space (quaternion has 3-DOF, not 4)
MEAS_DIM = 6  # Pose measurement: pos(3) + rotation(3)


def make_state(pos=(0, 0, 0), vel=(0, 0, 0), ang_vel=(0, 0, 0), quat=(1, 0, 0, 0), dims=(1, 1, 1)):
    """Create a valid 16D state vector for RigidBodyCVModel."""
    return np.array([*pos, *vel, *ang_vel, *quat, *dims], dtype=np.float64)


def make_imm(**kwargs):
    """Create a simple single-model IMM estimator with correct dimensions."""
    model = quak.RigidBodyCVModel()
    Q = np.eye(TANGENT_DIM) * 0.01
    R = np.eye(MEAS_DIM) * 0.1
    P0 = np.eye(TANGENT_DIM) * 1.0
    x0 = kwargs.get("initial_state", make_state())
    return quak.IMMEstimator(
        models=[model],
        process_noises=[Q],
        measurement_noise=R,
        initial_covariance=P0,
        initial_state=x0,
    )


def make_pool(**kwargs):
    """Create a TrackPool with correct dimensions."""
    model = quak.RigidBodyCVModel()
    return quak.TrackPool(
        models=[model],
        process_noises=[np.eye(TANGENT_DIM) * 0.01],
        measurement_noise=np.eye(MEAS_DIM) * 0.1,
        initial_covariance=np.eye(TANGENT_DIM) * 1.0,
    )


# ─── Task 1: StateEstimate + TimeMode ───────────────────────────────────────


class TestStateEstimate:
    """StateEstimate is a read-only output container for {x, P, t, probabilities}."""

    def test_state_estimate_has_required_fields(self):
        """StateEstimate should expose trackId, state, covariance, timestamp, modelProbabilities."""
        est = quak.StateEstimate()
        assert hasattr(est, "track_id")
        assert hasattr(est, "state")
        assert hasattr(est, "covariance")
        assert hasattr(est, "timestamp")
        assert hasattr(est, "model_probabilities")

    def test_state_estimate_default_values(self):
        """Default-constructed StateEstimate should have empty/zero fields."""
        est = quak.StateEstimate()
        assert est.track_id == ""
        assert est.timestamp == 0.0


class TestTimeMode:
    """TimeMode enum controls whether TrackPool uses relative dt or absolute timestamps."""

    def test_time_mode_enum_values(self):
        """TimeMode should have Unset, Relative, and Absolute values."""
        assert hasattr(quak, "TimeMode")
        assert quak.TimeMode.Unset.value == 0
        assert quak.TimeMode.Relative.value == 1
        assert quak.TimeMode.Absolute.value == 2


# ─── Task 2: IMMEstimator::projectTo ────────────────────────────────────────


class TestProjectTo:
    """projectTo returns a StateEstimate without mutating the filter."""

    @pytest.fixture
    def imm(self):
        return make_imm()

    def test_project_to_returns_state_estimate(self, imm):
        """projectTo should return a StateEstimate."""
        result = imm.project_to(0.1)
        assert isinstance(result, quak.StateEstimate)

    def test_project_to_does_not_mutate_state(self, imm):
        """projectTo must not change the filter's internal state or covariance."""
        x_before = imm.x.copy()
        P_before = imm.P.copy()

        imm.project_to(0.1)

        np.testing.assert_array_equal(imm.x, x_before)
        np.testing.assert_array_equal(imm.P, P_before)

    def test_project_to_matches_predict(self, imm):
        """projectTo(dt) should produce the same x and P as predict(dt)."""
        # First get the projectTo result (non-mutating)
        projection = imm.project_to(0.1)

        # Now actually predict (mutating)
        imm.predict(0.1)

        np.testing.assert_allclose(projection.state, imm.x, atol=1e-12)
        np.testing.assert_allclose(projection.covariance, imm.P, atol=1e-12)


# ─── Task 3: TrackPool Mode Lock ────────────────────────────────────────────


class TestModeLock:
    """TrackPool locks into Relative or Absolute mode on first temporal call."""

    def test_initial_mode_is_unset(self):
        """New TrackPool should have TimeMode.Unset."""
        pool = make_pool()
        assert pool.time_mode == quak.TimeMode.Unset

    def test_predict_dt_locks_to_relative(self):
        """First predict(dt) should lock mode to Relative."""
        pool = make_pool()
        pool.add("track_a", make_state(), 0.0)
        pool.predict(0.033)
        assert pool.time_mode == quak.TimeMode.Relative

    def test_predict_to_locks_to_absolute(self):
        """First predictTo(timestamp) should lock mode to Absolute."""
        pool = make_pool()
        pool.add("track_a", make_state(), 100.0)
        pool.predict_to(100.033)
        assert pool.time_mode == quak.TimeMode.Absolute

    def test_relative_rejects_predict_to(self):
        """After locking Relative, predictTo should throw."""
        pool = make_pool()
        pool.add("track_a", make_state(), 0.0)
        pool.predict(0.033)
        with pytest.raises(RuntimeError, match="Relative"):
            pool.predict_to(100.0)

    def test_absolute_rejects_predict_dt(self):
        """After locking Absolute, predict(dt) should throw."""
        pool = make_pool()
        pool.add("track_a", make_state(), 100.0)
        pool.predict_to(100.033)
        with pytest.raises(RuntimeError, match="Absolute"):
            pool.predict(0.033)

    def test_set_time_mode_locks_mode(self):
        """setTimeMode should pre-declare the mode."""
        pool = make_pool()
        pool.set_time_mode(quak.TimeMode.Absolute)
        assert pool.time_mode == quak.TimeMode.Absolute

    def test_set_time_mode_rejects_change(self):
        """setTimeMode should reject changing a locked mode."""
        pool = make_pool()
        pool.add("track_a", make_state(), 0.0)
        pool.predict(0.033)  # locks to Relative
        with pytest.raises(RuntimeError, match="locked"):
            pool.set_time_mode(quak.TimeMode.Absolute)


# ─── Task 4: TrackPool::predictTo ───────────────────────────────────────────


class TestPredictTo:
    """predictTo computes per-track dt from timestamps."""

    def test_predict_to_computes_per_track_dt(self):
        """Tracks created at different times should get different dt values."""
        pool = make_pool()
        x0 = make_state()

        # Track A born at t=10, Track B born at t=12
        pool.add("A", x0.copy(), 10.0)
        x0_b = make_state(pos=(5, 0, 0))
        pool.add("B", x0_b, 12.0)

        # Predict to t=15: A gets dt=5, B gets dt=3
        pool.predict_to(15.0)

        states = pool.states
        # Both tracks should have been propagated (resulting in 2 rows)
        assert states.shape[0] == 2

    def test_predict_to_negative_dt_throws(self):
        """predictTo with a timestamp before the track's last update should throw."""
        pool = make_pool()
        pool.add("A", make_state(), 10.0)
        pool.predict_to(11.0)
        with pytest.raises(RuntimeError, match="[Nn]egative"):
            pool.predict_to(9.0)

    def test_predict_to_max_dt_guard(self):
        """predictTo with dt > max threshold should throw."""
        pool = make_pool()
        pool.add("A", make_state(), 0.0)
        with pytest.raises(RuntimeError, match="[Mm]ax"):
            pool.predict_to(100.0)  # dt=100 > default max of 10

    def test_predict_to_zero_dt_is_noop(self):
        """predictTo with same timestamp should not propagate."""
        pool = make_pool()
        pool.add("A", make_state(), 10.0)
        pool.predict_to(10.0)  # dt=0 — first call, stores t₀
        x_after = pool.states[0].copy()
        pool.predict_to(10.0)  # dt=0 — should be no-op
        np.testing.assert_array_equal(pool.states[0], x_after)


# ─── Task 5: getSnapshotAt ──────────────────────────────────────────────────


class TestGetSnapshotAt:
    """getSnapshotAt returns a temporally coherent snapshot without mutating."""

    @pytest.fixture
    def pool_with_tracks(self):
        pool = make_pool()
        pool.add("A", make_state(), 10.0)
        pool.add("B", make_state(), 12.0)
        return pool

    def test_snapshot_returns_state_estimates(self, pool_with_tracks):
        """getSnapshotAt should return a list of StateEstimate."""
        snapshot = pool_with_tracks.get_snapshot_at(15.0)
        assert len(snapshot) == 2
        assert all(isinstance(s, quak.StateEstimate) for s in snapshot)

    def test_snapshot_does_not_mutate(self, pool_with_tracks):
        """getSnapshotAt must not change filters or timestamps."""
        states_before = pool_with_tracks.states.copy()
        pool_with_tracks.get_snapshot_at(15.0)
        np.testing.assert_array_equal(pool_with_tracks.states, states_before)

    def test_snapshot_timestamps_are_coherent(self, pool_with_tracks):
        """All StateEstimates in a snapshot should have the target timestamp."""
        snapshot = pool_with_tracks.get_snapshot_at(15.0)
        for est in snapshot:
            assert est.timestamp == 15.0


# ─── Task 6: Accessor Extensions ────────────────────────────────────────────


class TestAccessors:
    """New per-track and batch accessors."""

    @pytest.fixture
    def pool_with_track(self):
        pool = make_pool()
        pool.add("track_A", make_state(), 42.0)
        return pool

    def test_get_estimate_returns_state_estimate(self, pool_with_track):
        """getEstimate should return a StateEstimate for a given trackId."""
        est = pool_with_track.get_estimate("track_A")
        assert isinstance(est, quak.StateEstimate)
        assert est.track_id == "track_A"
        assert est.timestamp == 42.0

    def test_get_estimates_returns_all(self, pool_with_track):
        """getEstimates should return one StateEstimate per track."""
        estimates = pool_with_track.get_estimates()
        assert len(estimates) == 1
        assert estimates[0].track_id == "track_A"

    def test_get_timestamps_returns_vector(self, pool_with_track):
        """getTimestamps should return per-track timestamps as a vector."""
        timestamps = pool_with_track.get_timestamps()
        assert len(timestamps) == 1
        assert timestamps[0] == 42.0
