# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pythonic wrapper for TrackPool (track pool)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from quak._core import TrackPool as _CoreBatchedIMM
from quak.time_conversions import to_duration_seconds, to_epoch_seconds


class TrackPool:
    """Track pool: manages multiple IMM estimators keyed by UUID.

    Tracks are stored in a contiguous vector, partitioned as
    [active | suspended]. ``predict()`` sweeps only active tracks
    via OpenMP; ``correct()`` updates an arbitrary subset by UUID.

    Parameters
    ----------
    models : list
        Motion models — same set used for every track's IMMEstimator.
    process_noises : list[NDArray]
        Per-model process noise covariances.
    measurement_noise : NDArray
        Shared measurement noise R.
    initial_covariance : NDArray
        Default P₀ used when adding tracks without explicit covariance.
    persistence_probability : float
        Markov self-transition probability (default 0.95).
    probability_floor : float
        Minimum model probability floor (default 0.01).
    """

    def __init__(
        self,
        models: list,
        process_noises: list[NDArray],
        measurement_noise: NDArray,
        initial_covariance: NDArray,
        *,
        persistence_probability: float = 0.95,
        probability_floor: float = 0.01,
    ):
        self._pool = _CoreBatchedIMM(
            models,
            process_noises,
            measurement_noise,
            initial_covariance,
            True,
            persistence_probability,
            probability_floor,
        )

    def set_time_mode(self, mode) -> None:
        """Pre-declare the time mode (optional — auto-detected on first predict).

        Args:
            mode: TimeMode.Relative or TimeMode.Absolute.
        """
        self._pool.set_time_mode(mode)

    def add(
        self,
        track_id: str,
        state: NDArray,
        time: float | datetime | np.datetime64,
        covariance: NDArray | None = None,
    ) -> None:
        """Add a new active track with an initial timestamp.

        Parameters
        ----------
        track_id : str
            Unique UUID for this track.
        state : NDArray
            Initial state vector.
        time : float or datetime or numpy.datetime64
            Timestamp of the initial state (epoch seconds).
        covariance : NDArray, optional
            Initial covariance. Uses the pool default if omitted.
        """
        epoch = to_epoch_seconds(time)
        if covariance is not None:
            self._pool.add_with_covariance(track_id, state, epoch, covariance)
        else:
            self._pool.add(track_id, state, epoch)

    def remove(self, track_id: str) -> None:
        """Remove a track by UUID."""
        self._pool.remove(track_id)

    def suspend(self, track_id: str) -> None:
        """Move a track to the suspended partition (skipped by predict)."""
        self._pool.suspend(track_id)

    def resume(self, track_id: str) -> None:
        """Move a suspended track back to active."""
        self._pool.resume(track_id)

    # ─── Batch Operations ────────────────────────────────────────

    def predict(self, dt: float | timedelta | np.timedelta64) -> None:
        """Predict all active tracks forward by dt seconds (Relative mode)."""
        self._pool.predict(to_duration_seconds(dt))

    def predict_to(self, timestamp: float | datetime | np.datetime64) -> None:
        """Predict all active tracks to a target timestamp (Absolute mode).

        Computes per-track dt from stored timestamps.

        Args:
            timestamp: Target time to propagate all active tracks to.
        """
        self._pool.predict_to(to_epoch_seconds(timestamp))

    def get_snapshot_at(self, timestamp: float | datetime | np.datetime64) -> list:
        """Non-mutating snapshot at a target timestamp (Absolute mode).

        Returns a list of StateEstimate without modifying any filter state
        or timestamps. Uses projectTo internally for shadow predictions.

        Args:
            timestamp: Target time for the snapshot.

        Returns:
            List of StateEstimate, one per active track.
        """
        return self._pool.get_snapshot_at(to_epoch_seconds(timestamp))

    def correct(
        self,
        track_ids: list[str],
        measurements: NDArray,
        *,
        sensor=None,
    ) -> None:
        """Correct a subset of tracks.

        Parameters
        ----------
        track_ids : list[str]
            UUIDs of the tracks to update.
        measurements : NDArray
            (K, MeasDim) — one row per track.
        sensor : MeasurementModel, optional
            Standalone sensor for multi-sensor correction.
        """
        # C++ expects (MeasDim, K) column-major
        measurements_t = np.ascontiguousarray(measurements.T)
        if sensor is not None:
            self._pool.correct_with_sensor(track_ids, measurements_t, sensor)
        else:
            self._pool.correct(track_ids, measurements_t)

    # ─── Properties ──────────────────────────────────────────────

    @property
    def states(self) -> NDArray:
        """All track states as (N, StateDim) array."""
        raw = self._pool.get_states()
        return raw.T if raw.size > 0 else raw

    @property
    def model_probabilities(self) -> NDArray:
        """All model probabilities as (N, NumModels) array."""
        raw = self._pool.get_model_probabilities()
        return raw.T if raw.size > 0 else raw

    @property
    def track_ids(self) -> list[str]:
        """All track UUIDs (order matches states rows)."""
        return self._pool.track_ids()

    @property
    def active_track_ids(self) -> list[str]:
        """Active track UUIDs only."""
        return self._pool.active_track_ids()

    @property
    def active_count(self) -> int:
        """Number of active tracks."""
        return self._pool.active_count()

    @property
    def time_mode(self):
        """Current TimeMode (Unset, Relative, or Absolute)."""
        return self._pool.time_mode

    def get_estimate(self, track_id: str):
        """Get a StateEstimate for a single track by UUID.

        Args:
            track_id: UUID of the track.

        Returns:
            StateEstimate with state, covariance, model probabilities, and timestamp.
        """
        return self._pool.get_estimate(track_id)

    def get_estimates(self) -> list:
        """Get StateEstimates for all active tracks.

        Returns:
            List of StateEstimate, one per active track.
        """
        return self._pool.get_estimates()

    def get_timestamps(self) -> list[float]:
        """Get per-track timestamps as a list of epoch seconds.

        All tracks have timestamps assigned at creation time.

        Returns:
            List of timestamps as epoch seconds (float).
        """
        return self._pool.get_timestamps()

    def __len__(self) -> int:
        return self._pool.size()

    def __contains__(self, track_id: str) -> bool:
        return self._pool.contains(track_id)
