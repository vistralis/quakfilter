# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pythonic wrapper around the C++ UKF implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quak import _core

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UKF:
    """Unscented Kalman Filter with manifold-aware state estimation.

    Args:
        model: System model defining process and measurement dynamics.
        process_noise: Process noise covariance Q (tangent_dim × tangent_dim).
        measurement_noise: Measurement noise covariance R (meas_dim × meas_dim).
        initial_covariance: Initial error covariance P₀ (tangent_dim × tangent_dim).
        initial_state: Initial state vector x₀ (state_dim,).
        svd_decomposition: Use SVD for the matrix square root. If False,
            uses LDLT (faster but less robust). Default: True.
        scaled_sigma_points: Use scaled (2n+1) sigma points instead of
            symmetric (2n). Default: True.
    """

    def __init__(
        self,
        model: _core.UKFModel,
        process_noise: NDArray[np.float64],
        measurement_noise: NDArray[np.float64],
        initial_covariance: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        svd_decomposition: bool = True,
        scaled_sigma_points: bool = True,
    ) -> None:
        self._filter = _core.UKF(
            model,
            np.asarray(process_noise, dtype=np.float64),
            np.asarray(measurement_noise, dtype=np.float64),
            np.asarray(initial_covariance, dtype=np.float64),
            np.asarray(initial_state, dtype=np.float64),
            svd_decomposition,
            scaled_sigma_points,
        )

    def predict(self, dt: float, control: NDArray[np.float64] | None = None):
        """Propagate the state forward by dt seconds.

        Args:
            dt: Time step in seconds.
            control: Optional control input vector.

        Returns:
            PredictionResult with ``z_mean`` and ``S_zz`` attributes.
        """
        if control is not None:
            return self._filter.predict(dt, np.asarray(control, dtype=np.float64))
        return self._filter.predict(dt)

    def correct(
        self,
        measurement: NDArray[np.float64],
        sensor: _core.MeasurementModel | None = None,
    ) -> UKF:
        """Update the state with a measurement.

        Args:
            measurement: Observation vector.
            sensor: Optional standalone measurement model for multi-sensor fusion.

        Returns:
            self, for method chaining.
        """
        z = np.asarray(measurement, dtype=np.float64)
        if sensor is not None:
            self._filter.correct_with_sensor(z, sensor)
        else:
            self._filter.correct(z)
        return self

    def set_state(
        self,
        state: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> UKF:
        """Override the current state and covariance.

        Returns:
            self, for method chaining.
        """
        self._filter.set_state_and_error_covariance(
            np.asarray(state, dtype=np.float64),
            np.asarray(covariance, dtype=np.float64),
        )
        return self

    @property
    def state(self) -> NDArray[np.float64]:
        """Current state estimate (state_dim,)."""
        return self._filter.get_state()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """State error covariance P (tangent_dim × tangent_dim)."""
        return self._filter.get_state_error_covariance()

    @property
    def innovation_covariance(self) -> NDArray[np.float64] | None:
        """Innovation covariance S from the last correct() step."""
        return self._filter.get_innovation_covariance()

    @property
    def process_noise(self) -> NDArray[np.float64]:
        """Process noise covariance Q."""
        return self._filter.get_process_noise_covariance()

    @property
    def measurement_noise(self) -> NDArray[np.float64]:
        """Measurement noise covariance R."""
        return self._filter.get_measurement_noise_covariance()

    # Short aliases
    x = state
    P = covariance

    def __repr__(self) -> str:
        pos = self.state[:3]
        return f"UKF(pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])"
