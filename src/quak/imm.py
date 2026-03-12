# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pythonic wrapper around the C++ IMMEstimator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quak import _core

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IMMEstimator:
    """Interacting Multiple Model estimator for maneuvering target tracking.

    Runs multiple UKF sub-filters and fuses their estimates using
    Markov switching probabilities.

    **Explicit mode** — provide transition matrix directly::

        imm = IMMEstimator(
            models=[cv, ctrv],
            process_noises=[Q_cv, Q_ctrv],
            measurement_noise=R,
            initial_covariance=P0,
            initial_state=x0,
            transition_matrix=T,
            initial_probabilities=np.array([0.5, 0.5]),
        )

    **Auto mode** — uniform transition matrix from ``persistence_probability``::

        imm = IMMEstimator(
            models=[cv, ctrv],
            process_noises=[Q_cv, Q_ctrv],
            measurement_noise=R,
            initial_covariance=P0,
            initial_state=x0,
            persistence_probability=0.95,
        )

    Args:
        models: List of system models (one per sub-filter).
        process_noises: Per-model process noise covariances.
        measurement_noise: Shared measurement noise covariance R.
        initial_covariance: Initial error covariance P₀.
        initial_state: Initial state vector x₀.
        transition_matrix: Markov transition matrix (n × n). Mutually exclusive with persistence_probability.
        initial_probabilities: Initial model probabilities. Defaults to uniform.
        svd_decomposition: Use SVD for the matrix square root (default: True).
        persistence_probability: Probability of staying in same model. Mutually exclusive with transition_matrix.
        probability_floor: Minimum model probability to prevent collapse (default: 0.01).
    """

    def __init__(
        self,
        models: list[_core.UKFModel],
        process_noises: list[NDArray[np.float64]],
        measurement_noise: NDArray[np.float64],
        initial_covariance: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        *,
        transition_matrix: NDArray[np.float64] | None = None,
        initial_probabilities: NDArray[np.float64] | None = None,
        svd_decomposition: bool = True,
        persistence_probability: float | None = None,
        probability_floor: float = 0.01,
    ) -> None:
        R = np.asarray(measurement_noise, dtype=np.float64)
        P0 = np.asarray(initial_covariance, dtype=np.float64)
        x0 = np.asarray(initial_state, dtype=np.float64)
        Qs = [np.asarray(q, dtype=np.float64) for q in process_noises]

        if transition_matrix is not None and persistence_probability is not None:
            msg = "Specify either transition_matrix or persistence_probability, not both"
            raise ValueError(msg)

        if transition_matrix is not None:
            T = np.asarray(transition_matrix, dtype=np.float64)
            if initial_probabilities is None:
                initial_probabilities = np.ones(len(models)) / len(models)
            mu0 = np.asarray(initial_probabilities, dtype=np.float64)
            self._estimator = _core.IMMEstimator(
                models, T, Qs, R, P0, x0, mu0, svd_decomposition, probability_floor
            )
        else:
            probability = persistence_probability if persistence_probability is not None else 0.95
            self._estimator = _core.IMMEstimator(
                models, Qs, R, P0, x0, svd_decomposition, probability, probability_floor
            )

    def predict(self, dt: float, control: NDArray[np.float64] | None = None):
        """Propagate all sub-filters forward by dt seconds.

        Returns:
            PredictionResult with ``z_mean`` and ``S_zz`` attributes.
        """
        if control is not None:
            return self._estimator.predict(dt, np.asarray(control, dtype=np.float64))
        return self._estimator.predict(dt)

    def project_to(self, dt: float) -> _core.StateEstimate:
        """Non-mutating shadow predict — returns projected state without modifying the filter.

        Creates temporary copies of internal UKF filters, runs the full IMM predict
        pipeline on them, and returns a StateEstimate. The original filter state
        is untouched.

        Args:
            dt: Time step to project forward.

        Returns:
            StateEstimate with projected x, P, and model probabilities.
        """
        return self._estimator.project_to(dt)

    def correct(
        self,
        measurement: NDArray[np.float64],
        sensor: _core.MeasurementModel | None = None,
    ) -> IMMEstimator:
        """Update all sub-filters with a measurement and fuse probabilities.

        Returns:
            self, for method chaining.
        """
        z = np.asarray(measurement, dtype=np.float64)
        if sensor is not None:
            self._estimator.correct_with_sensor(z, sensor)
        else:
            self._estimator.correct(z)
        return self

    @property
    def state(self) -> NDArray[np.float64]:
        """Fused state estimate (state_dim,)."""
        return self._estimator.get_state()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Fused error covariance P (tangent_dim × tangent_dim)."""
        return self._estimator.get_state_error_covariance()

    @property
    def model_probabilities(self) -> NDArray[np.float64]:
        """Current model probabilities (n_models,)."""
        return self._estimator.get_model_probabilities()

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Markov transition matrix (n_models × n_models)."""
        return self._estimator.get_transition_matrix()

    @property
    def num_models(self) -> int:
        """Number of sub-models."""
        return self._estimator.get_num_models()

    @property
    def probability_floor(self) -> float:
        """Minimum model probability to prevent collapse."""
        return self._estimator.get_prob_floor()

    @probability_floor.setter
    def probability_floor(self, value: float) -> None:
        """Set minimum model probability to prevent any single model from collapsing to zero."""
        self._estimator.set_prob_floor(value)

    def model_state(self, index: int) -> NDArray[np.float64]:
        """State estimate of the i-th sub-filter."""
        return self._estimator.get_model_state(index)

    def model_covariance(self, index: int) -> NDArray[np.float64]:
        """Error covariance of the i-th sub-filter."""
        return self._estimator.get_model_covariance(index)

    # Short aliases
    x = state
    P = covariance
    mu = model_probabilities

    def __repr__(self) -> str:
        probabilities = self.model_probabilities
        prob_str = ", ".join(f"{p:.2f}" for p in probabilities)
        return f"IMMEstimator(models={self.num_models}, mu=[{prob_str}])"
