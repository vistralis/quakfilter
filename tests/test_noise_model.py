# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NoiseModel utility (Diagonal, Gaussian, Isotropic factories)."""

import numpy as np

from quak import NoiseModel

# ── Diagonal ─────────────────────────────────────────────────────────────


def test_noise_model_diagonal_creates_matrix():
    """Diagonal factory should produce a diagonal covariance matrix."""
    sigmas = np.array([0.1, 0.2, 0.3])
    R = NoiseModel.Diagonal(sigmas)

    assert R.shape == (3, 3)
    expected = np.diag([0.01, 0.04, 0.09])  # σ² on diagonal
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_noise_model_diagonal_single_element():
    """Diagonal factory works with a single-element vector."""
    R = NoiseModel.Diagonal(np.array([0.5]))
    assert R.shape == (1, 1)
    np.testing.assert_allclose(R[0, 0], 0.25, atol=1e-12)


def test_noise_model_diagonal_off_diagonal_zero():
    """Off-diagonal elements should be exactly zero."""
    R = NoiseModel.Diagonal(np.array([1.0, 2.0, 3.0, 4.0]))
    for i in range(4):
        for j in range(4):
            if i != j:
                assert R[i, j] == 0.0, f"R[{i},{j}] should be 0, got {R[i, j]}"


# ── Gaussian ─────────────────────────────────────────────────────────────


def test_noise_model_gaussian_preserves_matrix():
    """Gaussian factory should return the provided covariance matrix."""
    cov = np.array([[1.0, 0.1], [0.1, 2.0]])
    R = NoiseModel.Gaussian(cov)

    assert R.shape == (2, 2)
    np.testing.assert_allclose(R, cov, atol=1e-12)


def test_noise_model_gaussian_identity():
    """Gaussian with identity matrix."""
    cov = np.eye(5)
    R = NoiseModel.Gaussian(cov)
    np.testing.assert_allclose(R, cov, atol=1e-12)


# ── Isotropic ────────────────────────────────────────────────────────────


def test_noise_model_isotropic_creates_scaled_identity():
    """Isotropic factory should produce σ² · I."""
    R = NoiseModel.Isotropic(3, 0.5)

    assert R.shape == (3, 3)
    expected = np.eye(3) * 0.25  # 0.5² = 0.25
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_noise_model_isotropic_dimension_1():
    """Isotropic with dimension 1."""
    R = NoiseModel.Isotropic(1, 1.0)
    assert R.shape == (1, 1)
    np.testing.assert_allclose(R[0, 0], 1.0, atol=1e-12)


def test_noise_model_isotropic_large_dimension():
    """Isotropic with larger dimension to verify scaling."""
    R = NoiseModel.Isotropic(10, 0.1)
    assert R.shape == (10, 10)
    # Diagonal should be 0.01
    for i in range(10):
        assert np.isclose(R[i, i], 0.01), f"R[{i},{i}]={R[i, i]}"
    # Off-diagonal should be 0
    assert np.isclose(np.sum(np.abs(R)) - np.trace(np.abs(R)), 0.0)
