// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::VectorXd;

namespace quak
{

/**
 * @brief Abstract manifold interface for UKF state/measurement spaces.
 *
 * Defines the retraction (boxPlus), inverse retraction (boxMinus), and
 * weighted mean operations needed by the UKF to operate on non-Euclidean
 * state spaces such as unit quaternions.
 */
class Manifold
{
public:
    virtual ~Manifold() = default;

    /// Apply a tangent-space perturbation: x ⊕ δ → x'.
    virtual VectorXd boxPlus(const VectorXd& x, const VectorXd& delta) = 0;

    /// Compute the tangent-space difference: x₁ ⊖ x₂ → δ.
    virtual VectorXd boxMinus(const VectorXd& x1, const VectorXd& x2) = 0;

    /// Weighted mean on the manifold.
    virtual VectorXd mean(const std::vector<VectorXd>& samples,
                          const std::vector<double>& weights) = 0;

    /// Dimension of the state representation (e.g., 4 for quaternion).
    virtual int stateSize() const = 0;

    /// Dimension of the tangent space (e.g., 3 for quaternion → rotation vector).
    virtual int tangentSize() const = 0;
};

/// Flat Euclidean manifold — boxPlus is addition, boxMinus is subtraction.
class Euclidean : public Manifold
{
    int mDim;

public:
    explicit Euclidean(int dim) : mDim(dim) {}

    VectorXd boxPlus(const VectorXd& x, const VectorXd& delta) override { return x + delta; }
    VectorXd boxMinus(const VectorXd& x1, const VectorXd& x2) override { return x1 - x2; }

    VectorXd mean(const std::vector<VectorXd>& samples, const std::vector<double>& weights) override
    {
        if (samples.empty())
            return VectorXd(mDim);

        VectorXd sum = VectorXd::Zero(mDim);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            sum += samples[i] * weights[i];
        }
        return sum;
    }

    int stateSize() const override { return mDim; }
    int tangentSize() const override { return mDim; }
};

/**
 * @brief Unit quaternion manifold (S³) with rotation vector tangent space.
 *
 * State: 4D [w, x, y, z] (Eigen convention).
 * Tangent: 3D rotation vector (angle × axis).
 *
 * Uses right-multiplication convention: q_new = q ⊗ exp(δ),
 * where exp(δ) converts a rotation vector to a unit quaternion.
 * Mean is computed via Markley's quaternion averaging (max eigenvector
 * of the weighted outer product matrix).
 */
class Quaternion : public Manifold
{
public:
    Quaternion() {}

    /// boxPlus: q ⊕ δ = q ⊗ exp(δ), right-multiplication convention.
    VectorXd boxPlus(const VectorXd& x, const VectorXd& delta) override
    {
        Quaterniond q(x(0), x(1), x(2), x(3));

        double angle = delta.norm();
        Quaterniond qDelta;
        if (angle < 1e-12)
        {
            qDelta = Quaterniond::Identity();
        }
        else
        {
            qDelta = Quaterniond(Eigen::AngleAxisd(angle, delta / angle));
        }

        Quaterniond qNew = (q * qDelta).normalized();

        VectorXd res(4);
        res << qNew.w(), qNew.x(), qNew.y(), qNew.z();
        return res;
    }

    /// boxMinus: q₁ ⊖ q₂ = log(q₂⁻¹ ⊗ q₁), consistent with right-multiplication.
    VectorXd boxMinus(const VectorXd& x1, const VectorXd& x2) override
    {
        Quaterniond q1(x1(0), x1(1), x1(2), x1(3));
        Quaterniond q2(x2(0), x2(1), x2(2), x2(3));

        Quaterniond qDiff = q2.inverse() * q1;

        // Ensure shortest-path rotation (w >= 0)
        if (qDiff.w() < 0)
        {
            qDiff.w() = -qDiff.w();
            qDiff.x() = -qDiff.x();
            qDiff.y() = -qDiff.y();
            qDiff.z() = -qDiff.z();
        }

        // Convert to rotation vector (angle × axis)
        double angle = 2.0 * std::acos(std::max(-1.0, std::min(1.0, qDiff.w())));
        if (angle < 1e-12)
        {
            return VectorXd::Zero(3);
        }

        double sinHalfAngle = std::sqrt(1.0 - qDiff.w() * qDiff.w());
        if (sinHalfAngle < 1e-12)
        {
            return VectorXd::Zero(3);
        }

        Eigen::Vector3d axis(qDiff.x(), qDiff.y(), qDiff.z());
        axis /= sinHalfAngle;

        return axis * angle;
    }

    /// Weighted mean via Markley's algorithm: max eigenvector of Σ wᵢ qᵢ qᵢᵀ.
    VectorXd mean(const std::vector<VectorXd>& samples, const std::vector<double>& weights) override
    {
        Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
        for (size_t i = 0; i < samples.size(); ++i)
        {
            VectorXd qVec = samples[i];
            M += weights[i] * (qVec * qVec.transpose());
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(M);
        VectorXd qMean = eigensolver.eigenvectors().col(3);

        // Ensure w > 0 for canonical representation
        if (qMean(0) < 0)
            qMean = -qMean;

        return qMean;
    }

    int stateSize() const override { return 4; }
    int tangentSize() const override { return 3; }
};

} // namespace quak
