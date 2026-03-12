// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/manifolds.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace quak
{

/**
 * @brief Abstract interface for a combined process + measurement model.
 *
 * Defines the state transition (process), measurement function, and
 * manifold operations (mean, residual, correction) that the UKF uses
 * to propagate and update state estimates.
 */
class UKFModel
{
public:
    virtual ~UKFModel() = default;

    /// Propagate state forward by dt: x_{t+1} = f(x_t, u, dt).
    virtual VectorXd process(const VectorXd& state, const VectorXd& control, double dt) = 0;

    /// Map state to predicted measurement: z = h(x).
    virtual VectorXd measurement(const VectorXd& state) = 0;

    // ── Manifold Operations ──

    /// Compute manifold-aware weighted mean of sigma point states.
    virtual VectorXd stateMean(const std::vector<VectorXd>& sigmas,
                               const std::vector<double>& weights) = 0;

    /// Compute tangent-space residual: state ⊖ mean.
    virtual VectorXd stateResidual(const VectorXd& state, const VectorXd& mean) = 0;

    /// Apply tangent-space correction to state: state ⊕ correction.
    virtual VectorXd applyStateCorrection(const VectorXd& state, const VectorXd& correction) = 0;

    /// Compute manifold-aware weighted mean of measurement sigma points.
    virtual VectorXd measurementMean(const std::vector<VectorXd>& sigmas,
                                     const std::vector<double>& weights) = 0;

    /// Compute measurement tangent-space residual: z ⊖ zMean.
    virtual VectorXd measurementResidual(const VectorXd& z, const VectorXd& zMean) = 0;

    /// Full state dimension (e.g., 16 for rigid body).
    virtual int stateDim() const = 0;

    /// Tangent/error-state dimension (e.g., 15 for rigid body).
    virtual int tangentDim() const = 0;
};

/**
 * @brief Composite system model with pluggable manifold layout.
 *
 * Implements manifold-aware mean, residual, and correction by iterating
 * over a list of manifold components (e.g., Euclidean for position/velocity,
 * Quaternion for orientation). Subclasses only need to implement process()
 * and measurement().
 */
class SystemModel : public UKFModel
{
protected:
    std::vector<std::shared_ptr<Manifold>> mStateLayout;
    std::vector<std::shared_ptr<Manifold>> mMeasurementLayout;

    // Cached dimensions (computed once in constructor)
    int mStateDim = 0;
    int mTangentDim = 0;
    int mMeasurementStateDim = 0;
    int mMeasurementTangentDim = 0;

public:
    SystemModel(std::vector<std::shared_ptr<Manifold>> stateLayout,
                std::vector<std::shared_ptr<Manifold>> measurementLayout)
        : mStateLayout(std::move(stateLayout)), mMeasurementLayout(std::move(measurementLayout))
    {
        // Pre-compute dimensions once
        for (const auto& manifold : mStateLayout)
        {
            mStateDim += manifold->stateSize();
            mTangentDim += manifold->tangentSize();
        }
        for (const auto& manifold : mMeasurementLayout)
        {
            mMeasurementStateDim += manifold->stateSize();
            mMeasurementTangentDim += manifold->tangentSize();
        }
    }

    int stateDim() const override { return mStateDim; }
    int tangentDim() const override { return mTangentDim; }

    // --- State Operations ---
    VectorXd stateMean(const std::vector<VectorXd>& sigmas,
                       const std::vector<double>& weights) override
    {
        if (mStateLayout.empty())
        {
            if (sigmas.empty())
            {
                return VectorXd();
            }
            VectorXd sum = VectorXd::Zero(sigmas[0].size());
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                sum += sigmas[i] * weights[i];
            }
            return sum;
        }

        // Pre-allocate result; write each manifold mean directly via segment()
        VectorXd result(mStateDim);
        int stateOffset = 0;

        // Temporary: reuse across manifolds to avoid re-allocation per manifold
        std::vector<VectorXd> partSamples(sigmas.size());

        for (const auto& manifold : mStateLayout)
        {
            int partDim = manifold->stateSize();
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                partSamples[i] = sigmas[i].segment(stateOffset, partDim);
            }
            result.segment(stateOffset, partDim) = manifold->mean(partSamples, weights);
            stateOffset += partDim;
        }
        return result;
    }

    VectorXd stateResidual(const VectorXd& state, const VectorXd& mean) override
    {
        if (mStateLayout.empty())
        {
            return state - mean;
        }

        // Write directly into pre-sized result
        VectorXd result(mTangentDim);
        int stateOffset = 0;
        int tangentOffset = 0;

        for (const auto& manifold : mStateLayout)
        {
            int stateSize = manifold->stateSize();
            int tangentSize = manifold->tangentSize();
            result.segment(tangentOffset, tangentSize) = manifold->boxMinus(
                state.segment(stateOffset, stateSize), mean.segment(stateOffset, stateSize));
            stateOffset += stateSize;
            tangentOffset += tangentSize;
        }
        return result;
    }

    VectorXd applyStateCorrection(const VectorXd& state, const VectorXd& correction) override
    {
        if (mStateLayout.empty())
        {
            return state + correction;
        }

        // Write directly into pre-sized result
        VectorXd result(mStateDim);
        int stateOffset = 0;
        int tangentOffset = 0;

        for (const auto& manifold : mStateLayout)
        {
            int stateSize = manifold->stateSize();
            int tangentSize = manifold->tangentSize();
            result.segment(stateOffset, stateSize) =
                manifold->boxPlus(state.segment(stateOffset, stateSize),
                                  correction.segment(tangentOffset, tangentSize));
            stateOffset += stateSize;
            tangentOffset += tangentSize;
        }
        return result;
    }

    // --- Measurement Operations ---
    VectorXd measurementMean(const std::vector<VectorXd>& sigmas,
                             const std::vector<double>& weights) override
    {
        if (mMeasurementLayout.empty())
        {
            VectorXd sum = VectorXd::Zero(sigmas[0].size());
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                sum += sigmas[i] * weights[i];
            }
            return sum;
        }

        // Pre-allocate result
        VectorXd result(mMeasurementStateDim);
        int measurementOffset = 0;

        std::vector<VectorXd> partSamples(sigmas.size());

        for (const auto& manifold : mMeasurementLayout)
        {
            int partDim = manifold->stateSize();
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                partSamples[i] = sigmas[i].segment(measurementOffset, partDim);
            }
            result.segment(measurementOffset, partDim) = manifold->mean(partSamples, weights);
            measurementOffset += partDim;
        }
        return result;
    }

    VectorXd measurementResidual(const VectorXd& z, const VectorXd& zMean) override
    {
        if (mMeasurementLayout.empty())
        {
            return z - zMean;
        }

        // Write directly into pre-sized result
        VectorXd result(mMeasurementTangentDim);
        int stateOffset = 0;
        int tangentOffset = 0;

        for (const auto& manifold : mMeasurementLayout)
        {
            int stateSize = manifold->stateSize();
            int tangentSize = manifold->tangentSize();
            result.segment(tangentOffset, tangentSize) = manifold->boxMinus(
                z.segment(stateOffset, stateSize), zMean.segment(stateOffset, stateSize));
            stateOffset += stateSize;
            tangentOffset += tangentSize;
        }
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────
// MeasurementModel — standalone sensor interface for multi-sensor fusion
//
// Each sensor carries its own:
//   - predict()  : state → measurement mapping
//   - R          : measurement noise covariance
//   - manifold layout : for proper mean/residual computation
//
// Usage:
//   auto pose = make_shared<PoseSensor>(R_pose);
//   auto cam  = make_shared<ProjectiveBBoxSensor>(cam_params, R_cam);
//
//   ukf.correct(z_pose, pose);   // 7D pose measurement
//   ukf.correct(z_cam, cam);     // 4D bbox measurement
// ─────────────────────────────────────────────────────────────────────
class MeasurementModel
{
protected:
    std::vector<std::shared_ptr<Manifold>> mMeasurementLayout;
    int mRawMeasurementDim = 0;
    int mTangentMeasurementDim = 0;
    MatrixXd mNoiseCov; // R

public:
    MeasurementModel(std::vector<std::shared_ptr<Manifold>> measurementLayout,
                     const MatrixXd& noiseCov)
        : mMeasurementLayout(std::move(measurementLayout)), mNoiseCov(noiseCov)
    {
        for (const auto& manifold : mMeasurementLayout)
        {
            mRawMeasurementDim += manifold->stateSize();
            mTangentMeasurementDim += manifold->tangentSize();
        }
    }

    virtual ~MeasurementModel() = default;

    /// Map state to predicted measurement.
    virtual VectorXd predict(const VectorXd& state) = 0;

    int rawDim() const { return mRawMeasurementDim; }
    int tangentDim() const { return mTangentMeasurementDim; }
    const MatrixXd& noiseCov() const { return mNoiseCov; }
    void setNoiseCov(const MatrixXd& R) { mNoiseCov = R; }

    VectorXd computeMean(const std::vector<VectorXd>& sigmas, const std::vector<double>& weights)
    {
        if (mMeasurementLayout.empty())
        {
            VectorXd sum = VectorXd::Zero(sigmas[0].size());
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                sum += sigmas[i] * weights[i];
            }
            return sum;
        }

        VectorXd result(mRawMeasurementDim);
        int measurementOffset = 0;
        std::vector<VectorXd> partSamples(sigmas.size());

        for (const auto& manifold : mMeasurementLayout)
        {
            int partDim = manifold->stateSize();
            for (size_t i = 0; i < sigmas.size(); ++i)
            {
                partSamples[i] = sigmas[i].segment(measurementOffset, partDim);
            }
            result.segment(measurementOffset, partDim) = manifold->mean(partSamples, weights);
            measurementOffset += partDim;
        }
        return result;
    }

    VectorXd computeResidual(const VectorXd& z, const VectorXd& zMean)
    {
        if (mMeasurementLayout.empty())
        {
            return z - zMean;
        }

        VectorXd result(mTangentMeasurementDim);
        int stateOffset = 0;
        int tangentOffset = 0;

        for (const auto& manifold : mMeasurementLayout)
        {
            int stateSize = manifold->stateSize();
            int tangentSize = manifold->tangentSize();
            result.segment(tangentOffset, tangentSize) = manifold->boxMinus(
                z.segment(stateOffset, stateSize), zMean.segment(stateOffset, stateSize));
            stateOffset += stateSize;
            tangentOffset += tangentSize;
        }
        return result;
    }
};

} // namespace quak
