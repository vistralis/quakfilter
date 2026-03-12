// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/models.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

namespace quak
{

class NoiseModel
{
public:
    static MatrixXd Diagonal(const VectorXd& sigmas)
    {
        return sigmas.array().square().matrix().asDiagonal();
    }

    static MatrixXd Gaussian(const MatrixXd& cov) { return cov; }

    static MatrixXd Isotropic(int dim, double sigma)
    {
        return MatrixXd::Identity(dim, dim) * (sigma * sigma);
    }
};

struct PredictionResult
{
    VectorXd z_mean; // Predicted measurement mean
    MatrixXd S_zz;   // Innovation covariance (Pzz + R)
};

class UKF
{
    std::shared_ptr<UKFModel> mModel;
    MatrixXd mProcessNoiseCov;      // Q
    MatrixXd mMeasurementNoiseCov;  // R
    MatrixXd mStateErrorCovariance; // P
    MatrixXd mInnovationCovariance; // S (Pvv)
    VectorXd mState;                // x
    bool mUseSvd;
    bool mUseMerwe; // true = 2n+1 (Merwe Scaled), false = 2n (Symmetric)
    int mErrorDim;  // tangent dimension (state)
    int mNumSigmas; // 2n or 2n+1

    // Merwe Scaled parameters (only used when mUseMerwe = true)
    double mAlpha = 1.0;
    double mBeta = 2.0;
    double mKappa = 0.0;
    double mLambda = 0.0;
    double mGamma = 0.0; // scaling factor: sqrt(n + lambda)

    // ── Pre-allocated workspaces (allocated once in constructor) ──
    MatrixXd mSigmaPoints;      // (StateDim, NumSigmas)  input sigma points
    MatrixXd mPropagatedSigmas; // (StateDim, NumSigmas)  propagated through f()
    MatrixXd mTangentResiduals; // (TanDim, NumSigmas)    state residuals w_i

    // Measurement-space workspaces (sized lazily on first use)
    // CRITICAL: raw measurement dimension (e.g., 7 for [pos,quat]) differs from
    //           measurement tangent dimension (e.g., 6 for [pos,log(quat)])
    MatrixXd mMeasurementSigmas;    // (RawMeasDim, NumSigmas)   raw h(sigma_i)
    MatrixXd mMeasurementResiduals; // (TangentMeasDim, NumSigmas) tangent residuals
    int mRawMeasurementDim = 0;     // cached raw measurement dimension
    int mTangentMeasurementDim = 0; // cached measurement tangent dimension

    // Weights for mean and covariance (stored separately for 2n+1)
    std::vector<double> mWeightsMean; // mean weights (size: mNumSigmas)
    MatrixXd mWeightDiag;             // covariance weight diagonal (NumSigmas x NumSigmas)

    // Cached measurement projection (valid between predict and correct)
    VectorXd mMeasurementMean;
    bool mMeasurementProjected = false;

    // ── Private: ensure measurement workspaces are correctly sized ──
    void ensureMeasurementWorkspace(int rawDim, int tangentDim)
    {
        if (rawDim != mRawMeasurementDim)
        {
            mRawMeasurementDim = rawDim;
            mMeasurementSigmas.resize(rawDim, mNumSigmas);
        }
        if (tangentDim != mTangentMeasurementDim)
        {
            mTangentMeasurementDim = tangentDim;
            mMeasurementResiduals.resize(tangentDim, mNumSigmas);
        }
    }

    /// Symmetrize a covariance matrix and floor its eigenvalues.
    /// Ensures numerical stability after weighted outer products
    /// or subtraction-based covariance updates.
    static MatrixXd regularizeCovariance(const MatrixXd& matrix, double minEigenvalue = 1e-10)
    {
        MatrixXd symmetric = (matrix + matrix.transpose()) * 0.5;
        Eigen::SelfAdjointEigenSolver<MatrixXd> solver(symmetric);
        VectorXd eigenvalues = solver.eigenvalues();
        if (eigenvalues.minCoeff() >= minEigenvalue)
        {
            return symmetric;
        }
        // Floor negative eigenvalues
        for (int i = 0; i < eigenvalues.size(); ++i)
        {
            eigenvalues(i) = std::max(eigenvalues(i), minEigenvalue);
        }
        return solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
    }

public:
    UKF(std::shared_ptr<UKFModel> model,
        const MatrixXd& processNoise,
        const MatrixXd& measurementNoise,
        const MatrixXd& initialCov,
        const VectorXd& initialState,
        bool useSvd = true,
        bool useMerwe = false)
        : mModel(model), mProcessNoiseCov(processNoise), mMeasurementNoiseCov(measurementNoise),
          mStateErrorCovariance(initialCov), mState(initialState), mUseSvd(useSvd),
          mUseMerwe(useMerwe)
    {
        mErrorDim = mStateErrorCovariance.rows();
        int n = mErrorDim;

        if (mUseMerwe)
        {
            // Merwe Scaled Unscented Transform: 2n+1 sigma points
            mLambda = mAlpha * mAlpha * (n + mKappa) - n;
            mGamma = std::sqrt(n + mLambda);
            mNumSigmas = 2 * n + 1;

            double w0Mean = mLambda / (n + mLambda);
            double w0Cov = w0Mean + (1.0 - mAlpha * mAlpha + mBeta);
            double wi = 1.0 / (2.0 * (n + mLambda));

            mWeightsMean.resize(mNumSigmas);
            mWeightsMean[0] = w0Mean;
            for (int i = 1; i < mNumSigmas; ++i)
            {
                mWeightsMean[i] = wi;
            }

            mWeightDiag = MatrixXd::Identity(mNumSigmas, mNumSigmas) * wi;
            mWeightDiag(0, 0) = w0Cov;
        }
        else
        {
            // Symmetric: 2n sigma points, uniform weights
            mNumSigmas = 2 * n;
            mGamma = std::sqrt(n);
            double w = 1.0 / static_cast<double>(mNumSigmas);

            mWeightsMean.resize(mNumSigmas, w);
            mWeightDiag = MatrixXd::Identity(mNumSigmas, mNumSigmas) * w;
        }

        if (mErrorDim != mModel->tangentDim())
        {
            std::cerr << "Warning: initialCov dim " << mErrorDim << " != model tangent dim "
                      << mModel->tangentDim() << std::endl;
        }

        // Pre-allocate state-space workspaces
        int stateDim = mState.size();
        mSigmaPoints.resize(stateDim, mNumSigmas);
        mPropagatedSigmas.resize(stateDim, mNumSigmas);
        mTangentResiduals.resize(mErrorDim, mNumSigmas);
    }

    /// Perform prediction step — propagates sigma points through f() and h().
    /// Returns PredictionResult with predicted measurement mean and innovation covariance.
    PredictionResult predict(double dt, const VectorXd& control = VectorXd())
    {
        int n = mErrorDim;

        // 1. Sigma Point Generation via matrix square root
        MatrixXd augmentedP = mStateErrorCovariance + mProcessNoiseCov;

        MatrixXd S(n, n);
        if (mUseSvd)
        {
            Eigen::JacobiSVD<MatrixXd> svd(augmentedP, Eigen::ComputeThinU | Eigen::ComputeThinV);
            VectorXd s = svd.singularValues();
            for (int i = 0; i < n; ++i)
            {
                if (s(i) < 1e-6)
                {
                    s(i) = 1e-6;
                }
            }
            S = svd.matrixU() * s.cwiseSqrt().asDiagonal() * svd.matrixV().transpose();
        }
        else
        {
            Eigen::LLT<MatrixXd> llt(augmentedP);
            if (llt.info() == Eigen::Success)
            {
                S = llt.matrixL();
            }
            else
            {
                Eigen::LLT<MatrixXd> lltReg(augmentedP + MatrixXd::Identity(n, n) * 1e-9);
                S = lltReg.matrixL();
            }
        }

        S *= mGamma;

        // 2. Generate sigma points
        int offset = 0;
        if (mUseMerwe)
        {
            // σ₀ = mean (identity perturbation)
            mSigmaPoints.col(0) = mState;
            offset = 1;
        }
        for (int i = 0; i < n; ++i)
        {
            mSigmaPoints.col(offset + i) = mModel->applyStateCorrection(mState, S.col(i));
            mSigmaPoints.col(offset + i + n) = mModel->applyStateCorrection(mState, -S.col(i));
        }

        // 3. Propagate sigma points through process model
        for (int i = 0; i < mNumSigmas; ++i)
        {
            mPropagatedSigmas.col(i) = mModel->process(mSigmaPoints.col(i), control, dt);
        }

        // 4. State Mean (via manifold-aware weighted mean)
        std::vector<VectorXd> sigmasVec(mNumSigmas);
        for (int i = 0; i < mNumSigmas; ++i)
        {
            sigmasVec[i] = mPropagatedSigmas.col(i);
        }
        mState = mModel->stateMean(sigmasVec, mWeightsMean);

        // 4. State Covariance via GEMM: P = W_prime * W_diag * W_prime^T
        for (int i = 0; i < mNumSigmas; ++i)
        {
            mTangentResiduals.col(i) = mModel->stateResidual(mPropagatedSigmas.col(i), mState);
        }

        // Single dense matrix multiply — Eigen maps to optimized BLAS/SIMD
        mStateErrorCovariance =
            regularizeCovariance(mTangentResiduals * mWeightDiag * mTangentResiduals.transpose());

        // 5. Project sigma points through measurement model h()
        //    Detect both raw and tangent measurement dimensions
        VectorXd z0 = mModel->measurement(mPropagatedSigmas.col(0));
        int rawMeasurementDim = z0.size();

        // Compute measurement mean first to discover tangent dimension
        std::vector<VectorXd> measurementVec(mNumSigmas);
        measurementVec[0] = z0;
        for (int i = 1; i < mNumSigmas; ++i)
        {
            measurementVec[i] = mModel->measurement(mPropagatedSigmas.col(i));
        }
        mMeasurementMean = mModel->measurementMean(measurementVec, mWeightsMean);

        // Discover tangent dimension from first residual
        VectorXd r0 = mModel->measurementResidual(measurementVec[0], mMeasurementMean);
        int tangentMeasurementDim = r0.size();
        ensureMeasurementWorkspace(rawMeasurementDim, tangentMeasurementDim);

        // Store raw sigma measurements and tangent residuals
        for (int i = 0; i < mNumSigmas; ++i)
        {
            mMeasurementSigmas.col(i) = measurementVec[i];
            mMeasurementResiduals.col(i) =
                mModel->measurementResidual(measurementVec[i], mMeasurementMean);
        }

        // 6. Innovation covariance via GEMM: S = Pzz + R
        MatrixXd Pzz = regularizeCovariance(mMeasurementResiduals * mWeightDiag *
                                            mMeasurementResiduals.transpose());
        mInnovationCovariance = Pzz + mMeasurementNoiseCov;
        mMeasurementProjected = true;

        return PredictionResult{mMeasurementMean, mInnovationCovariance};
    }

    /// Perform correction step — reuses sigma points from predict().
    void correct(const VectorXd& measurement)
    {
        int n = mErrorDim;

        // Recompute measurement projection if predict() wasn't called
        if (!mMeasurementProjected)
        {
            std::vector<VectorXd> measurementVec(mNumSigmas);
            for (int i = 0; i < mNumSigmas; ++i)
            {
                measurementVec[i] = mModel->measurement(mPropagatedSigmas.col(i));
            }
            mMeasurementMean = mModel->measurementMean(measurementVec, mWeightsMean);

            VectorXd r0 = mModel->measurementResidual(measurementVec[0], mMeasurementMean);
            ensureMeasurementWorkspace(measurementVec[0].size(), r0.size());

            for (int i = 0; i < mNumSigmas; ++i)
            {
                mMeasurementSigmas.col(i) = measurementVec[i];
                mMeasurementResiduals.col(i) =
                    mModel->measurementResidual(measurementVec[i], mMeasurementMean);
            }
        }

        if (mTangentMeasurementDim == 0)
        {
            return;
        }

        // Recompute measurement residuals (needed even if cached,
        // because correct may reuse predict's sigma points)
        for (int i = 0; i < mNumSigmas; ++i)
        {
            mMeasurementResiduals.col(i) =
                mModel->measurementResidual(mMeasurementSigmas.col(i), mMeasurementMean);
        }

        // Innovation covariance via GEMM: Pvv = Pzz + R
        MatrixXd Pzz = regularizeCovariance(mMeasurementResiduals * mWeightDiag *
                                            mMeasurementResiduals.transpose());
        MatrixXd Pvv = Pzz + mMeasurementNoiseCov;
        mInnovationCovariance = Pvv;

        // Cross-correlation via GEMM: Pxz = tangent_residuals * weight * measurement_residuals^T
        MatrixXd Pxz = mTangentResiduals * mWeightDiag * mMeasurementResiduals.transpose();

        // Kalman Gain via LDLT with SVD fallback
        // Replaces naive: K = Pxz * Pvv.inverse()
        MatrixXd K;
        {
            auto ldlt = Pvv.ldlt();
            if (ldlt.info() == Eigen::Success && ldlt.isPositive())
            {
                K = Pxz *
                    ldlt.solve(MatrixXd::Identity(mTangentMeasurementDim, mTangentMeasurementDim));
            }
            else
            {
                // Fallback: SVD pseudo-inverse for degenerate cases
                Eigen::JacobiSVD<MatrixXd> svd(Pvv, Eigen::ComputeThinU | Eigen::ComputeThinV);
                VectorXd sv = svd.singularValues();
                double tol = 1e-8 * sv(0); // relative tolerance
                VectorXd svInv(sv.size());
                for (int i = 0; i < sv.size(); ++i)
                {
                    svInv(i) = (sv(i) > tol) ? (1.0 / sv(i)) : 0.0;
                }
                MatrixXd PvvPseudoInv =
                    svd.matrixV() * svInv.asDiagonal() * svd.matrixU().transpose();
                K = Pxz * PvvPseudoInv;
            }
        }

        // State update
        VectorXd innovation = mModel->measurementResidual(measurement, mMeasurementMean);
        VectorXd correction = K * innovation;
        mState = mModel->applyStateCorrection(mState, correction);

        // Covariance update: P = P - K * Pvv * K^T
        MatrixXd term = K * Pvv * K.transpose();
        mStateErrorCovariance = regularizeCovariance(mStateErrorCovariance - term);

        mMeasurementProjected = false;
    }

    /// Multi-sensor correct: use a standalone MeasurementModel instead of the
    /// model's built-in measurement(). R is stored inside the sensor.
    ///
    /// Usage:
    ///   auto pose = make_shared<PoseSensor>(R_pose);
    ///   ukf.correct(z_pose, pose);
    ///
    void correct(const VectorXd& z, std::shared_ptr<MeasurementModel> sensor)
    {
        int n = mErrorDim;

        // 1. Project propagated sigma points through sensor's predict()
        std::vector<VectorXd> measurementVec(mNumSigmas);
        for (int i = 0; i < mNumSigmas; ++i)
        {
            measurementVec[i] = sensor->predict(mPropagatedSigmas.col(i));
        }

        // 2. Measurement mean and residuals (via sensor's manifold ops)
        VectorXd zMean = sensor->computeMean(measurementVec, mWeightsMean);
        int sensorTangentDim = sensor->tangentDim();

        if (sensorTangentDim == 0)
        {
            return;
        }

        MatrixXd measResiduals(sensorTangentDim, mNumSigmas);
        for (int i = 0; i < mNumSigmas; ++i)
        {
            measResiduals.col(i) = sensor->computeResidual(measurementVec[i], zMean);
        }

        // 3. Innovation covariance: Pvv = Pzz + R_sensor
        MatrixXd Pzz =
            regularizeCovariance(measResiduals * mWeightDiag * measResiduals.transpose());
        MatrixXd Pvv = Pzz + sensor->noiseCov();

        // 4. Cross-correlation: Pxz = W' * diag * Z'^T
        MatrixXd Pxz = mTangentResiduals * mWeightDiag * measResiduals.transpose();

        // 5. Kalman Gain via LDLT with SVD fallback
        MatrixXd K;
        {
            auto ldlt = Pvv.ldlt();
            if (ldlt.info() == Eigen::Success && ldlt.isPositive())
            {
                K = Pxz * ldlt.solve(MatrixXd::Identity(sensorTangentDim, sensorTangentDim));
            }
            else
            {
                Eigen::JacobiSVD<MatrixXd> svd(Pvv, Eigen::ComputeThinU | Eigen::ComputeThinV);
                VectorXd singularValues = svd.singularValues();
                double tolerance = 1e-8 * singularValues(0);
                VectorXd singularValuesInv(singularValues.size());
                for (int i = 0; i < singularValues.size(); ++i)
                {
                    singularValuesInv(i) =
                        (singularValues(i) > tolerance) ? (1.0 / singularValues(i)) : 0.0;
                }
                MatrixXd PvvPseudoInv =
                    svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
                K = Pxz * PvvPseudoInv;
            }
        }

        // 6. State update
        VectorXd innovation = sensor->computeResidual(z, zMean);
        VectorXd correction = K * innovation;
        mState = mModel->applyStateCorrection(mState, correction);

        // 7. Covariance update: P = P - K * Pvv * K^T
        MatrixXd term = K * Pvv * K.transpose();
        mStateErrorCovariance = regularizeCovariance(mStateErrorCovariance - term);
    }

    // ── Getters ──
    MatrixXd getProcessNoiseCovariance() const { return mProcessNoiseCov; }
    MatrixXd getMeasurementNoiseCovariance() const { return mMeasurementNoiseCov; }
    MatrixXd getStateErrorCovariance() const { return mStateErrorCovariance; }
    MatrixXd getInnovationCovariance() const { return mInnovationCovariance; }
    VectorXd getState() const { return mState; }

    /// Replace state and error covariance (e.g., for manual reset).
    void setStateAndErrorCovariance(const VectorXd& state, const MatrixXd& errorCov)
    {
        mState = state;
        mStateErrorCovariance = errorCov;
    }

    /// Update process noise Q (must be tangentDim × tangentDim).
    void setProcessNoiseCovariance(const MatrixXd& processNoiseCovariance)
    {
        mProcessNoiseCov = processNoiseCovariance;
    }

    /// Update measurement noise R for the built-in sensor.
    void setMeasurementNoiseCovariance(const MatrixXd& measurementNoiseCovariance)
    {
        mMeasurementNoiseCov = measurementNoiseCovariance;
    }
};

} // namespace quak
