// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/types.hpp"
#include "quak/ukf.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace quak
{

/**
 * @brief Interacting Multiple Model (IMM) Estimator
 *
 * Implements the IMM algorithm per Genovese (2001):
 *   "The Interacting Multiple Model Algorithm for Accurate
 *    State Estimation of Maneuvering Targets"
 *   JHU/APL Technical Digest, Vol. 22, No. 4.
 *
 * Each model gets its own UKF instance. The algorithm:
 *   1. State interaction (mixing via conditional probabilities)
 *   2. Per-model UKF predict
 *   3. Per-model UKF correct
 *   4. Model probability update (Gaussian log-likelihood)
 *   5. State estimate combination
 */
class IMMEstimator
{
    std::vector<UKF> mFilters; // N internal UKF instances
    std::vector<std::shared_ptr<UKFModel>> mModels;
    MatrixXd mTransitionMatrix;     // N×N Markov switching matrix Π
    VectorXd mModelProbabilities;   // N×1 model probabilities μ
    VectorXd mState;                // Combined state estimate
    MatrixXd mStateErrorCovariance; // Combined covariance
    int mNumModels;
    double mProbFloor = 0.01; // Minimum per-model probability (prevents saturation)
    bool mInitialized = false;

    // ─── Internal helpers ───────────────────────────────────────────

    /**
     * Step 1: Compute conditional model probabilities μ_{i|j}
     *
     * μ_{i|j} = p_{ij} · μ_i / c̄_j
     * where c̄_j = Σ_i p_{ij} · μ_i
     */
    MatrixXd computeConditionalProbabilities() const
    {
        MatrixXd conditional(mNumModels, mNumModels);

        for (int j = 0; j < mNumModels; ++j)
        {
            double cBar = 0.0;
            for (int i = 0; i < mNumModels; ++i)
            {
                cBar += mTransitionMatrix(i, j) * mModelProbabilities(i);
            }
            for (int i = 0; i < mNumModels; ++i)
            {
                conditional(i, j) = mTransitionMatrix(i, j) * mModelProbabilities(i) / cBar;
            }
        }
        return conditional;
    }

    /**
     * Step 1b: State interaction — mix states and covariances across models
     *
     * x̂₀ⱼ = Σ_i μ_{i|j} · x̂ᵢ
     * P̂₀ⱼ = Σ_i μ_{i|j} · [Pᵢ + (x̂ᵢ - x̂₀ⱼ)(x̂ᵢ - x̂₀ⱼ)ᵀ]
     */
    void interaction(const MatrixXd& conditionalProb,
                     const std::vector<VectorXd>& states,
                     const std::vector<MatrixXd>& covariances,
                     std::vector<VectorXd>& mixedStates,
                     std::vector<MatrixXd>& mixedCovariances)
    {
        int tangentDim = covariances[0].rows();

        mixedStates.resize(mNumModels);
        mixedCovariances.resize(mNumModels);

        for (int j = 0; j < mNumModels; ++j)
        {
            // Mixed state: weighted sum using model's own mean operations
            std::vector<double> weights(mNumModels);
            for (int i = 0; i < mNumModels; ++i)
            {
                weights[i] = conditionalProb(i, j);
            }
            mixedStates[j] = mModels[j]->stateMean(states, weights);

            // Mixed covariance with spread-of-means
            mixedCovariances[j] = MatrixXd::Zero(tangentDim, tangentDim);
            for (int i = 0; i < mNumModels; ++i)
            {
                VectorXd diff = mModels[j]->stateResidual(states[i], mixedStates[j]);
                mixedCovariances[j] +=
                    conditionalProb(i, j) * (covariances[i] + diff * diff.transpose());
            }
            // Symmetrize
            mixedCovariances[j] = (mixedCovariances[j] + mixedCovariances[j].transpose()) * 0.5;
        }
    }

    /**
     * Step 4: Update model probabilities using Gaussian log-likelihood
     *
     * log Λⱼ = -½ log|2π Sⱼ| - ½ Zⱼᵀ Sⱼ⁻¹ Zⱼ
     * μ̂ⱼ = Λⱼ · μⱼ / Σ_i(Λᵢ · μᵢ)
     */
    void updateModelProbabilities(const VectorXd& measurement,
                                  const std::vector<PredictionResult>& predictions)
    {
        std::vector<double> logLikelihoods(mNumModels);

        for (int j = 0; j < mNumModels; ++j)
        {
            VectorXd innovation =
                mModels[j]->measurementResidual(measurement, predictions[j].z_mean);
            const MatrixXd& S = predictions[j].S_zz;

            // Mahalanobis distance and log-determinant via LDLT (stays in log-space)
            double mahalanobis;
            double logDetS;
            {
                auto ldlt = S.ldlt();
                if (ldlt.info() == Eigen::Success && ldlt.isPositive())
                {
                    mahalanobis = (innovation.transpose() * ldlt.solve(innovation))(0, 0);
                    // log|S| = sum(log|D_i|) where S = L*D*L^T
                    logDetS = ldlt.vectorD().array().abs().log().sum();
                }
                else
                {
                    // Fallback: SVD pseudo-inverse
                    Eigen::JacobiSVD<MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    VectorXd singularValues = svd.singularValues();
                    double tolerance = 1e-8 * singularValues(0);
                    VectorXd singularValuesInv(singularValues.size());
                    for (int i = 0; i < singularValues.size(); ++i)
                    {
                        singularValuesInv(i) =
                            (singularValues(i) > tolerance) ? (1.0 / singularValues(i)) : 0.0;
                    }
                    MatrixXd pseudoInverse =
                        svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
                    mahalanobis = (innovation.transpose() * pseudoInverse * innovation)(0, 0);
                    // log|S| from SVD: sum(log(σ_i)) for non-zero singular values
                    logDetS = 0.0;
                    for (int i = 0; i < singularValues.size(); ++i)
                    {
                        if (singularValues(i) > tolerance)
                        {
                            logDetS += std::log(singularValues(i));
                        }
                    }
                }
            }

            // Log-likelihood in log-space (never compute raw determinant)
            // log Λⱼ = -½ [n·log(2π) + log|S|] - ½ νᵀS⁻¹ν
            int measurementDim = S.rows();
            double logNormConst = measurementDim * std::log(2.0 * M_PI) + logDetS;
            logLikelihoods[j] = -0.5 * logNormConst - 0.5 * mahalanobis;
        }

        // ExpNormalize: convert log-likelihoods to normalized probabilities
        double maxLogLikelihood = *std::max_element(logLikelihoods.begin(), logLikelihoods.end());
        std::vector<double> likelihoods(mNumModels);
        for (int j = 0; j < mNumModels; ++j)
        {
            likelihoods[j] = std::exp(logLikelihoods[j] - maxLogLikelihood);
        }

        // Bayesian update: μ̂ⱼ = Λⱼ · μⱼ / Σ(Λᵢ · μᵢ)
        double normConst = 0.0;
        for (int j = 0; j < mNumModels; ++j)
        {
            normConst += likelihoods[j] * mModelProbabilities(j);
        }

        for (int j = 0; j < mNumModels; ++j)
        {
            mModelProbabilities(j) = likelihoods[j] * mModelProbabilities(j) / normConst;
        }

        // Clamp to [floor, 1 - (N-1)*floor] so no model saturates
        double ceiling = 1.0 - (mNumModels - 1) * mProbFloor;
        for (int j = 0; j < mNumModels; ++j)
        {
            mModelProbabilities(j) = std::clamp(mModelProbabilities(j), mProbFloor, ceiling);
        }
        // Re-normalize after clamping
        double sum = mModelProbabilities.sum();
        mModelProbabilities /= sum;
    }

    /**
     * Step 5: Combine state estimates and covariances
     *
     * x̂ = Σ_i μ̂ᵢ · x̂ᵢ
     * P̂ = Σ_i μ̂ᵢ · [Pᵢ + (x̂ᵢ - x̂)(x̂ᵢ - x̂)ᵀ]
     */
    void combineEstimates(const std::vector<VectorXd>& states,
                          const std::vector<MatrixXd>& covariances)
    {
        // Use model 0 for manifold operations (all models share same state layout)
        std::vector<double> weights(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            weights[i] = mModelProbabilities(i);
        }
        mState = mModels[0]->stateMean(states, weights);

        int tangentDim = covariances[0].rows();
        mStateErrorCovariance = MatrixXd::Zero(tangentDim, tangentDim);
        for (int i = 0; i < mNumModels; ++i)
        {
            VectorXd diff = mModels[0]->stateResidual(states[i], mState);
            mStateErrorCovariance +=
                mModelProbabilities(i) * (covariances[i] + diff * diff.transpose());
        }
        mStateErrorCovariance = (mStateErrorCovariance + mStateErrorCovariance.transpose()) * 0.5;
    }

public:
    /**
     * @brief Construct an IMM estimator
     *
     * @param models           N motion models (each gets its own UKF)
     * @param transitionMatrix N×N Markov switching matrix Π (rows must sum to ~1)
     * @param processNoises    Per-model process noise covariance Q
     * @param measurementNoise Shared measurement noise covariance R
     * @param initialCov       Initial state error covariance P₀
     * @param initialState     Initial state estimate x₀
     * @param initialProbs     Initial model probabilities μ₀ (N×1, must sum to 1)
     * @param useSvd           Use SVD for sigma point generation (default: true)
     * @param probFloor        Minimum per-model probability floor (default: 0.01)
     */
    IMMEstimator(std::vector<std::shared_ptr<UKFModel>> models,
                 const MatrixXd& transitionMatrix,
                 const std::vector<MatrixXd>& processNoises,
                 const MatrixXd& measurementNoise,
                 const MatrixXd& initialCov,
                 const VectorXd& initialState,
                 const VectorXd& initialProbs,
                 bool useSvd = true,
                 double probFloor = 0.01)
        : mModels(std::move(models)), mTransitionMatrix(transitionMatrix),
          mModelProbabilities(initialProbs), mState(initialState),
          mStateErrorCovariance(initialCov), mNumModels(static_cast<int>(mModels.size())),
          mProbFloor(probFloor)
    {
        // Create N UKF instances, one per model
        mFilters.reserve(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            mFilters.emplace_back(
                mModels[i], processNoises[i], measurementNoise, initialCov, initialState, useSvd);
        }
        mInitialized = true;
    }

    /**
     * @brief Convenience constructor with uniform initial probabilities
     *        and default transition matrix
     *
     * @param pSame     Probability of staying in same model (default: 0.95)
     * @param probFloor Minimum per-model probability floor (default: 0.01)
     */
    IMMEstimator(std::vector<std::shared_ptr<UKFModel>> models,
                 const std::vector<MatrixXd>& processNoises,
                 const MatrixXd& measurementNoise,
                 const MatrixXd& initialCov,
                 const VectorXd& initialState,
                 bool useSvd = true,
                 double pSame = 0.95,
                 double probFloor = 0.01)
        : mModels(std::move(models)), mState(initialState), mStateErrorCovariance(initialCov),
          mNumModels(static_cast<int>(mModels.size())), mProbFloor(probFloor)
    {
        // Uniform initial probabilities
        mModelProbabilities = VectorXd::Constant(mNumModels, 1.0 / mNumModels);

        // Default transition matrix: pSame on diagonal, rest uniform
        double pOther = (mNumModels > 1) ? (1.0 - pSame) / (mNumModels - 1) : 0.0;
        mTransitionMatrix = MatrixXd::Constant(mNumModels, mNumModels, pOther);
        for (int i = 0; i < mNumModels; ++i)
        {
            mTransitionMatrix(i, i) = pSame;
        }

        mFilters.reserve(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            mFilters.emplace_back(
                mModels[i], processNoises[i], measurementNoise, initialCov, initialState, useSvd);
        }
        mInitialized = true;
    }

    /**
     * @brief IMM predict step
     *
     * 1. Compute conditional probabilities
     * 2. Mix states/covariances across models (interaction)
     * 3. Per-model UKF predict
     * 4. Combine predicted states
     *
     * @return PredictionResult with combined z_mean and S_zz for gating
     */
    PredictionResult predict(double dt, const VectorXd& control = VectorXd())
    {
        // Fast path: single model — bypass all IMM overhead
        if (mNumModels == 1)
        {
            auto result = mFilters[0].predict(dt, control);
            mState = mFilters[0].getState();
            mStateErrorCovariance = mFilters[0].getStateErrorCovariance();
            return result;
        }

        // 1. Conditional probabilities
        MatrixXd conditionalProb = computeConditionalProbabilities();

        // Gather current per-model states and covariances
        std::vector<VectorXd> states(mNumModels);
        std::vector<MatrixXd> covariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            states[i] = mFilters[i].getState();
            covariances[i] = mFilters[i].getStateErrorCovariance();
        }

        // 2. Interaction (state mixing)
        std::vector<VectorXd> mixedStates;
        std::vector<MatrixXd> mixedCovariances;
        interaction(conditionalProb, states, covariances, mixedStates, mixedCovariances);

        // 3. Per-model predict
        std::vector<PredictionResult> predictions(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            mFilters[i].setStateAndErrorCovariance(mixedStates[i], mixedCovariances[i]);
            predictions[i] = mFilters[i].predict(dt, control);
        }

        // 4. Combine predicted states
        std::vector<VectorXd> predictedStates(mNumModels);
        std::vector<MatrixXd> predictedCovariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            predictedStates[i] = mFilters[i].getState();
            predictedCovariances[i] = mFilters[i].getStateErrorCovariance();
        }
        combineEstimates(predictedStates, predictedCovariances);

        // Combined predicted measurement (weighted by model probabilities)
        int zDim = predictions[0].z_mean.size();
        int sDim = predictions[0].S_zz.rows();
        VectorXd combinedZ = VectorXd::Zero(zDim);
        MatrixXd combinedS = MatrixXd::Zero(sDim, sDim);

        for (int i = 0; i < mNumModels; ++i)
        {
            combinedZ += mModelProbabilities(i) * predictions[i].z_mean;
        }
        for (int i = 0; i < mNumModels; ++i)
        {
            VectorXd dz = predictions[i].z_mean - combinedZ;
            combinedS += mModelProbabilities(i) * (predictions[i].S_zz + dz * dz.transpose());
        }
        combinedS = (combinedS + combinedS.transpose()) * 0.5;

        return PredictionResult{combinedZ, combinedS};
    }

    /**
     * @brief IMM correct step
     *
     * 1. Per-model UKF correct
     * 2. Update model probabilities from innovations
     * 3. Combine corrected states
     */
    void correct(const VectorXd& measurement)
    {
        // Fast path: single model — bypass all IMM overhead
        if (mNumModels == 1)
        {
            mFilters[0].correct(measurement);
            mState = mFilters[0].getState();
            mStateErrorCovariance = mFilters[0].getStateErrorCovariance();
            return;
        }

        // Save prediction results for probability update
        std::vector<PredictionResult> predictions(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            // Re-derive z_mean from current predicted state through measurement model
            predictions[i].z_mean = mModels[i]->measurement(mFilters[i].getState());
            predictions[i].S_zz = mFilters[i].getInnovationCovariance();
        }

        // 1. Per-model correct
        for (int i = 0; i < mNumModels; ++i)
        {
            mFilters[i].correct(measurement);
        }

        // 2. Update model probabilities
        updateModelProbabilities(measurement, predictions);

        // 3. Combine corrected states
        std::vector<VectorXd> correctedStates(mNumModels);
        std::vector<MatrixXd> correctedCovariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            correctedStates[i] = mFilters[i].getState();
            correctedCovariances[i] = mFilters[i].getStateErrorCovariance();
        }
        combineEstimates(correctedStates, correctedCovariances);
    }

    /// Multi-sensor correct: use a standalone MeasurementModel
    void correct(const VectorXd& measurement, std::shared_ptr<MeasurementModel> sensor)
    {
        // Fast path: single model
        if (mNumModels == 1)
        {
            mFilters[0].correct(measurement, sensor);
            mState = mFilters[0].getState();
            mStateErrorCovariance = mFilters[0].getStateErrorCovariance();
            return;
        }

        // Save prediction results for probability update (using sensor)
        std::vector<PredictionResult> predictions(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            predictions[i].z_mean = sensor->predict(mFilters[i].getState());
            predictions[i].S_zz = mFilters[i].getInnovationCovariance();
        }

        // 1. Per-model correct with sensor
        for (int i = 0; i < mNumModels; ++i)
        {
            mFilters[i].correct(measurement, sensor);
        }

        // 2. Update model probabilities
        updateModelProbabilities(measurement, predictions);

        // 3. Combine corrected states
        std::vector<VectorXd> correctedStates(mNumModels);
        std::vector<MatrixXd> correctedCovariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            correctedStates[i] = mFilters[i].getState();
            correctedCovariances[i] = mFilters[i].getStateErrorCovariance();
        }
        combineEstimates(correctedStates, correctedCovariances);
    }

    // ─── Getters ────────────────────────────────────────────────────

    VectorXd getState() const { return mState; }
    MatrixXd getStateErrorCovariance() const { return mStateErrorCovariance; }
    VectorXd getModelProbabilities() const { return mModelProbabilities; }
    MatrixXd getTransitionMatrix() const { return mTransitionMatrix; }
    int getNumModels() const { return mNumModels; }

    // Per-model accessors
    VectorXd getModelState(int index) const { return mFilters[index].getState(); }
    MatrixXd getModelCovariance(int index) const
    {
        return mFilters[index].getStateErrorCovariance();
    }

    // Setters
    void setTransitionMatrix(const MatrixXd& transitionMatrix)
    {
        mTransitionMatrix = transitionMatrix;
    }

    void setModelProbabilities(const VectorXd& probabilities)
    {
        mModelProbabilities = probabilities;
    }

    double getProbFloor() const { return mProbFloor; }
    void setProbFloor(double floor) { mProbFloor = floor; }

    /**
     * @brief Non-mutating shadow predict — returns predicted state without modifying the filter.
     *
     * Creates temporary copies of internal UKF filters, runs the full IMM predict
     * pipeline on them, and returns a StateEstimate. The original filter state is
     * untouched.
     *
     * For the single-model fast path, only one UKF copy is needed (~15 KB).
     * For multi-model, N copies are created (stack-scoped, destroyed on return).
     *
     * @param dt Time step to project forward
     * @return StateEstimate with projected x, P, and model probabilities
     */
    StateEstimate projectTo(double dt) const
    {
        StateEstimate result;
        result.timestamp = TimePoint{}; // Caller (TrackPool) sets the real timestamp

        if (mNumModels == 1)
        {
            // Fast path: copy single UKF, predict, extract
            UKF projection = mFilters[0];
            projection.predict(dt);

            result.state = projection.getState();
            result.covariance = projection.getStateErrorCovariance();
            result.modelProbabilities = mModelProbabilities;
            return result;
        }

        // Multi-model: copy all UKFs and run full IMM predict pipeline
        std::vector<UKF> projections = mFilters; // N copies

        // 1. Conditional probabilities (const — no copies needed)
        MatrixXd conditionalProb = computeConditionalProbabilities();

        // 2. Gather current per-model states and covariances
        std::vector<VectorXd> states(mNumModels);
        std::vector<MatrixXd> covariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            states[i] = projections[i].getState();
            covariances[i] = projections[i].getStateErrorCovariance();
        }

        // 3. Interaction (state mixing) — need mutable interaction helper
        int tangentDim = covariances[0].rows();
        std::vector<VectorXd> mixedStates(mNumModels);
        std::vector<MatrixXd> mixedCovariances(mNumModels);

        for (int j = 0; j < mNumModels; ++j)
        {
            std::vector<double> weights(mNumModels);
            for (int i = 0; i < mNumModels; ++i)
            {
                weights[i] = conditionalProb(i, j);
            }
            mixedStates[j] = mModels[j]->stateMean(states, weights);

            mixedCovariances[j] = MatrixXd::Zero(tangentDim, tangentDim);
            for (int i = 0; i < mNumModels; ++i)
            {
                VectorXd diff = mModels[j]->stateResidual(states[i], mixedStates[j]);
                mixedCovariances[j] +=
                    conditionalProb(i, j) * (covariances[i] + diff * diff.transpose());
            }
            mixedCovariances[j] = (mixedCovariances[j] + mixedCovariances[j].transpose()) * 0.5;
        }

        // 4. Per-model predict on copies
        for (int i = 0; i < mNumModels; ++i)
        {
            projections[i].setStateAndErrorCovariance(mixedStates[i], mixedCovariances[i]);
            projections[i].predict(dt);
        }

        // 5. Combine predicted states (using current model probabilities)
        std::vector<VectorXd> predictedStates(mNumModels);
        std::vector<MatrixXd> predictedCovariances(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            predictedStates[i] = projections[i].getState();
            predictedCovariances[i] = projections[i].getStateErrorCovariance();
        }

        // Weighted mean and spread-of-means covariance
        std::vector<double> weights(mNumModels);
        for (int i = 0; i < mNumModels; ++i)
        {
            weights[i] = mModelProbabilities(i);
        }
        result.state = mModels[0]->stateMean(predictedStates, weights);

        result.covariance = MatrixXd::Zero(tangentDim, tangentDim);
        for (int i = 0; i < mNumModels; ++i)
        {
            VectorXd diff = mModels[0]->stateResidual(predictedStates[i], result.state);
            result.covariance +=
                mModelProbabilities(i) * (predictedCovariances[i] + diff * diff.transpose());
        }
        result.covariance = (result.covariance + result.covariance.transpose()) * 0.5;

        result.modelProbabilities = mModelProbabilities;
        return result;
    }
};

} // namespace quak
