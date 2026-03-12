// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/imm.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#ifdef QUAK_HAS_OPENMP
#include <omp.h>
#endif
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quak
{

/**
 * @brief Track pool: batched IMM estimator with UUID-keyed tracks.
 *
 * Manages a dynamic set of IMMEstimator instances (one per tracked object)
 * in a contiguous vector, partitioned as [active | suspended].
 *
 * - predict(dt) runs on all active tracks via OpenMP
 * - correct(ids, measurements) updates only the specified subset
 * - add/remove manage the pool dynamically
 * - suspend/resume move tracks between partitions
 */
class TrackPool
{
    // ── Storage ─────────────────────────────────────────────────────
    std::vector<IMMEstimator> mFilters;
    std::vector<std::string> mTrackIds;             // index → UUID
    std::unordered_map<std::string, int> mIndexMap; // UUID → index
    int mActiveCount = 0; // [0, mActiveCount) = active, [mActiveCount, size) = suspended

    // ── Temporal layer ──────────────────────────────────────────────
    std::vector<TimePoint> mTimestamps;
    TimeMode mTimeMode = TimeMode::Unset;
    Duration mMaxDt{10.0}; // Safety guard: reject dt > 10 seconds

    // ── Shared configuration for spawning new filters ───────────────
    std::vector<std::shared_ptr<UKFModel>> mModels;
    std::vector<MatrixXd> mProcessNoises;
    MatrixXd mMeasurementNoise;
    MatrixXd mDefaultCovariance;
    bool mUseSvd;
    double mPSame;
    double mProbFloor;

    // ── Internal: swap two slots and update maps ────────────────────
    void swapSlots(int a, int b)
    {
        if (a == b)
        {
            return;
        }
        std::swap(mFilters[a], mFilters[b]);
        std::swap(mTrackIds[a], mTrackIds[b]);
        std::swap(mTimestamps[a], mTimestamps[b]);
        mIndexMap[mTrackIds[a]] = a;
        mIndexMap[mTrackIds[b]] = b;
    }

public:
    /**
     * @brief Construct an empty track pool with shared configuration.
     *
     * @param models           Shared motion models (one per IMM sub-filter)
     * @param processNoises    Per-model process noise covariances
     * @param measurementNoise Shared measurement noise R
     * @param initialCovariance Default P₀ for new tracks
     * @param useSvd           Use SVD for sigma points (default: true)
     * @param pSame            Markov self-transition probability (default: 0.95)
     * @param probFloor        Minimum model probability (default: 0.01)
     */
    TrackPool(std::vector<std::shared_ptr<UKFModel>> models,
              const std::vector<MatrixXd>& processNoises,
              const MatrixXd& measurementNoise,
              const MatrixXd& initialCovariance,
              bool useSvd = true,
              double pSame = 0.95,
              double probFloor = 0.01)
        : mModels(std::move(models)), mProcessNoises(processNoises),
          mMeasurementNoise(measurementNoise), mDefaultCovariance(initialCovariance),
          mUseSvd(useSvd), mPSame(pSame), mProbFloor(probFloor)
    {
    }

    // ─── Track Lifecycle ────────────────────────────────────────────

    /// Pre-declare the time mode for this pool (optional — auto-detected on first predict).
    void setTimeMode(TimeMode mode)
    {
        if (mTimeMode != TimeMode::Unset && mTimeMode != mode)
        {
            throw std::runtime_error("TrackPool time mode already locked — cannot change after "
                                     "first temporal operation");
        }
        mTimeMode = mode;
    }

    /// Add a new active track with an initial timestamp.
    void add(const std::string& trackId, const VectorXd& initialState, TimePoint time)
    {
        int index = static_cast<int>(mFilters.size());

        mFilters.emplace_back(mModels,
                              mProcessNoises,
                              mMeasurementNoise,
                              mDefaultCovariance,
                              initialState,
                              mUseSvd,
                              mPSame,
                              mProbFloor);
        mTrackIds.push_back(trackId);
        mIndexMap[trackId] = index;
        mTimestamps.push_back(time);

        // New track goes into active partition — swap into active zone
        swapSlots(mActiveCount, index);
        ++mActiveCount;
    }

    /// Add with custom initial covariance.
    void add(const std::string& trackId,
             const VectorXd& initialState,
             TimePoint time,
             const MatrixXd& initialCovariance)
    {
        int index = static_cast<int>(mFilters.size());

        mFilters.emplace_back(mModels,
                              mProcessNoises,
                              mMeasurementNoise,
                              initialCovariance,
                              initialState,
                              mUseSvd,
                              mPSame,
                              mProbFloor);
        mTrackIds.push_back(trackId);
        mIndexMap[trackId] = index;
        mTimestamps.push_back(time);

        swapSlots(mActiveCount, index);
        ++mActiveCount;
    }

    /// Remove a track by UUID.
    void remove(const std::string& trackId)
    {
        auto it = mIndexMap.find(trackId);
        if (it == mIndexMap.end())
        {
            return;
        }
        int index = it->second;
        bool isActive = index < mActiveCount;

        if (isActive)
        {
            --mActiveCount;
            swapSlots(index, mActiveCount);
            index = mActiveCount;
        }

        int last = static_cast<int>(mFilters.size()) - 1;
        swapSlots(index, last);

        mIndexMap.erase(mTrackIds[last]);
        mFilters.pop_back();
        mTrackIds.pop_back();
        mTimestamps.pop_back();
    }

    /// Suspend a track (moves to inactive partition, skipped by predict).
    void suspend(const std::string& trackId)
    {
        auto it = mIndexMap.find(trackId);
        if (it == mIndexMap.end())
        {
            return;
        }
        int index = it->second;
        if (index >= mActiveCount)
        {
            return; // already suspended
        }
        --mActiveCount;
        swapSlots(index, mActiveCount);
    }

    /// Resume a suspended track (moves back to active partition).
    void resume(const std::string& trackId)
    {
        auto it = mIndexMap.find(trackId);
        if (it == mIndexMap.end())
        {
            return;
        }
        int index = it->second;
        if (index < mActiveCount)
        {
            return; // already active
        }
        swapSlots(index, mActiveCount);
        ++mActiveCount;
    }

    // ─── Batch Operations ───────────────────────────────────────────

    /// Predict all active tracks forward by dt seconds (Relative mode).
    void predict(double dt)
    {
        if (!std::isfinite(dt))
        {
            throw std::runtime_error("predict(): dt is not finite");
        }
        if (mTimeMode == TimeMode::Absolute)
        {
            throw std::runtime_error(
                "TrackPool is in Absolute mode — use predictTo(timestamp) instead of predict(dt)");
        }
        mTimeMode = TimeMode::Relative;

#ifdef QUAK_HAS_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < mActiveCount; ++i)
        {
            mFilters[i].predict(dt);
        }

        // Advance timestamps so they stay meaningful in Relative mode
        Duration duration{dt};
        for (int i = 0; i < mActiveCount; ++i)
        {
            mTimestamps[i] += std::chrono::duration_cast<Clock::duration>(duration);
        }
    }

    /// Predict all active tracks to a target timestamp (Absolute mode).
    ///
    /// Two-pass design for performance:
    ///   Pass 1 (serial): compute per-track dt, enforce safety guards
    ///   Pass 2 (parallel): OpenMP-parallel predict for tracks with dt > 0
    ///   Pass 3 (serial): update timestamps atomically after success
    void predictTo(TimePoint timestamp)
    {
        if (mTimeMode == TimeMode::Relative)
        {
            throw std::runtime_error(
                "TrackPool is in Relative mode — use predict(dt) instead of predictTo(timestamp)");
        }
        mTimeMode = TimeMode::Absolute;

        // Pass 1: Serial validation — compute dt, enforce guards (no mutation)
        std::vector<double> perTrackDt(mActiveCount, 0.0);

        for (int i = 0; i < mActiveCount; ++i)
        {
            Duration elapsed = timestamp - mTimestamps[i];
            double dt = elapsed.count();

            // NaN guard
            if (!std::isfinite(dt))
            {
                throw std::runtime_error("Non-finite dt for track '" + mTrackIds[i] + "'");
            }

            // Safety guards
            if (dt < 0.0)
            {
                throw std::runtime_error("Negative dt detected for track '" + mTrackIds[i] + "'" +
                                         ": dt=" + std::to_string(dt) + "s");
            }
            if (elapsed > mMaxDt)
            {
                throw std::runtime_error("Max dt exceeded for track '" + mTrackIds[i] + "'" +
                                         ": dt=" + std::to_string(dt) +
                                         " > max=" + std::to_string(mMaxDt.count()));
            }

            perTrackDt[i] = (dt < 1e-12) ? 0.0 : dt;
        }

        // Pass 2: Parallel predict — only propagate tracks with dt > 0
#ifdef QUAK_HAS_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < mActiveCount; ++i)
        {
            if (perTrackDt[i] > 0.0)
            {
                mFilters[i].predict(perTrackDt[i]);
            }
        }

        // Pass 3: Atomic timestamp update — only after all validation + predict succeeded
        for (int i = 0; i < mActiveCount; ++i)
        {
            mTimestamps[i] = timestamp;
        }
    }

    /// Non-mutating snapshot at a target timestamp (Absolute mode).
    ///
    /// Uses projectTo on each filter to generate a temporally coherent view
    /// without modifying any filter state or timestamps.
    ///
    /// Locks mode to Absolute on first call (snapshot uses absolute timestamps).
    std::vector<StateEstimate> getSnapshotAt(TimePoint timestamp)
    {
        if (mTimeMode == TimeMode::Relative)
        {
            throw std::runtime_error(
                "TrackPool is in Relative mode — getSnapshotAt requires Absolute mode");
        }
        mTimeMode = TimeMode::Absolute; // Lock to Absolute on first snapshot

        std::vector<StateEstimate> snapshot(mActiveCount);

#ifdef QUAK_HAS_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < mActiveCount; ++i)
        {
            Duration elapsed = timestamp - mTimestamps[i];
            double dt = elapsed.count();

            if (dt < 0.0)
            {
                // Negative dt: return current state with its actual timestamp
                snapshot[i] = mFilters[i].projectTo(0.0);
                snapshot[i].trackId = mTrackIds[i];
                snapshot[i].timestamp = mTimestamps[i];
            }
            else
            {
                snapshot[i] = mFilters[i].projectTo(dt);
                snapshot[i].trackId = mTrackIds[i];
                snapshot[i].timestamp = timestamp;
            }
        }
        return snapshot;
    }

    /// Correct a subset of tracks by UUID.
    /// trackIds and measurements must have the same length.
    /// measurements: (MeasDim, K) — one column per track.
    void correct(const std::vector<std::string>& trackIds, const MatrixXd& measurements)
    {
        int count = static_cast<int>(trackIds.size());

        // Resolve UUIDs → indices (serial, fast map lookup)
        std::vector<int> indices(count, -1);
        for (int k = 0; k < count; ++k)
        {
            auto it = mIndexMap.find(trackIds[k]);
            if (it != mIndexMap.end())
            {
                indices[k] = it->second;
            }
        }

#ifdef QUAK_HAS_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < count; ++k)
        {
            if (indices[k] >= 0)
            {
                mFilters[indices[k]].correct(measurements.col(k));
            }
        }
    }

    /// Correct a subset using a standalone sensor.
    void correct(const std::vector<std::string>& trackIds,
                 const MatrixXd& measurements,
                 std::shared_ptr<MeasurementModel> sensor)
    {
        int count = static_cast<int>(trackIds.size());

        std::vector<int> indices(count, -1);
        for (int k = 0; k < count; ++k)
        {
            auto it = mIndexMap.find(trackIds[k]);
            if (it != mIndexMap.end())
            {
                indices[k] = it->second;
            }
        }

#ifdef QUAK_HAS_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < count; ++k)
        {
            if (indices[k] >= 0)
            {
                mFilters[indices[k]].correct(measurements.col(k), sensor);
            }
        }
    }

    // ─── Accessors ──────────────────────────────────────────────────

    /// All states as (StateDim, N) matrix (active + suspended).
    MatrixXd getStates() const
    {
        int n = static_cast<int>(mFilters.size());
        if (n == 0)
        {
            return MatrixXd();
        }
        int stateDim = mFilters[0].getState().size();
        MatrixXd states(stateDim, n);
        for (int i = 0; i < n; ++i)
        {
            states.col(i) = mFilters[i].getState();
        }
        return states;
    }

    /// Model probabilities as (NumModels, N) matrix.
    MatrixXd getModelProbabilities() const
    {
        int n = static_cast<int>(mFilters.size());
        if (n == 0)
        {
            return MatrixXd();
        }
        int numModels = mFilters[0].getNumModels();
        MatrixXd probs(numModels, n);
        for (int i = 0; i < n; ++i)
        {
            probs.col(i) = mFilters[i].getModelProbabilities();
        }
        return probs;
    }

    /// All track UUIDs (order matches states columns).
    const std::vector<std::string>& trackIds() const { return mTrackIds; }

    /// Active track UUIDs only.
    std::vector<std::string> activeTrackIds() const
    {
        return std::vector<std::string>(mTrackIds.begin(), mTrackIds.begin() + mActiveCount);
    }

    int size() const { return static_cast<int>(mFilters.size()); }
    int activeCount() const { return mActiveCount; }
    bool contains(const std::string& trackId) const { return mIndexMap.count(trackId) > 0; }
    TimeMode getTimeMode() const { return mTimeMode; }

    /// Access individual filter by UUID.
    const IMMEstimator& filter(const std::string& trackId) const
    {
        return mFilters[mIndexMap.at(trackId)];
    }
    IMMEstimator& filter(const std::string& trackId) { return mFilters[mIndexMap.at(trackId)]; }

    // ─── Temporal Accessors ─────────────────────────────────────────

    /// Get a complete StateEstimate for a single track.
    StateEstimate getEstimate(const std::string& trackId) const
    {
        auto it = mIndexMap.find(trackId);
        if (it == mIndexMap.end())
        {
            throw std::runtime_error("Track not found: " + trackId);
        }
        int index = it->second;

        StateEstimate est;
        est.trackId = trackId;
        est.state = mFilters[index].getState();
        est.covariance = mFilters[index].getStateErrorCovariance();
        est.modelProbabilities = mFilters[index].getModelProbabilities();
        est.timestamp = mTimestamps[index];
        return est;
    }

    /// Get StateEstimates for all active tracks.
    std::vector<StateEstimate> getEstimates() const
    {
        std::vector<StateEstimate> estimates(mActiveCount);
        for (int i = 0; i < mActiveCount; ++i)
        {
            estimates[i].trackId = mTrackIds[i];
            estimates[i].state = mFilters[i].getState();
            estimates[i].covariance = mFilters[i].getStateErrorCovariance();
            estimates[i].modelProbabilities = mFilters[i].getModelProbabilities();
            estimates[i].timestamp = mTimestamps[i];
        }
        return estimates;
    }

    /// Get timestamps for all tracks as a vector.
    const std::vector<TimePoint>& getTimestamps() const { return mTimestamps; }
};

} // namespace quak
