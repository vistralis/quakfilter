// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <string>

namespace quak
{

/// Standard clock and type aliases for temporal operations.
using Clock = std::chrono::system_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::duration<double>;

/// Time mode for TrackPool — locks on first temporal call.
enum class TimeMode
{
    Unset = 0,    ///< No temporal call made yet
    Relative = 1, ///< Using predict(dt) — simulation mode
    Absolute = 2  ///< Using predictTo(timestamp) — production mode
};

/// Read-only output container for a complete state estimate snapshot.
///
/// Assembles state (x), covariance (P), model probabilities, and timestamp
/// into a single self-describing object. This is an output type — not the
/// internal storage format of UKF or IMMEstimator.
struct StateEstimate
{
    std::string trackId;
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
    Eigen::VectorXd modelProbabilities;
    TimePoint timestamp;
};

} // namespace quak
