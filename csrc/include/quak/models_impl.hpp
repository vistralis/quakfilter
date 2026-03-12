// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/models.hpp"

namespace quak
{

// ─────────────────────────────────────────────────────────────────────
// Shared 16D state layout for all rigid-body models:
//   [x, y, z,  vx, vy, vz,  ax, ay, az,  qw, qx, qy, qz,  wx, wy, wz]
//    Pos(3)     Vel(3)        Acc(3)        Ori(4)            AngVel(3)
//
// Tangent dimension = 15  (Euclidean(9) + so(3) + Euclidean(3))
// Measurement = 7 = Pos(3) + Ori(4),  tangent = 6
// ─────────────────────────────────────────────────────────────────────

static constexpr int kPosIdx = 0;
static constexpr int kVelIdx = 3;
static constexpr int kAccIdx = 6;
static constexpr int kOriIdx = 9;
static constexpr int kAngVelIdx = 13;

static constexpr int kPosDim = 3;
static constexpr int kVelDim = 3;
static constexpr int kAccDim = 3;
static constexpr int kOriDim = 4;
static constexpr int kAngVelDim = 3;

static constexpr int kStateDim = 16;
static constexpr int kTangentDim = 15;
static constexpr int kMeasDim = 7;

/// Build the standard 16D rigid-body state manifold layout.
inline auto makeStateLayout()
{
    return std::vector<std::shared_ptr<Manifold>>{
        std::make_shared<Euclidean>(3), // pos
        std::make_shared<Euclidean>(3), // vel
        std::make_shared<Euclidean>(3), // acc
        std::make_shared<Quaternion>(), // ori
        std::make_shared<Euclidean>(3)  // angvel
    };
}

/// Build the standard 7D [pos, quat] measurement manifold layout.
inline auto makeMeasLayout()
{
    return std::vector<std::shared_ptr<Manifold>>{
        std::make_shared<Euclidean>(3), // pos
        std::make_shared<Quaternion>()  // ori
    };
}

/// Extract pose measurement h(x) = [pos, quat] from 16D state.
inline VectorXd rigidBodyMeasurement(const VectorXd& state)
{
    VectorXd z(kMeasDim);
    z.segment<kPosDim>(0) = state.segment<kPosDim>(kPosIdx);
    z.segment<kOriDim>(3) = state.segment<kOriDim>(kOriIdx);
    return z;
}

/// Integrate quaternion orientation by angular velocity over dt.
inline Quaterniond integrateOrientation(const Quaterniond& q, const Eigen::Vector3d& w, double dt)
{
    double angle = w.norm() * dt;
    if (angle > 1e-9)
    {
        Eigen::Vector3d axis = w / w.norm();
        Quaterniond qDelta(Eigen::AngleAxisd(angle, axis));
        return (q * qDelta).normalized();
    }
    return q;
}

/// Pack position, velocity, acceleration, orientation, angular velocity into 16D state.
inline VectorXd packState(const Eigen::Vector3d& p,
                          const Eigen::Vector3d& v,
                          const Eigen::Vector3d& a,
                          const Quaterniond& q,
                          const Eigen::Vector3d& w)
{
    VectorXd s(kStateDim);
    s.segment<kPosDim>(kPosIdx) = p;
    s.segment<kVelDim>(kVelIdx) = v;
    s.segment<kAccDim>(kAccIdx) = a;
    s.segment<kOriDim>(kOriIdx) << q.w(), q.x(), q.y(), q.z();
    s.segment<kAngVelDim>(kAngVelIdx) = w;
    return s;
}

/// Constant Velocity (CV) — acceleration is zeroed each step.
class RigidBodyCVModel : public SystemModel
{
public:
    RigidBodyCVModel() : SystemModel(makeStateLayout(), makeMeasLayout()) {}

    /// Propagate at constant velocity: p += v*dt, a = 0.
    VectorXd process(const VectorXd& state, const VectorXd& /*control*/, double dt) override
    {
        Eigen::Vector3d p = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d v = state.segment<kVelDim>(kVelIdx);
        Eigen::Vector4d qVec = state.segment<kOriDim>(kOriIdx);
        Quaterniond q(qVec(0), qVec(1), qVec(2), qVec(3));
        Eigen::Vector3d w = state.segment<kAngVelDim>(kAngVelIdx);

        Eigen::Vector3d pNext = p + v * dt;
        Eigen::Vector3d vNext = v; // Constant velocity
        Quaterniond qNext = integrateOrientation(q, w, dt);

        return packState(pNext, vNext, Eigen::Vector3d::Zero(), qNext, w);
    }

    VectorXd measurement(const VectorXd& state) override { return rigidBodyMeasurement(state); }
};

// Backward-compatible alias
using RigidBodyPoseSensorModel = RigidBodyCVModel;

/// Helical — velocity rotates with orientation change, acceleration zeroed.
class RigidBodyHelicalModel : public SystemModel
{
public:
    RigidBodyHelicalModel() : SystemModel(makeStateLayout(), makeMeasLayout()) {}

    /// Propagate with velocity co-rotating with body orientation.
    VectorXd process(const VectorXd& state, const VectorXd& /*control*/, double dt) override
    {
        Eigen::Vector3d p = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d v = state.segment<kVelDim>(kVelIdx);
        Eigen::Vector4d qVec = state.segment<kOriDim>(kOriIdx);
        Quaterniond q(qVec(0), qVec(1), qVec(2), qVec(3));
        Eigen::Vector3d w = state.segment<kAngVelDim>(kAngVelIdx);

        Eigen::Vector3d pNext = p + v * dt;
        Quaterniond qNext = integrateOrientation(q, w, dt);

        // Helical: velocity rotates with orientation change
        Eigen::Vector3d vNext;
        double angle = w.norm() * dt;
        if (angle > 1e-9)
        {
            Eigen::Vector3d axis = w / w.norm();
            Quaterniond qDelta(Eigen::AngleAxisd(angle, axis));
            vNext = qDelta * v;
        }
        else
        {
            vNext = v;
        }

        return packState(pNext, vNext, Eigen::Vector3d::Zero(), qNext, w);
    }

    VectorXd measurement(const VectorXd& state) override { return rigidBodyMeasurement(state); }
};

/// Constant Turn Rate and Velocity (CTRV) — velocity locked to body X axis.
class RigidBodyCTRVModel : public SystemModel
{
public:
    RigidBodyCTRVModel() : SystemModel(makeStateLayout(), makeMeasLayout()) {}

    /// Propagate with velocity aligned to body-frame X axis.
    VectorXd process(const VectorXd& state, const VectorXd& /*control*/, double dt) override
    {
        Eigen::Vector3d p = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d v = state.segment<kVelDim>(kVelIdx);
        Eigen::Vector4d qVec = state.segment<kOriDim>(kOriIdx);
        Quaterniond q(qVec(0), qVec(1), qVec(2), qVec(3));
        Eigen::Vector3d w = state.segment<kAngVelDim>(kAngVelIdx);

        double speed = v.norm();
        Eigen::Vector3d pNext = p + v * dt;
        Quaterniond qNext = integrateOrientation(q, w, dt);

        // CTRV: Velocity always points in body X direction
        Eigen::Vector3d vNext = qNext * Eigen::Vector3d(speed, 0, 0);

        return packState(pNext, vNext, Eigen::Vector3d::Zero(), qNext, w);
    }

    VectorXd measurement(const VectorXd& state) override { return rigidBodyMeasurement(state); }
};

/// Constant Acceleration (CA) — acceleration is propagated unchanged.
class RigidBodyCAModel : public SystemModel
{
public:
    RigidBodyCAModel() : SystemModel(makeStateLayout(), makeMeasLayout()) {}

    /// Propagate with constant acceleration: p += v*dt + 0.5*a*dt², v += a*dt.
    VectorXd process(const VectorXd& state, const VectorXd& /*control*/, double dt) override
    {
        Eigen::Vector3d p = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d v = state.segment<kVelDim>(kVelIdx);
        Eigen::Vector3d a = state.segment<kAccDim>(kAccIdx);
        Eigen::Vector4d qVec = state.segment<kOriDim>(kOriIdx);
        Quaterniond q(qVec(0), qVec(1), qVec(2), qVec(3));
        Eigen::Vector3d w = state.segment<kAngVelDim>(kAngVelIdx);

        Eigen::Vector3d pNext = p + v * dt + 0.5 * a * dt * dt;
        Eigen::Vector3d vNext = v + a * dt;
        Quaterniond qNext = integrateOrientation(q, w, dt);

        return packState(pNext, vNext, a, qNext, w);
    }

    VectorXd measurement(const VectorXd& state) override { return rigidBodyMeasurement(state); }
};

// ─────────────────────────────────────────────────────────────────────
// Singer Maneuvering Target Model
//
// Models acceleration as a first-order Markov (Ornstein-Uhlenbeck) process:
//   a(t+dt) = exp(-dt/τ) · a(t)
//
// The time constant τ controls maneuver duration:
//   τ large (e.g., 5s) → sustained maneuvers (aircraft, ships)
//   τ small (e.g., 1s) → brief impulses (pedestrians, cars)
//
// Exact discrete-time state transition (Singer 1970):
//   p(t+dt) = p + v·dt + a · [τ·dt - τ²(1 - α)]
//   v(t+dt) = v + a · τ(1 - α)
//   a(t+dt) = α · a
//
// where α = exp(-dt/τ)
//
// Reference: Singer, R.A. (1970). "Estimating Optimal Tracking Filter
//   Performance for Manned Maneuvering Targets." IEEE Trans. AES.
// ─────────────────────────────────────────────────────────────────────
class RigidBodySingerModel : public SystemModel
{
public:
    /// @param tau Maneuver time constant in seconds.
    ///            Rule of thumb: set to expected maneuver duration.
    explicit RigidBodySingerModel(double tau = 2.0)
        : SystemModel(makeStateLayout(), makeMeasLayout()), mTau(tau)
    {
    }

    VectorXd process(const VectorXd& state, const VectorXd& /*control*/, double dt) override
    {
        Eigen::Vector3d p = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d v = state.segment<kVelDim>(kVelIdx);
        Eigen::Vector3d a = state.segment<kAccDim>(kAccIdx);
        Eigen::Vector4d qVec = state.segment<kOriDim>(kOriIdx);
        Quaterniond q(qVec(0), qVec(1), qVec(2), qVec(3));
        Eigen::Vector3d w = state.segment<kAngVelDim>(kAngVelIdx);

        // Singer discrete transition
        double alpha = std::exp(-dt / mTau);
        double tauDt = mTau * dt;
        double tauSqOneMinusAlpha = mTau * mTau * (1.0 - alpha);

        // Exact integration of the Ornstein-Uhlenbeck process
        Eigen::Vector3d pNext = p + v * dt + a * (tauDt - tauSqOneMinusAlpha);
        Eigen::Vector3d vNext = v + a * mTau * (1.0 - alpha);
        Eigen::Vector3d aNext = a * alpha;

        Quaterniond qNext = integrateOrientation(q, w, dt);

        return packState(pNext, vNext, aNext, qNext, w);
    }

    VectorXd measurement(const VectorXd& state) override { return rigidBodyMeasurement(state); }

    double getTau() const { return mTau; }
    void setTau(double tau) { mTau = tau; }

private:
    double mTau; // maneuver time constant (seconds)
};

// ═════════════════════════════════════════════════════════════════════
// Standalone Measurement Models (for multi-sensor correct)
// ═════════════════════════════════════════════════════════════════════

/// PoseSensor — extracts [pos(3), quat(4)] from state → 7D, tangent 6D.
class PoseSensor : public MeasurementModel
{
public:
    explicit PoseSensor(const MatrixXd& R) : MeasurementModel(makeMeasLayout(), R) {}

    VectorXd predict(const VectorXd& state) override { return rigidBodyMeasurement(state); }
};

// ─────────────────────────────────────────────────────────────────────
// VelocitySensor — extracts [vel(3), angvel(3)] from state → 6D
// ─────────────────────────────────────────────────────────────────────

inline auto makeVelocityMeasLayout()
{
    return std::vector<std::shared_ptr<Manifold>>{
        std::make_shared<Euclidean>(3), // vel
        std::make_shared<Euclidean>(3)  // angvel
    };
}

/// VelocitySensor — extracts [vel(3), angvel(3)] from state → 6D.
class VelocitySensor : public MeasurementModel
{
public:
    explicit VelocitySensor(const MatrixXd& R) : MeasurementModel(makeVelocityMeasLayout(), R) {}

    VectorXd predict(const VectorXd& state) override
    {
        VectorXd z(6);
        z.segment<kVelDim>(0) = state.segment<kVelDim>(kVelIdx);
        z.segment<kAngVelDim>(3) = state.segment<kAngVelDim>(kAngVelIdx);
        return z;
    }
};

} // namespace quak
