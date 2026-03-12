// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "quak/models_impl.hpp"

namespace quak
{

// ─────────────────────────────────────────────────────────────────────
// Projective Bounding Box Measurement Model
//
// Composable: wraps any existing dynamics model (CV, CTRV, CA, etc.)
// and replaces the measurement function with a 3D → 2D camera projection.
//
// State:       Same 16D: [pos(3), vel(3), acc(3), quat(4), angvel(3)]
// Measurement: 4D:       [u_center, v_center, bbox_width, bbox_height]
//              All Euclidean (pixel coordinates)
//
// The projection function:
//   1. Transform world point to camera frame:  p_cam = R_cw * p_world + t_cw
//   2. Perspective divide:  u = fx * x_cam / z_cam + cx
//                           v = fy * y_cam / z_cam + cy
//   3. Apparent bbox size:  w = fx * obj_width  / z_cam
//                           h = fy * obj_height / z_cam
//
// The UKF propagates sigma points through this projection — no Jacobians
// needed. This handles the 1/z nonlinearity naturally.
// ─────────────────────────────────────────────────────────────────────

static constexpr int kProjMeasDim = 4; // [u, v, w, h]

/// Camera parameters for the projective model.
struct CameraParams
{
    Eigen::Matrix3d K;        // Intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    Eigen::Matrix3d R_cw;     // Rotation: world → camera frame
    Eigen::Vector3d t_cw;     // Translation: world → camera frame
                              //   p_cam = R_cw * p_world + t_cw
    Eigen::Vector3d obj_dims; // Object dimensions [width, height, depth] in meters
    double min_depth{0.1};    // Minimum depth clamp to prevent 1/z singularity
};

/// Measurement layout for projective: all Euclidean(4)
inline auto makeProjMeasLayout()
{
    return std::vector<std::shared_ptr<Manifold>>{std::make_shared<Euclidean>(kProjMeasDim)};
}

/// Projective bounding box model.
///
/// Uses composition: delegates process() to an inner dynamics model,
/// overrides measurement() with camera projection.
///
/// Usage:
///   auto cv_proj = make_shared<ProjectiveBBoxModel>(
///       make_shared<RigidBodyCVModel>(), cam_params);
///
///   // Use in UKF or IMMEstimator exactly like any other model
///   UKF ukf(cv_proj, Q, R_4x4, P0, x0);
///
class ProjectiveBBoxModel : public SystemModel
{
public:
    ProjectiveBBoxModel(std::shared_ptr<SystemModel> dynamics, const CameraParams& cam)
        : SystemModel(makeStateLayout(), makeProjMeasLayout()), mDynamics(std::move(dynamics)),
          mCam(cam)
    {
    }

    /// Delegate process to the inner dynamics model.
    VectorXd process(const VectorXd& state, const VectorXd& control, double dt) override
    {
        return mDynamics->process(state, control, dt);
    }

    /// Project 3D state to 2D bounding box [u, v, w, h].
    VectorXd measurement(const VectorXd& state) override
    {
        // Extract world position from state
        Eigen::Vector3d p_world = state.segment<kPosDim>(kPosIdx);

        // Transform to camera frame: p_cam = R_cw * p_world + t_cw
        Eigen::Vector3d p_cam = mCam.R_cw * p_world + mCam.t_cw;

        // Clamp depth to prevent division by zero / behind-camera
        double z = std::max(p_cam.z(), mCam.min_depth);

        // Camera intrinsics
        double fx = mCam.K(0, 0);
        double fy = mCam.K(1, 1);
        double cx = mCam.K(0, 2);
        double cy = mCam.K(1, 2);

        // Perspective projection → bbox center
        double u = fx * p_cam.x() / z + cx;
        double v = fy * p_cam.y() / z + cy;

        // Apparent bbox size (pinhole model)
        double w = fx * mCam.obj_dims(0) / z; // width in pixels
        double h = fy * mCam.obj_dims(1) / z; // height in pixels

        VectorXd meas(kProjMeasDim);
        meas << u, v, std::abs(w), std::abs(h);
        return meas;
    }

    /// Access camera parameters.
    const CameraParams& cameraParams() const { return mCam; }

    /// Update camera extrinsics at runtime (e.g., moving camera).
    void setCameraExtrinsics(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw)
    {
        mCam.R_cw = R_cw;
        mCam.t_cw = t_cw;
    }

    /// Update object dimensions (e.g., different object class).
    void setObjectDimensions(const Eigen::Vector3d& dims) { mCam.obj_dims = dims; }

private:
    std::shared_ptr<SystemModel> mDynamics;
    CameraParams mCam;
};

// ─────────────────────────────────────────────────────────────────────
// ProjectiveBBoxSensor — standalone measurement model for multi-sensor fusion
//
// Same projection math as ProjectiveBBoxModel, but as a MeasurementModel
// so it can be used with ukf.correct(z, sensor) without wrapping a dynamics model.
// ─────────────────────────────────────────────────────────────────────
class ProjectiveBBoxSensor : public MeasurementModel
{
public:
    ProjectiveBBoxSensor(const CameraParams& cam, const MatrixXd& R)
        : MeasurementModel(makeProjMeasLayout(), R), mCam(cam)
    {
    }

    /// Project 3D state to 2D bounding box [u, v, w, h].
    VectorXd predict(const VectorXd& state) override
    {
        Eigen::Vector3d p_world = state.segment<kPosDim>(kPosIdx);
        Eigen::Vector3d p_cam = mCam.R_cw * p_world + mCam.t_cw;
        double z = std::max(p_cam.z(), mCam.min_depth);

        double fx = mCam.K(0, 0);
        double fy = mCam.K(1, 1);
        double cx = mCam.K(0, 2);
        double cy = mCam.K(1, 2);

        double u = fx * p_cam.x() / z + cx;
        double v = fy * p_cam.y() / z + cy;
        double w = fx * mCam.obj_dims(0) / z;
        double h = fy * mCam.obj_dims(1) / z;

        VectorXd meas(kProjMeasDim);
        meas << u, v, std::abs(w), std::abs(h);
        return meas;
    }

    /// Access camera parameters.
    const CameraParams& cameraParams() const { return mCam; }

    /// Update camera extrinsics at runtime (e.g., moving camera).
    void setCameraExtrinsics(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw)
    {
        mCam.R_cw = R_cw;
        mCam.t_cw = t_cw;
    }

    /// Update object dimensions (e.g., different object class).
    void setObjectDimensions(const Eigen::Vector3d& dims) { mCam.obj_dims = dims; }

private:
    CameraParams mCam;
};

} // namespace quak
