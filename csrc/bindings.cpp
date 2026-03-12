// Copyright (c) 2026 Vistralis Labs. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "quak/track_pool.hpp"
#include "quak/manifolds.hpp"
#include "quak/models_impl.hpp"
#include "quak/projective_model.hpp"
#include "quak/types.hpp"
#include "quak/ukf.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace quak;

/// Convert epoch seconds (double) to system_clock::time_point.
inline TimePoint epochToTimePoint(double epochSeconds)
{
    return TimePoint(std::chrono::duration_cast<Clock::duration>(Duration(epochSeconds)));
}

/// Convert system_clock::time_point to epoch seconds (double).
inline double timePointToEpoch(TimePoint tp)
{
    return Duration(tp.time_since_epoch()).count();
}

NB_MODULE(_core, m)
{
    m.doc() = "Fast UKF implementation using Nanobind and Eigen";

    // TimeMode Enum
    nb::enum_<TimeMode>(m, "TimeMode")
        .value("Unset", TimeMode::Unset)
        .value("Relative", TimeMode::Relative)
        .value("Absolute", TimeMode::Absolute);

    // StateEstimate Binding — read-only output container
    nb::class_<StateEstimate>(m, "StateEstimate")
        .def(nb::init<>())
        .def_ro("track_id", &StateEstimate::trackId)
        .def_ro("state", &StateEstimate::state)
        .def_ro("covariance", &StateEstimate::covariance)
        .def_ro("model_probabilities", &StateEstimate::modelProbabilities)
        .def_prop_ro("timestamp",
                     [](const StateEstimate& self) { return timePointToEpoch(self.timestamp); });

    // Manifold Bindings
    nb::class_<Manifold>(m, "Manifold")
        .def("box_plus", &Manifold::boxPlus)
        .def("box_minus", &Manifold::boxMinus)
        .def("mean", &Manifold::mean)
        .def("state_size", &Manifold::stateSize)
        .def("tangent_size", &Manifold::tangentSize);

    nb::class_<Euclidean, Manifold>(m, "Euclidean").def(nb::init<int>());

    nb::class_<Quaternion, Manifold>(m, "Quaternion").def(nb::init<>());

    // Model Bindings
    nb::class_<UKFModel>(m, "UKFModel");

    nb::class_<SystemModel, UKFModel>(m, "SystemModel")
        .def("process", &SystemModel::process, "state"_a, "control"_a, "dt"_a)
        .def("measurement", &SystemModel::measurement, "state"_a);

    nb::class_<RigidBodyCVModel, SystemModel>(m, "RigidBodyCVModel").def(nb::init<>());

    // Backward-compatible alias
    m.attr("RigidBodyPoseSensorModel") = m.attr("RigidBodyCVModel");

    nb::class_<RigidBodyHelicalModel, SystemModel>(m, "RigidBodyHelicalModel").def(nb::init<>());

    nb::class_<RigidBodyCTRVModel, SystemModel>(m, "RigidBodyCTRVModel").def(nb::init<>());

    nb::class_<RigidBodyCAModel, SystemModel>(m, "RigidBodyCAModel").def(nb::init<>());

    nb::class_<RigidBodySingerModel, SystemModel>(m, "RigidBodySingerModel")
        .def(nb::init<double>(), "tau"_a = 2.0)
        .def("get_tau", &RigidBodySingerModel::getTau)
        .def("set_tau", &RigidBodySingerModel::setTau, "tau"_a);

    // CameraParams for projective model
    nb::class_<CameraParams>(m, "CameraParams")
        .def(nb::init<>())
        .def_rw("K", &CameraParams::K)
        .def_rw("R_cw", &CameraParams::R_cw)
        .def_rw("t_cw", &CameraParams::t_cw)
        .def_rw("obj_dims", &CameraParams::obj_dims)
        .def_rw("min_depth", &CameraParams::min_depth);

    // Projective bounding box model (wraps any dynamics model)
    nb::class_<ProjectiveBBoxModel, SystemModel>(m, "ProjectiveBBoxModel")
        .def(nb::init<std::shared_ptr<SystemModel>, const CameraParams&>(),
             "dynamics_model"_a,
             "camera_params"_a)
        .def("set_camera_extrinsics", &ProjectiveBBoxModel::setCameraExtrinsics, "R_cw"_a, "t_cw"_a)
        .def("set_object_dimensions", &ProjectiveBBoxModel::setObjectDimensions, "dims"_a);

    // ── Standalone Measurement Models (for multi-sensor correct) ──
    nb::class_<MeasurementModel>(m, "MeasurementModel")
        .def("predict", &MeasurementModel::predict, "state"_a)
        .def("raw_dim", &MeasurementModel::rawDim)
        .def("tangent_dim", &MeasurementModel::tangentDim)
        .def("noise_cov", &MeasurementModel::noiseCov)
        .def("set_noise_cov", &MeasurementModel::setNoiseCov, "R"_a);

    nb::class_<PoseSensor, MeasurementModel>(m, "PoseSensor")
        .def(nb::init<const MatrixXd&>(), "R"_a);

    nb::class_<VelocitySensor, MeasurementModel>(m, "VelocitySensor")
        .def(nb::init<const MatrixXd&>(), "R"_a);

    nb::class_<ProjectiveBBoxSensor, MeasurementModel>(m, "ProjectiveBBoxSensor")
        .def(nb::init<const CameraParams&, const MatrixXd&>(), "camera_params"_a, "R"_a)
        .def(
            "set_camera_extrinsics", &ProjectiveBBoxSensor::setCameraExtrinsics, "R_cw"_a, "t_cw"_a)
        .def("set_object_dimensions", &ProjectiveBBoxSensor::setObjectDimensions, "dims"_a);

    // NoiseModel Binding
    nb::class_<NoiseModel>(m, "NoiseModel")
        .def_static("Diagonal", &NoiseModel::Diagonal, "sigmas"_a)
        .def_static("Gaussian", &NoiseModel::Gaussian, "cov"_a)
        .def_static("Isotropic", &NoiseModel::Isotropic, "dim"_a, "sigma"_a);

    // UKF PredictionResult (from predict step — for gating)
    nb::class_<PredictionResult>(m, "UKFPredictionResult")
        .def_ro("z_mean", &PredictionResult::z_mean)
        .def_ro("S_zz", &PredictionResult::S_zz);

    // UKF Binding (Standard)
    nb::class_<UKF>(m, "UKF")
        .def(nb::init<std::shared_ptr<UKFModel>,
                      const MatrixXd&,
                      const MatrixXd&,
                      const MatrixXd&,
                      const VectorXd&,
                      bool,
                      bool>(),
             "model"_a,
             "process_noise"_a,
             "measurement_noise"_a,
             "initial_cov"_a,
             "initial_state"_a,
             "svd_decomposition"_a = true,
             "scaled_sigma_points"_a = true)
        .def("predict", &UKF::predict, "dt"_a, "control"_a = VectorXd())
        .def("correct", static_cast<void (UKF::*)(const VectorXd&)>(&UKF::correct), "measurement"_a)
        .def("correct_with_sensor",
             static_cast<void (UKF::*)(const VectorXd&, std::shared_ptr<MeasurementModel>)>(
                 &UKF::correct),
             "measurement"_a,
             "sensor"_a)
        .def("get_state", &UKF::getState)
        .def("get_state_error_covariance", &UKF::getStateErrorCovariance)
        .def("get_innovation_covariance", &UKF::getInnovationCovariance)
        .def("get_process_noise_covariance", &UKF::getProcessNoiseCovariance)
        .def("get_measurement_noise_covariance", &UKF::getMeasurementNoiseCovariance)
        .def("set_state_and_error_covariance",
             &UKF::setStateAndErrorCovariance,
             "state"_a,
             "error_cov"_a)
        .def_prop_ro("x", &UKF::getState)
        .def_prop_ro("P", &UKF::getStateErrorCovariance);

    // IMMEstimator Binding
    nb::class_<IMMEstimator>(m, "IMMEstimator")
        .def(nb::init<std::vector<std::shared_ptr<UKFModel>>,
                      const MatrixXd&,
                      const std::vector<MatrixXd>&,
                      const MatrixXd&,
                      const MatrixXd&,
                      const VectorXd&,
                      const VectorXd&,
                      bool,
                      double>(),
             "models"_a,
             "transition_matrix"_a,
             "process_noises"_a,
             "measurement_noise"_a,
             "initial_cov"_a,
             "initial_state"_a,
             "initial_probs"_a,
             "svd_decomposition"_a = true,
             "prob_floor"_a = 0.01)
        .def(nb::init<std::vector<std::shared_ptr<UKFModel>>,
                      const std::vector<MatrixXd>&,
                      const MatrixXd&,
                      const MatrixXd&,
                      const VectorXd&,
                      bool,
                      double,
                      double>(),
             "models"_a,
             "process_noises"_a,
             "measurement_noise"_a,
             "initial_cov"_a,
             "initial_state"_a,
             "svd_decomposition"_a = true,
             "p_same"_a = 0.95,
             "prob_floor"_a = 0.01)
        .def("predict", &IMMEstimator::predict, "dt"_a, "control"_a = VectorXd())
        .def("correct",
             static_cast<void (IMMEstimator::*)(const VectorXd&)>(&IMMEstimator::correct),
             "measurement"_a)
        .def(
            "correct_with_sensor",
            static_cast<void (IMMEstimator::*)(const VectorXd&, std::shared_ptr<MeasurementModel>)>(
                &IMMEstimator::correct),
            "measurement"_a,
            "sensor"_a)
        .def("get_state", &IMMEstimator::getState)
        .def("get_state_error_covariance", &IMMEstimator::getStateErrorCovariance)
        .def("get_model_probabilities", &IMMEstimator::getModelProbabilities)
        .def("get_transition_matrix", &IMMEstimator::getTransitionMatrix)
        .def("get_num_models", &IMMEstimator::getNumModels)
        .def("get_model_state", &IMMEstimator::getModelState, "i"_a)
        .def("get_model_covariance", &IMMEstimator::getModelCovariance, "i"_a)
        .def("project_to", &IMMEstimator::projectTo, "dt"_a)
        .def("get_prob_floor", &IMMEstimator::getProbFloor)
        .def("set_prob_floor", &IMMEstimator::setProbFloor, "floor"_a)
        .def_prop_ro("x", &IMMEstimator::getState)
        .def_prop_ro("P", &IMMEstimator::getStateErrorCovariance)
        .def_prop_ro("mu", &IMMEstimator::getModelProbabilities);

    // TrackPool Binding (Track Pool)
    nb::class_<TrackPool>(m, "TrackPool")
        .def(nb::init<std::vector<std::shared_ptr<UKFModel>>,
                      const std::vector<MatrixXd>&,
                      const MatrixXd&,
                      const MatrixXd&,
                      bool,
                      double,
                      double>(),
             "models"_a,
             "process_noises"_a,
             "measurement_noise"_a,
             "initial_covariance"_a,
             "svd_decomposition"_a = true,
             "p_same"_a = 0.95,
             "prob_floor"_a = 0.01)
        .def("set_time_mode", &TrackPool::setTimeMode, "mode"_a)
        .def(
            "add",
            [](TrackPool& self, const std::string& trackId, const VectorXd& state, double time)
            { self.add(trackId, state, epochToTimePoint(time)); },
            "track_id"_a,
            "initial_state"_a,
            "time"_a)
        .def(
            "add_with_covariance",
            [](TrackPool& self,
               const std::string& trackId,
               const VectorXd& state,
               double time,
               const MatrixXd& cov) { self.add(trackId, state, epochToTimePoint(time), cov); },
            "track_id"_a,
            "initial_state"_a,
            "time"_a,
            "initial_covariance"_a)
        .def("remove", &TrackPool::remove, "track_id"_a)
        .def("suspend", &TrackPool::suspend, "track_id"_a)
        .def("resume", &TrackPool::resume, "track_id"_a)
        .def("predict", &TrackPool::predict, "dt"_a)
        .def(
            "predict_to",
            [](TrackPool& self, double timestamp) { self.predictTo(epochToTimePoint(timestamp)); },
            "timestamp"_a)
        .def(
            "get_snapshot_at",
            [](TrackPool& self, double timestamp)
            { return self.getSnapshotAt(epochToTimePoint(timestamp)); },
            "timestamp"_a)
        .def("correct",
             static_cast<void (TrackPool::*)(const std::vector<std::string>&, const MatrixXd&)>(
                 &TrackPool::correct),
             "track_ids"_a,
             "measurements"_a)
        .def("correct_with_sensor",
             static_cast<void (TrackPool::*)(const std::vector<std::string>&,
                                             const MatrixXd&,
                                             std::shared_ptr<MeasurementModel>)>(
                 &TrackPool::correct),
             "track_ids"_a,
             "measurements"_a,
             "sensor"_a)
        .def("get_states", &TrackPool::getStates)
        .def("get_model_probabilities", &TrackPool::getModelProbabilities)
        .def("get_estimate", &TrackPool::getEstimate, "track_id"_a)
        .def("get_estimates", &TrackPool::getEstimates)
        .def("get_timestamps",
             [](const TrackPool& self)
             {
                 const auto& tp = self.getTimestamps();
                 std::vector<double> result(tp.size());
                 for (size_t i = 0; i < tp.size(); ++i)
                 {
                     result[i] = timePointToEpoch(tp[i]);
                 }
                 return result;
             })
        .def("track_ids", &TrackPool::trackIds)
        .def("active_track_ids", &TrackPool::activeTrackIds)
        .def("size", &TrackPool::size)
        .def("active_count", &TrackPool::activeCount)
        .def("contains", &TrackPool::contains, "track_id"_a)
        .def_prop_ro("time_mode", &TrackPool::getTimeMode);
}
