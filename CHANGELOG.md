# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-03-07

### Added
- Manifold-aware UKF with proper quaternion handling via boxPlus/boxMinus
- 16D state / 15D error-state for rigid body pose tracking
- Motion models: Constant Velocity, Helical, CTRV, Constant Acceleration, Singer maneuvering
- IMM estimator with Markov switching and log-likelihood model probabilities
- TrackPool for multi-object IMM with UUID handles, active/suspended partitioning, OpenMP-parallel predict
- Multi-sensor fusion: PoseSensor, VelocitySensor, ProjectiveBBoxSensor
- Projective bounding box measurement model (3D → 2D camera projection)
- Sigma point schemes: Symmetric (2n) and Merwe Scaled (2n+1)
- Numerical stability: SVD/LDLT solver selection, log-space likelihoods, covariance regularization
- Python bindings via nanobind with zero-copy Eigen interop
- Spatiotemporal state layer for TrackPool:
  - `StateEstimate` output container with track ID, state, covariance, timestamp, model probabilities
  - `TimeMode` enum (`Unset`, `Relative`, `Absolute`) with runtime mode lock
  - `IMMEstimator.project_to(dt)` — non-mutating shadow predict returning `StateEstimate`
  - `TrackPool.predict_to(timestamp)` — absolute-time prediction with per-track dt computation
  - `TrackPool.get_snapshot_at(timestamp)` — non-mutating temporally coherent snapshot
  - `TrackPool.set_time_mode(mode)` — explicit pool time mode declaration
  - Per-track accessors: `get_estimate(id)`, `get_estimates()`, `get_timestamps()`
- `std::chrono` type safety layer (`Clock`, `TimePoint`, `Duration` aliases in `types.hpp`)
- Python `time_conversions` module:
  - `to_epoch_seconds()` — converts `float | datetime | numpy.datetime64` to epoch seconds
  - `from_epoch_seconds()` — converts epoch seconds to `numpy.datetime64`
  - `to_duration_seconds()` — converts `float | timedelta | numpy.timedelta64` to seconds
- TrackPool Python wrapper accepts `float`, `datetime`, or `numpy.datetime64` for time inputs
- 114 tests with 93% line+branch coverage
