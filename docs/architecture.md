# System Architecture

## Overview

quakfilter implements the Unscented Kalman Filter using an **error-state formulation** on manifolds. The 16-dimensional state (including a unit quaternion) is paired with a 15-dimensional error/tangent space where all covariance operations take place.

## Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                        │
│  nanobind bindings (bindings.cpp)                       │
├─────────────────────────────────────────────────────────┤
│                     Estimators                          │
│  ┌──────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │   UKF    │  │ IMMEstimator   │  │   TrackPool    │  │
│  │ (core)   │  │ (multi-model)  │  │ (multi-object) │  │
│  └────┬─────┘  └───────┬────────┘  └───────┬────────┘  │
│       │                │                    │           │
│  ┌────▼────────────────▼────────────────────▼──────┐    │
│  │               Model Layer                       │    │
│  │  UKFModel → SystemModel → RigidBody*Model       │    │
│  │  MeasurementModel → PoseSensor / VelocitySensor │    │
│  │                   → ProjectiveBBoxSensor        │    │
│  └─────────────────────┬───────────────────────────┘    │
│                        │                                │
│  ┌─────────────────────▼───────────────────────────┐    │
│  │            Manifold Layer                       │    │
│  │  Manifold → Euclidean | Quaternion              │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  Eigen3          OpenMP           Intel MKL (optional)  │
└─────────────────────────────────────────────────────────┘
```

## Manifold Abstraction

The [`Manifold`](../csrc/include/quak/manifolds.hpp) base class defines three operations that enable the UKF to work on non-Euclidean spaces:

| Operation | Signature | Purpose |
|-----------|-----------|---------|
| `box_plus(x, δ)` | State × Tangent → State | Apply a tangent-space perturbation to a state |
| `box_minus(x₁, x₂)` | State × State → Tangent | Compute the tangent-space difference between states |
| `mean(samples, weights)` | States × Weights → State | Compute the weighted mean on the manifold |

### Implementations

- **`Euclidean(n)`** — Standard vector addition/subtraction. `box_plus = x + δ`, `box_minus = x₁ - x₂`.
- **`Quaternion`** — Unit quaternion (S³) manifold. Uses rotation-vector parameterization in the tangent space (3D) and [Markley's quaternion averaging](https://ntrs.nasa.gov/citations/20070017872) via eigendecomposition of the 4×4 outer-product matrix for the mean.

## Model Hierarchy

```
UKFModel (abstract)
  │  process(state, control, dt) → state
  │  measurement(state) → z
  │  state_mean / state_residual / apply_state_correction
  │  measurement_mean / measurement_residual
  │
  └─► SystemModel
        │  Composite manifold layout (vector of Manifold*)
        │  Automatic slicing for mean/residual/correction
        │
        ├─► RigidBodyCVModel             (constant velocity)
        ├─► RigidBodyHelicalModel         (velocity rotates with orientation)
        ├─► RigidBodyCTRVModel            (speed along body-X axis)
        ├─► RigidBodyCAModel              (constant acceleration)
        └─► RigidBodySingerModel          (exponentially correlated acceleration)

MeasurementModel (abstract)
  │  project(state) → z
  │  measurement_mean / measurement_residual
  │
  ├─► RigidBodyPoseSensorModel  (pose: position + quaternion)
  ├─► PoseSensor                (pose with configurable R)
  ├─► VelocitySensor            (linear velocity)
  └─► ProjectiveBBoxSensor      (camera 2D bounding box via 3D→2D projection)
```

### State Layout

All motion models share the same 16D state vector:

| Index | Field | Dim | Manifold | Tangent Index |
|-------|-------|-----|----------|---------------|
| 0–2 | Position (x, y, z) | 3 | Euclidean(3) | 0–2 |
| 3–5 | Velocity (vx, vy, vz) | 3 | Euclidean(3) | 3–5 |
| 6–8 | Acceleration (ax, ay, az) | 3 | Euclidean(3) | 6–8 |
| 9–12 | Orientation (qw, qx, qy, qz) | 4 | Quaternion | 9–11 |
| 13–15 | Angular velocity (ωx, ωy, ωz) | 3 | Euclidean(3) | 12–14 |

### Measurement Layout (Pose Sensor)

Default measurement model observes position + orientation (7D state-space, 6D tangent):

| Index | Field | Dim | Manifold |
|-------|-------|-----|----------|
| 0–2 | Position | 3 | Euclidean(3) |
| 3–6 | Orientation (qw, qx, qy, qz) | 4 | Quaternion |

Multi-sensor fusion allows different sensors to observe different subsets of the state (velocity, 2D bounding boxes, etc.).

## Sigma Point Schemes

The UKF supports two sigma point generation schemes:

| Scheme | Points | Weights | Flag |
|--------|--------|---------|------|
| Symmetric (default) | 2n | Uniform 1/(2n) | `scaled_sigma_points=False` |
| Merwe Scaled | 2n+1 | Separate Wm/Wc, includes mean point | `scaled_sigma_points=True` |

The Merwe scheme adds a zeroth sigma point at the mean and uses scaling parameter λ = α²(n + κ) − n. Benchmarks show ~8% faster convergence with equivalent accuracy.

## Numerical Stability

1. **Covariance symmetrization** — `P = (P + Pᵀ) / 2` after every update
2. **Eigenvalue regularization** — if `min(eigenvalues(P)) < 1e-10`, add `(1e-9 - λ_min) · I`
3. **Matrix square root selection** — SVD (robust, handles semi-definite) or LDLT (fast, requires positive-definite)
4. **Log-space likelihoods** — IMM model probability updates use log-sum-exp to avoid overflow

## Temporal State Layer

The `TrackPool` includes a temporal layer that tracks per-object timestamps
and supports two time modes:

| Mode | Method | dt Source | Use Case |
|------|--------|-----------|----------|
| Relative | `predict(dt)` | Caller provides dt | Fixed-rate simulation |
| Absolute | `predictTo(t)` | `t - mTimestamps[i]` per track | Multi-rate production |

The mode locks on the first temporal call and cannot be changed. The
`setTimeMode()` method allows pre-declaration before any temporal call.

### predictTo Design (Two-Pass)

```
Pass 1 (serial):    Validate per-track dt, enforce safety guards
Pass 2 (parallel):  OpenMP-parallel predict for tracks with dt > 0
```

Safety guards:
- **Negative dt** → throws (monotonicity violation)
- **Max dt > 10s** → throws (runaway prediction guard)
- **Zero dt** → skip (no-op for already-current tracks)

### getSnapshotAt (Non-Mutating)

`getSnapshotAt(t)` returns a `vector<StateEstimate>` computed via
each filter’s `projectTo(dt)` method. The internal filter state and
timestamps remain unchanged — useful for lookahead or visualization.

### StateEstimate

Read-only output container:

| Field | Type | Description |
|-------|------|-------------|
| `trackId` | `string` | Track identifier |
| `state` | `VectorXd` | 16D state vector |
| `covariance` | `MatrixXd` | 15×15 error covariance |
| `timestamp` | `double` | Last update time |
| `modelProbabilities` | `VectorXd` | IMM model weights |

## Source Files

| File | Contents |
|------|----------|
| [`manifolds.hpp`](../csrc/include/quak/manifolds.hpp) | `Manifold`, `Euclidean`, `Quaternion` |
| [`models.hpp`](../csrc/include/quak/models.hpp) | `UKFModel`, `SystemModel`, `MeasurementModel` |
| [`models_impl.hpp`](../csrc/include/quak/models_impl.hpp) | CV, Helical, CTRV, CA, Singer motion models |
| [`projective_model.hpp`](../csrc/include/quak/projective_model.hpp) | `ProjectiveBBoxModel`, `CameraParams` |
| [`ukf.hpp`](../csrc/include/quak/ukf.hpp) | `UKF`, `NoiseModel`, `PredictionResult` |
| [`imm.hpp`](../csrc/include/quak/imm.hpp) | `IMMEstimator` (Markov switching, `projectTo`) |
| [`types.hpp`](../csrc/include/quak/types.hpp) | `StateEstimate`, `TimeMode` |
| [`track_pool.hpp`](../csrc/include/quak/track_pool.hpp) | `TrackPool` (multi-object IMM pool, temporal layer) |
| [`bindings.cpp`](../csrc/bindings.cpp) | nanobind Python bindings |
