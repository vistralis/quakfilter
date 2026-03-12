# quakfilter

**QUA**ternion C++ **K**alman **Filter** — a high-performance Unscented Kalman Filter library in C++ with Python bindings, designed for real-time 6-DoF rigid body pose tracking.

```bash
pip install quakfilter
```

```python
import numpy as np
from quak import UKF, RigidBodyCVModel

model = RigidBodyCVModel()
x0 = np.zeros(16); x0[9] = 1.0  # identity quaternion
ukf = UKF(model, Q=np.eye(15)*0.01, R=np.eye(6)*0.001, P0=np.eye(15)*0.1, x0=x0)

ukf.predict(dt=0.01)
ukf.correct(measurement)  # 7D: [px, py, pz, qw, qx, qy, qz]
```

## Features

- **Manifold-aware UKF** — proper quaternion handling via boxPlus/boxMinus operations and Markley's quaternion mean
- **16D state / 15D error-state** — position, velocity, acceleration, unit quaternion orientation, angular velocity
- **Multiple motion models** — Constant Velocity (CV), Helical, Constant Turn-Rate-and-Velocity (CTRV), Constant Acceleration (CA), Singer maneuvering target
- **IMM Estimator** — Interacting Multiple Model with Markov switching, log-likelihood model probabilities, single-model fast path
- **TrackPool** — Multi-object IMM pool with UUID handles, active/suspended partitioning, OpenMP-parallel predict, selective correct
- **Temporal state layer** — `predictTo(timestamp)` for absolute-time sync, `getSnapshotAt(t)` for non-mutating previews, `StateEstimate` output containers, per-track timestamps
- **Multi-sensor fusion** — PoseSensor, VelocitySensor, ProjectiveBBoxSensor with `correct(z, sensor)` overload
- **Projective model** — Camera-based 2D bounding box tracking via 3D→2D projection (composable with any dynamics model)
- **Sigma point schemes** — Symmetric (2n) and Merwe Scaled (2n+1) with `scaled_sigma_points` flag
- **Python bindings** — Zero-copy interop via [nanobind](https://github.com/wjakob/nanobind) + Eigen
- **Numerical stability** — SVD/LDLT selection, log-space likelihoods, covariance symmetrization, eigenvalue regularization

## State Vector Layout

```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Field: px py pz vx vy vz ax ay az qw qx qy qz wx wy wz
       ├─ pos ─┤  ├─ vel ─┤  ├─ acc ─┤  ├── quat ──┤  ├ω─┤
       └────────────────── 16D state ───────────────────────┘
       └────────────────── 15D error (tangent) ─────────────┘
```

## Quick Start

### Build & Install

```bash
# Prerequisites: Python 3.10+, CMake 3.15+, OpenMP, Eigen3 (auto-fetched if missing)
pip install .
```

### Python Usage

```python
import numpy as np
from quak import UKF, RigidBodyCVModel, RigidBodyCTRVModel

# Choose a motion model
model = RigidBodyCVModel()  # or RigidBodyCTRVModel, RigidBodyCAModel, etc.

x0 = np.zeros(16)
x0[9] = 1.0  # identity quaternion [qw]

P0 = np.eye(15) * 0.1
Q  = np.eye(15) * 0.01
R  = np.eye(6)  * 0.001

ukf = UKF(model, Q, R, P0, x0)

# Predict-correct loop
ukf.predict(dt=0.01)
ukf.correct(measurement)  # 7D: [px, py, pz, qw, qx, qy, qz]

state = ukf.state       # 16D state estimate
P     = ukf.covariance  # 15×15 error covariance
```

### Multi-Sensor Fusion

```python
from quak import PoseSensor, VelocitySensor

pose_sensor = PoseSensor(R_pose)
vel_sensor  = VelocitySensor(R_vel)

ukf.predict(dt=0.01)
ukf.correct(z_pose, sensor=pose_sensor)
ukf.correct(z_vel, sensor=vel_sensor)
```

### IMM Usage

```python
from quak import IMMEstimator, RigidBodyCVModel, RigidBodyCTRVModel

cv   = RigidBodyCVModel()
ctrv = RigidBodyCTRVModel()

imm = IMMEstimator([cv, ctrv], [Q_cv, Q_ctrv], R, P0, x0)

imm.predict(dt=0.01)
imm.correct(measurement)

probabilities = imm.model_probabilities  # [p_cv, p_ctrv]
```

### TrackPool Usage

```python
from quak import TrackPool, TimeMode, RigidBodyCVModel, RigidBodyCTRVModel

pool = TrackPool([RigidBodyCVModel(), RigidBodyCTRVModel()],
                 [Q_cv, Q_ctrv], R, P0)

pool.add("track_0", initial_state, time=0.0)  # every add requires a timestamp
pool.predict(dt=0.01)                         # all tracks in parallel
pool.correct(["track_0"], z)                  # selective update
pool.suspend("track_0")                       # skip from predict
pool.resume("track_0")                        # re-activate
pool.remove("track_0")                        # delete track
```

### Temporal Operations

```python
# Absolute-time mode: sync all tracks to a wall-clock timestamp
pool.add("track_a", state_a, time=100.0)
pool.add("track_b", state_b, time=102.0)
pool.predict_to(105.0)  # A gets dt=5, B gets dt=3

# Non-mutating preview at a future time
snapshot = pool.get_snapshot_at(106.0)  # list of StateEstimate
for est in snapshot:
    print(est.track_id, est.state[:3], est.timestamp)
```

## Architecture

```
csrc/include/quak/
├── models.hpp          UKFModel / SystemModel / MeasurementModel base classes
├── models_impl.hpp     CV, Helical, CTRV, CA, Singer motion models
├── projective_model.hpp  Projective bounding box measurement model
├── manifolds.hpp       Euclidean + Quaternion manifold operations
├── ukf.hpp             Core UKF (predict/correct, LDLT+SVD, workspaces)
├── imm.hpp             IMM estimator (Markov switching, log-likelihood)
└── track_pool.hpp      Multi-object IMM pool (TrackPool)
csrc/bindings.cpp       nanobind Python bindings
```

## Coding Style

This project follows the [Vistralis C++ Coding Style](docs/coding-style.md):

- **Allman braces**, 4-space indent, 100-char line limit
- **camelCase** methods, **mCamelCase** members, **PascalCase** classes
- **No abbreviations** — `measurement` not `meas`, `stateSize` not `stateSz`
- **Numerical stability** — LDLT/SVD solvers, log-space likelihoods, eigenvalue regularization
- Formatting: `clang-format` (C++), `ruff` (Python)

## Documentation

| Topic | Link |
|-------|------|
| System Architecture | [docs/architecture.md](docs/architecture.md) |
| UKF Reference | [docs/ukf-reference.md](docs/ukf-reference.md) |
| Q/R Tuning Guide | [docs/tuning.md](docs/tuning.md) |
| Tutorials & Examples | [docs/tutorials.md](docs/tutorials.md) |
| Coding Style | [docs/coding-style.md](docs/coding-style.md) |

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Vistralis Labs. All rights reserved.
