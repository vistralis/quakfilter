# Tutorials

Step-by-step examples demonstrating the core quakfilter API. Each example
is a self-contained script with noisy simulation, validation checks, and
detailed comments explaining every step.

Run any example with:

```bash
python examples/<name>.py
```

---

## UKF — Linear Motion

**File**: [`ukf_linear.py`](../examples/ukf_linear.py)

Tracks an object moving in a straight line at constant velocity using a
bare `UKF` with the `RigidBodyCVModel`. This is the simplest possible
setup — a single motion model and a pose sensor.

**What you'll learn**:
- Creating a `UKF` with process noise `Q`, measurement noise `R`, and initial state
- The predict-correct loop
- Reading the state via the `.state` property
- Verifying convergence: position error shrinks over time

**Key state vector indices**:
| Index | Field | Meaning |
|-------|-------|---------|
| 0–2 | px, py, pz | Position |
| 3–5 | vx, vy, vz | Velocity |
| 6–8 | ax, ay, az | Acceleration |
| 9–12 | qw, qx, qy, qz | Orientation quaternion |
| 13–15 | wx, wy, wz | Angular velocity |

---

## UKF — Curvilinear Motion (CTRV)

**File**: [`ukf_curvilinear.py`](../examples/ukf_curvilinear.py)

Tracks an object moving in a circle using the `RigidBodyCTRVModel`
(Constant Turn-Rate and Velocity). The CTRV model naturally handles
circular motion by maintaining the turn rate as part of the state.

**What you'll learn**:
- Using `RigidBodyCTRVModel` for curved trajectories
- Setting initial angular velocity (`state[15] = ω`)
- Verifying orbit radius and turn rate estimation

**Trajectory**: Circle of radius 6 m at 3 m/s, turn rate 0.5 rad/s.

---

## UKF — Tag Orientation (Figure-8 with 6-DoF)

**File**: [`ukf_tag_orientation.py`](../examples/ukf_tag_orientation.py)

Simulates a tag (e.g. an AprilTag on a palm) tracing a figure-8
lemniscate while continuously flipping orientation — like a hand doing
figure-8 motions with the palm alternately facing up and down.

**What you'll learn**:
- Using `RigidBodyHelicalModel` for coordinated translation + rotation
- Setting initial angular velocity for continuous orientation change
- Tracking both position and quaternion orientation simultaneously
- Verifying sub-degree orientation accuracy

**Trajectory**: 30 cm amplitude lemniscate, one full palm flip per cycle.

---

## IMM — Single Model Baseline

**File**: [`imm_single_model.py`](../examples/imm_single_model.py)

Wraps a single `RigidBodyCVModel` inside an `IMMEstimator` to show that
the IMM degrades gracefully to a plain UKF when given only one model.
Useful as a baseline before adding more models.

**What you'll learn**:
- `IMMEstimator` constructor with `models=[...]` and `process_noises=[...]`
- Keyword arguments: `measurement_noise`, `initial_covariance`, `initial_state`
- Accessing `.state`, `.model_probabilities`, `.covariance`
- Single-model probability is always `1.0`

---

## IMM — Model Switching (CV + CA)

**File**: [`imm_model_switching.py`](../examples/imm_model_switching.py)

Runs a Constant Velocity and Constant Acceleration model in parallel via the
IMM. The object cruises at constant speed, then accelerates. The IMM
automatically fuses both models, weighting each by how well it predicts
the current measurements.

**What you'll learn**:
- Configuring two models with different process noises
- Using `p_same` for the Markov transition probability
- Reading per-model states via `.model_state(i)` and `.model_covariance(i)`
- Observing model probability evolution across regime changes

**Trajectory**: 2 m/s cruise → 1 m/s² acceleration onset.

---

## TrackPool — Multi-Object Tracking

**File**: [`track_pool.py`](../examples/track_pool.py)

Manages three objects simultaneously using `TrackPool`. Each object gets
its own IMM estimator internally. The pool handles parallel prediction
(OpenMP) and selective per-track correction.

**What you'll learn**:
- Creating a `TrackPool` with shared models and noise parameters
- Adding tracks with `pool.add(uuid, state, time)` — time is always required
- Batch predict via `pool.predict(dt)`
- Per-track correct via `pool.correct([uuid], measurements)`
- Bulk state access via `pool.states` and `pool.track_ids`
- Lifecycle operations: `suspend()`, `resume()`, `remove()`

---

## Temporal Operations — predictTo & getSnapshotAt

The temporal layer adds absolute-time awareness to `TrackPool`. Instead of
passing a relative `dt`, you provide wall-clock timestamps and the pool
computes per-track dt values automatically.

### Two Time Modes

| Mode | API | Use Case |
|------|-----|----------|
| **Relative** | `pool.predict(dt)` | Simulation, fixed-rate loops |
| **Absolute** | `pool.predict_to(timestamp)` | Production, multi-rate sensors |

The mode locks automatically on the first temporal call. You can also
pre-declare it with `pool.set_time_mode(TimeMode.Absolute)`.

### Absolute-Time Workflow

```python
from quak import TrackPool, TimeMode, RigidBodyCVModel

pool = TrackPool([RigidBodyCVModel()], [Q], R, P0)

# Tracks arrive at different times
pool.add("car_A", state_a, time=100.0)
pool.add("car_B", state_b, time=102.0)

# Sync all tracks to t=105 — A gets dt=5, B gets dt=3
pool.predict_to(105.0)

# Non-mutating preview: what would the states be at t=106?
snapshot = pool.get_snapshot_at(106.0)
for est in snapshot:
    print(f"{est.track_id}: pos={est.state[:3]}, t={est.timestamp}")
    # Internal filters are unchanged!
```

### StateEstimate

Both `get_estimate()` and `get_snapshot_at()` return `StateEstimate` objects:

```python
est = pool.get_estimate("car_A")
est.track_id            # "car_A"
est.state               # 16D state vector
est.covariance          # 15×15 error covariance
est.timestamp           # last update time
est.model_probabilities # IMM model weights
```

### Safety Guards

- **Negative dt** — `predict_to(past_time)` throws `RuntimeError`
- **Max dt** — prevents runaway predictions (default: 10s max)
- **Mode mixing** — `predict(dt)` after `predict_to()` throws `RuntimeError`
