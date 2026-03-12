#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tutorial 05 — TrackPool for Multi-Object Tracking.

Demonstrates the TrackPool managing multiple objects simultaneously.
Three objects move in different patterns:
    Object A: straight line along +X at 3 m/s
    Object B: straight line along +Y at 2 m/s
    Object C: diagonal line at 1 m/s along both +X and +Y

Each object gets its own IMM estimator inside the pool, with
independent predict/correct cycles and UUID-based handle access.
"""

# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

import numpy as np

from quak import RigidBodyCTRVModel, RigidBodyCVModel, TrackPool

# ── Step 1: Create a TrackPool ───────────────────────────────────────────

models = [RigidBodyCVModel(), RigidBodyCTRVModel()]
Q_list = [np.eye(15) * 0.01, np.eye(15) * 0.01]
R = np.eye(6) * 0.1
P0 = np.eye(15) * 1.0

pool = TrackPool(models, Q_list, R, P0, persistence_probability=0.95)

# ── Step 2: Add three objects with different velocities ──────────────────


def make_state(vx, vy, vz=0.0):
    """Create an initial state with given velocity."""
    x = np.zeros(16)
    x[3] = vx
    x[4] = vy
    x[5] = vz
    x[9] = 1.0  # identity quaternion
    return x


# Generate UUID handles for each track
uuid_a = str(uuid.uuid4())
uuid_b = str(uuid.uuid4())
uuid_c = str(uuid.uuid4())

pool.add(uuid_a, make_state(3.0, 0.0), 0.0)  # moving along +X
pool.add(uuid_b, make_state(0.0, 2.0), 0.0)  # moving along +Y
pool.add(uuid_c, make_state(1.0, 1.0), 0.0)  # diagonal

assert pool.active_count == 3, "Should have 3 active tracks"
print(f"  Track A: {uuid_a}")
print(f"  Track B: {uuid_b}")
print(f"  Track C: {uuid_c}")

# ── Step 3: Simulate and track ───────────────────────────────────────────

dt = 0.05
num_steps = 80  # 4 seconds

np.random.seed(42)
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

velocities = {
    uuid_a: np.array([3.0, 0.0, 0.0]),
    uuid_b: np.array([0.0, 2.0, 0.0]),
    uuid_c: np.array([1.0, 1.0, 0.0]),
}

for step in range(num_steps):
    time = (step + 1) * dt

    # Predict all tracks in parallel (uses OpenMP internally)
    pool.predict(dt)

    # Correct each track with its own measurement (batch of 1)
    for track_id, velocity in velocities.items():
        true_position = velocity * time
        noise = np.random.randn(3) * 0.1
        measurement = np.concatenate([true_position + noise, identity_quat])
        # correct() takes a list of track IDs and a (K, meas_dim) array
        pool.correct([track_id], measurement.reshape(1, -1))

# ── Step 4: Validate results ─────────────────────────────────────────────

final_time = num_steps * dt

# Read all states at once (N, 16) array
all_states = pool.states
track_ids = pool.track_ids

for name, track_id, velocity in [
    ("A (+X)", uuid_a, velocities[uuid_a]),
    ("B (+Y)", uuid_b, velocities[uuid_b]),
    ("C (diag)", uuid_c, velocities[uuid_c]),
]:
    # Find this track's row in the states matrix
    idx = track_ids.index(track_id)
    state = all_states[idx]
    true_final = velocity * final_time

    position_error = np.linalg.norm(state[:3] - true_final)
    velocity_error = np.linalg.norm(state[3:6] - velocity)

    assert position_error < 0.5, f"Track {name} position error: {position_error:.3f} m"
    assert velocity_error < 1.0, f"Track {name} velocity error: {velocity_error:.3f} m/s"

    print(f"  Track {name}: pos_err={position_error:.4f} m, vel_err={velocity_error:.4f} m/s")

# ── Step 5: Test suspend and resume ──────────────────────────────────────

pool.suspend(uuid_b)
assert pool.active_count == 2, "Should have 2 active after suspend"

# Predict only affects active tracks
pool.predict(dt)

# Resume
pool.resume(uuid_b)
assert pool.active_count == 3, "Should have 3 active after resume"

# ── Step 6: Test removal ─────────────────────────────────────────────────

pool.remove(uuid_c)
assert pool.active_count == 2, "Should have 2 after removing track C"

# Bulk state access
states = pool.states
assert states.shape[0] == 2, f"States shape should be (2, 16), got {states.shape}"
assert states.shape[1] == 16

print("  Suspend/resume: ✓")
print("  Remove track:   ✓")
print(f"  Bulk states shape: {states.shape}")
print("Tutorial 05 — TrackPool Multi-Object Tracking")
print("  ✓ All checks passed")
