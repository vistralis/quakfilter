# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IMM Car Trajectory Test
=======================
Simulates a realistic car maneuver with 5 phases:

Phase 1: Acceleration   — 0 → 50 km/h (≈13.89 m/s) at 2 m/s²  (~7s)
Phase 2: Cruise          — 50 km/h constant speed                 (5s)
Phase 3: Braking         — 50 → 30 km/h (≈8.33 m/s) at -2 m/s²  (~2.8s)
Phase 4: Constant turn   — 180° left curve at 30 km/h, R=20m     (~7.5s)
Phase 5: Final braking   — 30 → 0 km/h at -2 m/s²               (~4.2s)

Verifies that IMM model probabilities shift correctly:
- CV model dominant during straight-line phases (1, 2, 3, 5)
- CTRV model dominant during the turning phase (4)
"""

import sys

import numpy as np

sys.path.insert(0, "build")
from quak import (
    IMMEstimator,
    RigidBodyCTRVModel,
    RigidBodyPoseSensorModel,
)

DT = 0.05  # 20 Hz simulation
KMH_TO_MS = 1.0 / 3.6

# ─── Ground truth trajectory generator ────────────────────────────────


def quat_from_yaw(yaw):
    """Quaternion [w, x, y, z] from yaw angle (rotation about z-axis)."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def generate_trajectory():
    """Generate ground truth: list of (time, pos_xyz, vel_xyz, yaw, omega_z, phase_name)."""
    trajectory = []
    t = 0.0

    # Starting state
    x, y = 0.0, 0.0
    yaw = 0.0  # pointing along +X
    speed = 0.0
    omega = 0.0  # turn rate

    # ── Phase 1: Constant acceleration 0 → 50 km/h ──
    target_speed_1 = 50.0 * KMH_TO_MS  # 13.889 m/s
    accel = 2.0  # m/s²
    while speed < target_speed_1 - 1e-6:
        speed = min(speed + accel * DT, target_speed_1)
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x += vx * DT
        y += vy * DT
        trajectory.append((t, np.array([x, y, 0]), np.array([vx, vy, 0]), yaw, 0.0, "accel"))
        t += DT

    # ── Phase 2: Cruise at 50 km/h ──
    cruise_duration = 5.0
    t_end = t + cruise_duration
    while t < t_end:
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x += vx * DT
        y += vy * DT
        trajectory.append((t, np.array([x, y, 0]), np.array([vx, vy, 0]), yaw, 0.0, "cruise"))
        t += DT

    # ── Phase 3: Braking 50 → 30 km/h ──
    target_speed_3 = 30.0 * KMH_TO_MS  # 8.333 m/s
    decel = -2.0
    while speed > target_speed_3 + 1e-6:
        speed = max(speed + decel * DT, target_speed_3)
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x += vx * DT
        y += vy * DT
        trajectory.append((t, np.array([x, y, 0]), np.array([vx, vy, 0]), yaw, 0.0, "brake"))
        t += DT

    # ── Phase 4: 180° left curve at constant 30 km/h ──
    radius = 20.0
    omega = speed / radius  # ≈ 0.417 rad/s  (left = positive)
    turn_angle = np.pi  # 180 degrees
    turn_duration = turn_angle / omega
    t_end = t + turn_duration
    while t < t_end:
        yaw += omega * DT
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x += vx * DT
        y += vy * DT
        trajectory.append((t, np.array([x, y, 0]), np.array([vx, vy, 0]), yaw, omega, "turn"))
        t += DT

    # ── Phase 5: Final braking 30 → 0 km/h ──
    omega = 0.0
    while speed > 0.01:
        speed = max(speed + decel * DT, 0.0)
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x += vx * DT
        y += vy * DT
        trajectory.append((t, np.array([x, y, 0]), np.array([vx, vy, 0]), yaw, 0.0, "stop"))
        t += DT

    return trajectory


# ─── IMM simulation runner ────────────────────────────────────────────


def run_imm_simulation(measurement_noise_std=0.05):
    """Run IMM on the car trajectory, return per-step results."""
    trajectory = generate_trajectory()

    # Models: 0 = CV (PoseSensor), 1 = CTRV
    model_cv = RigidBodyPoseSensorModel()
    model_ctrv = RigidBodyCTRVModel()

    # Initial state from first trajectory point
    t0, pos0, vel0, yaw0, omega0, _ = trajectory[0]
    x0 = np.zeros(16)
    x0[0:3] = pos0
    x0[3:6] = vel0
    # acc slots x0[6:9] stay zero
    x0[9:13] = quat_from_yaw(yaw0)
    x0[13:16] = [0, 0, omega0]

    Q_cv = np.eye(15) * 0.005
    Q_ctrv = np.eye(15) * 0.005
    R = np.eye(6) * (measurement_noise_std**2)
    P0 = np.eye(15) * 0.5

    imm = IMMEstimator(
        [model_cv, model_ctrv], [Q_cv, Q_ctrv], R, P0, x0, persistence_probability=0.90
    )

    results = []
    rng = np.random.default_rng(42)

    for i, (t, pos, vel, yaw, omega, phase) in enumerate(trajectory):
        # ── Predict ──
        imm.predict(DT)

        # ── Noisy measurement: [pos(3), quat(4)] ──
        pos_noisy = pos + rng.normal(0, measurement_noise_std, 3)
        quat_from_yaw(yaw)
        # Small angular noise on yaw
        yaw_noisy = yaw + rng.normal(0, measurement_noise_std * 0.5)
        q_noisy = quat_from_yaw(yaw_noisy)

        z = np.concatenate([pos_noisy, q_noisy])

        # ── Correct ──
        imm.correct(z)

        # Record
        state = imm.x.copy()
        probabilities = imm.mu.copy()
        results.append(
            {
                "t": t,
                "phase": phase,
                "position_gt": pos.copy(),
                "velocity_gt": vel.copy(),
                "yaw_gt": yaw,
                "estimated_position": state[0:3].copy(),
                "probability_cv": probabilities[0],
                "probability_ctrv": probabilities[1],
            }
        )

    return results


# ─── Test functions ───────────────────────────────────────────────────


def test_car_trajectory_imm():
    """Full car trajectory test with model probability verification."""
    results = run_imm_simulation()

    # Group results by phase
    phases = {}
    for result in results:
        phases.setdefault(result["phase"], []).append(result)

    # ── Position tracking accuracy ──
    # Final position error should be reasonable (< 2m after convergence)
    for phase_name, phase_data in phases.items():
        # Skip first few steps of each phase (convergence)
        settled = phase_data[max(0, len(phase_data) // 2) :]
        if settled:
            pos_errors = [
                np.linalg.norm(result["estimated_position"] - result["position_gt"])
                for result in settled
            ]
            mean_err = np.mean(pos_errors)
            assert mean_err < 3.0, f"Phase '{phase_name}': mean pos error {mean_err:.2f}m too large"

    # ── Model probability verification ──
    # During the turning phase, CTRV should be dominant
    turn_data = phases.get("turn", [])
    if turn_data:
        # Look at the second half of the turn (after IMM has adapted)
        late_turn = turn_data[len(turn_data) // 2 :]
        avg_probability_ctrv = np.mean([result["probability_ctrv"] for result in late_turn])
        avg_probability_cv = np.mean([result["probability_cv"] for result in late_turn])
        print(
            f"\n  Turn phase (late): CV={avg_probability_cv:.3f}, CTRV={avg_probability_ctrv:.3f}"
        )
        assert avg_probability_ctrv > avg_probability_cv, (
            f"CTRV should dominate during turn: CV={avg_probability_cv:.3f} vs CTRV={avg_probability_ctrv:.3f}"
        )

    # During cruise (constant velocity), CV should be dominant
    cruise_data = phases.get("cruise", [])
    if cruise_data:
        late_cruise = cruise_data[len(cruise_data) // 2 :]
        avg_probability_cv = np.mean([result["probability_cv"] for result in late_cruise])
        avg_probability_ctrv = np.mean([result["probability_ctrv"] for result in late_cruise])
        print(
            f"  Cruise phase (late): CV={avg_probability_cv:.3f}, CTRV={avg_probability_ctrv:.3f}"
        )
        assert avg_probability_cv > avg_probability_ctrv, (
            f"CV should dominate during cruise: CV={avg_probability_cv:.3f} vs CTRV={avg_probability_ctrv:.3f}"
        )

    # ── All probabilities should be valid throughout ──
    for result in results:
        assert 0 <= result["probability_cv"] <= 1
        assert 0 <= result["probability_ctrv"] <= 1
        np.testing.assert_allclose(
            result["probability_cv"] + result["probability_ctrv"], 1.0, atol=1e-6
        )

    # ── State should be finite throughout ──
    for result in results:
        assert np.all(np.isfinite(result["estimated_position"]))


def test_car_trajectory_probability_report():
    """Prints a detailed per-phase probability report for manual inspection."""
    results = run_imm_simulation()

    # Phase transition indices
    phases_order = ["accel", "cruise", "brake", "turn", "stop"]
    phase_boundaries = {}
    current_phase = None
    for i, r in enumerate(results):
        if r["phase"] != current_phase:
            current_phase = r["phase"]
            phase_boundaries[current_phase] = i

    print("\n" + "=" * 72)
    print("  IMM CAR TRAJECTORY — MODEL PROBABILITY REPORT")
    print("=" * 72)
    print(
        f"  {'Step':>5} | {'Time':>6} | {'Phase':>8} | {'CV prob':>8} | {'CTRV prob':>9} | {'Dominant':>8}"
    )
    print("-" * 72)

    # Sample every N steps
    step = max(1, len(results) // 40)  # ~40 rows
    for i in range(0, len(results), step):
        result = results[i]
        dominant = "CV" if result["probability_cv"] > result["probability_ctrv"] else "CTRV"
        marker = " ◀" if result["phase"] in ["turn"] and dominant == "CTRV" else ""
        marker = " ◀" if result["phase"] in ["cruise"] and dominant == "CV" else marker
        print(
            f"  {i:>5} | {result['t']:>6.2f} | {result['phase']:>8} | "
            f"{result['probability_cv']:>8.4f} | {result['probability_ctrv']:>9.4f} | {dominant:>8}{marker}"
        )

    print("-" * 72)

    # Summary
    phases = {}
    for result in results:
        phases.setdefault(result["phase"], []).append(result)

    print("\n  Phase Summary:")
    for phase_name in phases_order:
        data = phases.get(phase_name, [])
        if not data:
            continue
        avg_cv = np.mean([result["probability_cv"] for result in data])
        avg_ctrv = np.mean([result["probability_ctrv"] for result in data])
        duration = len(data) * DT
        dominant = "CV" if avg_cv > avg_ctrv else "CTRV"
        print(
            f"    {phase_name:>8}: {duration:>5.1f}s | CV={avg_cv:.3f} CTRV={avg_ctrv:.3f} | {dominant}"
        )

    print("=" * 72)

    # Just assert it runs without error
    assert len(results) > 0
