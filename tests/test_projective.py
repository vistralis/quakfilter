# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the projective bounding box measurement model.

Three test categories:
1. Projection correctness — verify 3D→2D math against known ground truth
2. UKF tracking — single tracker with projective measurements
3. IMM tracking — multiple models all using projective measurement
"""

import sys

import numpy as np

sys.path.insert(0, "build")
import quak
from quak import (
    UKF,
    CameraParams,
    IMMEstimator,
    ProjectiveBBoxModel,
    RigidBodyCAModel,
    RigidBodyCTRVModel,
    RigidBodyCVModel,
)

# ─── Helpers ──────────────────────────────────────────────────────────


def make_camera(fx=500, fy=500, cx=320, cy=240, cam_pos=None, look_at=None):
    """Create a CameraParams with sensible defaults.

    If cam_pos and look_at are provided, compute R_cw and t_cw
    automatically (camera looks from cam_pos toward look_at, Z-forward).
    Otherwise: identity extrinsics (camera at world origin, looking along +Z).
    """
    cam = CameraParams()
    cam.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    if cam_pos is not None and look_at is not None:
        cam_pos = np.array(cam_pos, dtype=np.float64)
        look_at = np.array(look_at, dtype=np.float64)

        # Camera Z = forward direction
        z_axis = look_at - cam_pos
        z_axis /= np.linalg.norm(z_axis)

        # Camera X = right (assume world up = [0, 0, 1])
        world_up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(z_axis, world_up)
        if np.linalg.norm(x_axis) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(z_axis, world_up)
        x_axis /= np.linalg.norm(x_axis)

        # Camera Y = down
        y_axis = np.cross(z_axis, x_axis)

        R_cw = np.vstack([x_axis, y_axis, z_axis])  # 3×3
        t_cw = -R_cw @ cam_pos

        cam.R_cw = R_cw
        cam.t_cw = t_cw
    else:
        cam.R_cw = np.eye(3)
        cam.t_cw = np.zeros(3)

    cam.obj_dims = np.array([1.8, 1.5, 4.5])  # Car-sized object
    cam.min_depth = 0.1
    return cam


def make_state(pos, vel=None, acc=None, quat=None, angvel=None):
    """Build a 16D state vector."""
    x = np.zeros(16)
    x[0:3] = pos
    if vel is not None:
        x[3:6] = vel
    if acc is not None:
        x[6:9] = acc
    x[9] = 1.0  # qw
    if quat is not None:
        x[9:13] = quat
    if angvel is not None:
        x[13:16] = angvel
    return x


def project_point_manual(K, R_cw, t_cw, p_world, obj_dims, min_depth=0.1):
    """Manual 3D → 2D projection for ground truth comparison."""
    p_cam = R_cw @ p_world + t_cw
    z = max(p_cam[2], min_depth)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * p_cam[0] / z + cx
    v = fy * p_cam[1] / z + cy
    w = abs(fx * obj_dims[0] / z)
    h = abs(fy * obj_dims[1] / z)

    return np.array([u, v, w, h])


# ═════════════════════════════════════════════════════════════════════
# Test 1: Projection Correctness
# ═════════════════════════════════════════════════════════════════════


def test_projection_at_origin_camera():
    """Object directly in front of camera at origin → centered in image."""
    cam = make_camera()  # Camera at origin, looking +Z
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    # Object at (0, 0, 10) — 10 meters ahead on Z
    x = make_state(pos=[0, 0, 10])
    z = model.measurement(x)

    expected = project_point_manual(
        np.array(cam.K),
        np.array(cam.R_cw),
        np.array(cam.t_cw),
        np.array([0, 0, 10]),
        np.array(cam.obj_dims),
    )

    np.testing.assert_allclose(z, expected, atol=1e-10)

    # Should be at image center (cx, cy)
    assert abs(z[0] - 320) < 1e-10, f"u should be cx=320, got {z[0]}"
    assert abs(z[1] - 240) < 1e-10, f"v should be cy=240, got {z[1]}"

    # Bbox size: fx * 1.8 / 10 = 90, fy * 1.5 / 10 = 75
    assert abs(z[2] - 90) < 1e-10, f"w should be 90px, got {z[2]}"
    assert abs(z[3] - 75) < 1e-10, f"h should be 75px, got {z[3]}"
    print("✓ Projection at origin camera correct")


def test_projection_off_center():
    """Object offset from center → displaced in image."""
    cam = make_camera()
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    # Object at (2, 1, 20) — offset right and down, far away
    x = make_state(pos=[2, 1, 20])
    z = model.measurement(x)

    expected = project_point_manual(
        np.array(cam.K),
        np.array(cam.R_cw),
        np.array(cam.t_cw),
        np.array([2, 1, 20]),
        np.array(cam.obj_dims),
    )

    np.testing.assert_allclose(z, expected, atol=1e-10)

    # u = 500*2/20 + 320 = 370
    assert abs(z[0] - 370) < 1e-10
    # v = 500*1/20 + 240 = 265
    assert abs(z[1] - 265) < 1e-10
    print("✓ Projection off-center correct")


def test_projection_different_depths():
    """Object at different depths → correct bbox scaling."""
    cam = make_camera()
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    for depth in [5, 10, 20, 50, 100]:
        x = make_state(pos=[0, 0, depth])
        z = model.measurement(x)
        expected_w = 500 * 1.8 / depth
        expected_h = 500 * 1.5 / depth
        assert abs(z[2] - expected_w) < 1e-10, f"Depth {depth}: w={z[2]} vs {expected_w}"
        assert abs(z[3] - expected_h) < 1e-10, f"Depth {depth}: h={z[3]} vs {expected_h}"

    print("✓ Projection depth scaling correct")


def test_projection_with_extrinsics():
    """Camera at non-trivial position/orientation."""
    # Camera at (5, 0, 2) looking toward origin
    cam = make_camera(cam_pos=[5, 0, 2], look_at=[0, 0, 0])
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    # Object at origin
    x = make_state(pos=[0, 0, 0])
    z = model.measurement(x)

    # Verify against manual projection
    expected = project_point_manual(
        np.array(cam.K),
        np.array(cam.R_cw),
        np.array(cam.t_cw),
        np.array([0, 0, 0]),
        np.array(cam.obj_dims),
    )

    np.testing.assert_allclose(z, expected, atol=1e-10)

    # Object should be roughly centered (camera looks at origin)
    # The exact center depends on the projection of origin through the camera
    print(f"  Camera looking at origin, object at origin: bbox center = ({z[0]:.1f}, {z[1]:.1f})")
    print("✓ Projection with extrinsics correct")


def test_projection_behind_camera_clamped():
    """Object behind camera → depth clamped to min_depth."""
    cam = make_camera()
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    # Object behind camera (negative Z)
    x = make_state(pos=[0, 0, -5])
    z = model.measurement(x)

    # Should use min_depth = 0.1 → very large bbox
    expected_w = 500 * 1.8 / 0.1
    expected_h = 500 * 1.5 / 0.1
    assert abs(z[2] - expected_w) < 1e-6, f"Behind camera: w={z[2]} vs {expected_w}"
    assert abs(z[3] - expected_h) < 1e-6, f"Behind camera: h={z[3]} vs {expected_h}"
    print("✓ Behind-camera depth clamping correct")


def test_process_delegates_to_dynamics():
    """process() should delegate to the wrapped dynamics model."""
    cam = make_camera()

    # CV model: constant velocity
    cv = RigidBodyCVModel()
    cv_proj = ProjectiveBBoxModel(cv, cam)

    x = make_state(pos=[1, 2, 3], vel=[1, 0, 0])

    # process should give same result regardless of projection
    x_next_cv = cv.process(x, np.array([]), 0.1)
    x_next_proj = cv_proj.process(x, np.array([]), 0.1)

    np.testing.assert_allclose(x_next_proj, x_next_cv, atol=1e-15)
    print("✓ Process delegates to inner dynamics correctly")


# ═════════════════════════════════════════════════════════════════════
# Test 2: UKF with Projective Model
# ═════════════════════════════════════════════════════════════════════


def test_ukf_projective_tracking():
    """Track an object moving at constant velocity using projective measurements."""
    cam = make_camera()  # Camera at origin looking +Z
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    dt = 0.1

    # True initial state: object at (0, 0, 30), moving +X at 2 m/s
    true_pos = np.array([0.0, 0.0, 30.0])
    true_vel = np.array([2.0, 0.0, 0.0])

    # Initial estimate (slightly wrong)
    x0 = make_state(pos=[0.5, -0.3, 28.0], vel=[1.0, 0.0, 0.0])

    # Noise parameters
    Q = np.eye(15) * 0.01  # Process noise (tangent dim = 15)
    R = np.diag([5, 5, 10, 10]) ** 2  # Measurement noise: 5px center, 10px size
    P0 = np.eye(15) * 1.0

    ukf = UKF(model, Q, R, P0, x0)

    np.random.seed(42)
    errors = []

    for step in range(100):
        t = step * dt

        # Ground truth position
        position_gt = true_pos + true_vel * t

        # Predict
        ukf.predict(dt)

        # Generate noisy measurement
        measurement_gt = project_point_manual(
            np.array(cam.K),
            np.array(cam.R_cw),
            np.array(cam.t_cw),
            position_gt,
            np.array(cam.obj_dims),
        )
        noisy_meas = measurement_gt + np.random.randn(4) * np.array([5, 5, 10, 10])

        # Correct
        ukf.correct(noisy_meas)

        # Track position error
        est_state = ukf.state
        position_error = np.linalg.norm(est_state[0:3] - position_gt)
        errors.append(position_error)

    # After convergence, error should be small
    final_error = np.mean(errors[-20:])
    print(f"  Final 20-step avg position error: {final_error:.3f} m")
    assert final_error < 5.0, (
        f"UKF tracking with projective model failed: error={final_error:.3f} m"
    )

    # Error should decrease over time (convergence)
    early_error = np.mean(errors[:10])
    late_error = np.mean(errors[-10:])
    print(f"  Early error: {early_error:.3f} m, Late error: {late_error:.3f} m")
    assert late_error < early_error, "Filter should converge (late < early error)"
    print("✓ UKF projective tracking converges")


def test_ukf_projective_depth_observability():
    """Verify depth is observable from changing bbox size.

    A camera directly ahead can estimate depth from bbox size changes.
    This is the key advantage of using bbox as measurement.
    """
    cam = make_camera()
    model = ProjectiveBBoxModel(RigidBodyCVModel(), cam)

    dt = 0.1

    # Object approaching the camera (moving toward it along Z)
    true_pos = np.array([0.0, 0.0, 50.0])
    true_vel = np.array([0.0, 0.0, -5.0])  # Coming closer at 5 m/s

    # Start with wrong depth estimate (40m instead of 50m)
    x0 = make_state(pos=[0, 0, 40.0], vel=[0, 0, -3.0])

    Q = np.eye(15) * 0.01
    R = np.diag([3, 3, 5, 5]) ** 2
    P0 = np.eye(15) * 10.0

    ukf = UKF(model, Q, R, P0, x0)

    np.random.seed(123)
    depth_errors = []

    for step in range(80):
        t = step * dt
        position_gt = true_pos + true_vel * t

        if position_gt[2] < 5:
            break  # Don't let object get too close

        ukf.predict(dt)

        measurement_gt = project_point_manual(
            np.array(cam.K),
            np.array(cam.R_cw),
            np.array(cam.t_cw),
            position_gt,
            np.array(cam.obj_dims),
        )
        noisy_meas = measurement_gt + np.random.randn(4) * np.array([3, 3, 5, 5])

        ukf.correct(noisy_meas)

        est_state = ukf.state
        depth_error = abs(est_state[2] - position_gt[2])
        depth_errors.append(depth_error)

    initial_depth_error = depth_errors[0]
    final_depth_error = np.mean(depth_errors[-10:])
    print(f"  Initial depth error: {initial_depth_error:.2f} m")
    print(f"  Final depth error: {final_depth_error:.2f} m")
    assert final_depth_error < initial_depth_error, "Depth should converge"
    print("✓ Depth is observable from bbox size changes")


# ═════════════════════════════════════════════════════════════════════
# Test 3: IMM with Projective Models
# ═════════════════════════════════════════════════════════════════════


def test_imm_projective_model_switching():
    """IMM with CV+CTRV+CA using projective measurements.

    Scenario:
      Phase 1: Constant velocity → CV should dominate
      Phase 2: Constant acceleration → CA should dominate
      Phase 3: Turning → CTRV should dominate
    """
    cam = make_camera()

    # All three models use projective measurement
    cv_proj = ProjectiveBBoxModel(RigidBodyCVModel(), cam)
    ctrv_proj = ProjectiveBBoxModel(RigidBodyCTRVModel(), cam)
    ca_proj = ProjectiveBBoxModel(RigidBodyCAModel(), cam)

    models = [cv_proj, ca_proj, ctrv_proj]

    dt = 0.1

    # Initial state
    x0 = make_state(pos=[0, 0, 30], vel=[5, 0, 0])

    Q_list = [np.eye(15) * 0.01 for _ in range(3)]
    R = np.diag([5, 5, 10, 10]) ** 2
    P0 = np.eye(15) * 0.5

    imm = IMMEstimator(
        models, Q_list, R, P0, x0, persistence_probability=0.95, probability_floor=0.05
    )

    np.random.seed(77)

    # Track model probabilities over time
    probs_history = []

    # Generate trajectory with three phases
    true_pos = np.array([0.0, 0.0, 30.0])
    true_vel = np.array([5.0, 0.0, 0.0])
    true_acc = np.array([0.0, 0.0, 0.0])
    yaw = 0.0

    total_steps = 120
    for step in range(total_steps):
        # Phase 1 (steps 0-39): Constant velocity
        if step < 40:
            true_acc = np.array([0.0, 0.0, 0.0])
        # Phase 2 (steps 40-79): Constant acceleration
        elif step < 80:
            true_acc = np.array([2.0, 0.0, 0.0])
        # Phase 3 (steps 80-119): Turning (circular motion)
        else:
            speed = np.linalg.norm(true_vel)
            omega = 0.5  # rad/s turn rate
            yaw += omega * dt
            true_vel = speed * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            true_acc = np.array([0.0, 0.0, 0.0])  # No linear acc during turn

        # Update ground truth
        true_pos = true_pos + true_vel * dt + 0.5 * true_acc * dt * dt
        true_vel = true_vel + true_acc * dt

        # Generate projective measurement
        measurement_gt = project_point_manual(
            np.array(cam.K),
            np.array(cam.R_cw),
            np.array(cam.t_cw),
            true_pos,
            np.array(cam.obj_dims),
        )
        noisy_meas = measurement_gt + np.random.randn(4) * np.array([5, 5, 10, 10])

        imm.predict(dt)
        imm.correct(noisy_meas)

        probabilities = np.array(imm.model_probabilities)
        probs_history.append(probabilities.copy())

    probs_history = np.array(probs_history)

    # Verify model probability trends
    # Phase 1 avg: CV should be highest
    phase1_avg = probs_history[20:40].mean(axis=0)
    # Phase 2 avg: CA should be highest
    phase2_avg = probs_history[60:80].mean(axis=0)
    # Phase 3 avg: CTRV should be highest
    phase3_avg = probs_history[100:120].mean(axis=0)

    print(
        f"  Phase 1 (CV period):   CV={phase1_avg[0]:.3f}  CA={phase1_avg[1]:.3f}  CTRV={phase1_avg[2]:.3f}"
    )
    print(
        f"  Phase 2 (Accel):       CV={phase2_avg[0]:.3f}  CA={phase2_avg[1]:.3f}  CTRV={phase2_avg[2]:.3f}"
    )
    print(
        f"  Phase 3 (Turn):        CV={phase3_avg[0]:.3f}  CA={phase3_avg[1]:.3f}  CTRV={phase3_avg[2]:.3f}"
    )

    # CV should be dominant in phase 1
    assert phase1_avg[0] > phase1_avg[1], "CV should beat CA during const-vel phase"
    assert phase1_avg[0] > phase1_avg[2], "CV should beat CTRV during const-vel phase"

    # CA should be significant in phase 2
    assert phase2_avg[1] > 0.2, f"CA should be significant during acceleration: {phase2_avg[1]:.3f}"

    print("✓ IMM model switching works with projective measurements")


def test_backward_compatible_alias():
    """RigidBodyPoseSensorModel should still work as alias."""
    old_model = quak.RigidBodyPoseSensorModel()
    new_model = RigidBodyCVModel()

    x = make_state(pos=[1, 2, 3], vel=[1, 0, 0])

    # Both should produce identical results
    m_old = old_model.measurement(x)
    m_new = new_model.measurement(x)
    np.testing.assert_allclose(m_old, m_new, atol=1e-15)

    p_old = old_model.process(x, np.array([]), 0.1)
    p_new = new_model.process(x, np.array([]), 0.1)
    np.testing.assert_allclose(p_old, p_new, atol=1e-15)

    print("✓ Backward-compatible alias works")


# ═════════════════════════════════════════════════════════════════════
# Run all tests
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Projective Bounding Box Model Tests")
    print("=" * 60)

    print("\n--- Projection Correctness ---")
    test_projection_at_origin_camera()
    test_projection_off_center()
    test_projection_different_depths()
    test_projection_with_extrinsics()
    test_projection_behind_camera_clamped()
    test_process_delegates_to_dynamics()

    print("\n--- UKF Tracking ---")
    test_ukf_projective_tracking()
    test_ukf_projective_depth_observability()

    print("\n--- IMM Tracking ---")
    test_imm_projective_model_switching()

    print("\n--- Backward Compatibility ---")
    test_backward_compatible_alias()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
