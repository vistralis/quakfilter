# UKF Reference

> **Source:** [`csrc/include/quak/ukf.hpp`](../csrc/include/quak/ukf.hpp)
> **Class:** `UKF`

## Overview

The `UKF` class implements the full Unscented Kalman Filter algorithm with manifold support, configurable matrix square root, and robust numerical safeguards. It operates through the generic `UKFModel` interface, meaning any model with any manifold layout can be plugged in.

## Algorithm

### Prediction

1. **Augment covariance:** `P_aug = P + Q`
2. **Matrix square root:** SVD (default) or LDLT, producing factor `S`
3. **Generate sigma points** (scheme-dependent, see below)
4. **Propagate:** `Yᵢ = f(χᵢ, u, Δt)` through the process model
5. **Compute mean:** weighted manifold mean of propagated sigma points
6. **Compute covariance:** weighted sum of outer products of residuals
7. **Symmetrize:** `P = (P + Pᵀ) / 2`

### Correction

1. **Project sigma points** to measurement space: `Zᵢ = h(Yᵢ)`
2. **Measurement mean** via manifold mean
3. **Innovation covariance:** `Pᵥᵥ = Pzz + R`
4. **Cross-correlation:** weighted sum of state-measurement outer products
5. **Kalman gain:** `K = Pxz · Pᵥᵥ⁻¹`
6. **State update:** `x = x ⊕ (K · innovation)`
7. **Covariance update:** `P = P − K · Pᵥᵥ · Kᵀ`
8. **Symmetrize + regularize** (restore positive-definiteness if needed)

## Sigma Point Schemes

### Symmetric (2n) — Default

- **Points:** 2n sigma points, no mean point
- **Scale:** `S *= √n` where n = tangent dimension (default 15)
- **Generation:** `χᵢ = x ⊕ S_col(i)` and `χᵢ₊ₙ = x ⊕ (−S_col(i))`
- **Weights:** Uniform `1/(2n)` for all sigma points
- **Advantage:** No negative weights → covariance stays positive-definite

### Merwe Scaled (2n+1) — `scaled_sigma_points=True`

- **Points:** 2n+1 sigma points, including the mean as zeroth point
- **Scale:** `γ = √(n + λ)` where `λ = α²(n + κ) − n`
- **Parameters:** `α = 0.001`, `β = 2.0`, `κ = 0.0` (defaults)
- **Weights:** Separate weights for mean (`Wm`) and covariance (`Wc`)
  - `Wm₀ = λ / (n + λ)`
  - `Wc₀ = λ / (n + λ) + (1 − α² + β)`
  - `Wmᵢ = Wcᵢ = 1 / (2(n + λ))` for i > 0
- **Advantage:** ~8% faster than symmetric in benchmarks; matches OpenCV UKF

## Key Design Decisions

### SVD vs LDLT

| Method | Pros | Cons |
|--------|------|------|
| SVD (`svd_decomposition=True`, default) | Handles semi-definite matrices, numerically robust | ~3× slower |
| LDLT (`svd_decomposition=False`) | Fast, cache-friendly | Requires strictly positive-definite; falls back to regularized LLT |

### Eigenvalue Regularization

After correction, if the minimum eigenvalue of `P` drops below `1e-10`, a diagonal shift is added:

```cpp
P += I · (1e-9 − λ_min)
```

This prevents covariance collapse during aggressive measurement updates.

## Python API

```python
ukf = quak.UKF(model, Q, R, P0, x0, svd_decomposition=True, scaled_sigma_points=False)

ukf.predict(dt, control=np.array([]))
ukf.correct(measurement)

# Multi-sensor correction
ukf.correct_with_sensor(z, sensor)

# Accessors
ukf.x                                # state (property)
ukf.P                                # error covariance (property)
ukf.get_state()                      # state (method)
ukf.get_state_error_covariance()     # P
ukf.get_process_noise_covariance()   # Q
ukf.get_measurement_noise_covariance()  # R
ukf.get_innovation_covariance()      # S (after correct())

# Mutators
ukf.set_state_and_error_covariance(x, P)
ukf.set_process_noise_covariance(Q)
ukf.set_measurement_noise_covariance(R)
```

## Characteristics

| Property | Value |
|----------|-------|
| Zero-allocation predict/correct | Yes (pre-allocated workspaces) |
| Parallelism | None (single filter; use TrackPool for batch) |
| Model coupling | Generic (`UKFModel` + `MeasurementModel` interfaces) |
| Dimensions | Dynamic (runtime) |
| Numerical safeguards | Symmetrization + eigenvalue regularization |
| Multi-sensor | Yes (`correct_with_sensor` overload) |
