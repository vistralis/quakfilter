# Vistralis C++ Coding Style Guide

*Adapted from [OpenCV Coding Style Guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide), tailored for robotics perception and tracking.*

---

## 1. General Principles

- **C++17** minimum. Use modern features: `auto`, structured bindings, `constexpr`, `[[nodiscard]]`, `std::optional`.
- Cross-platform. Avoid platform-specific constructs in headers.
- No external GPL/LGPL dependencies in core libraries.
- Minimize patches — don't reformat unrelated code in the same commit.
- English only (ASCII) in code, comments, and string literals.

---

## 2. File Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Source files | `snake_case.cpp` | `projective_model.cpp` |
| Header files | `snake_case.hpp` | `projective_model.hpp` |
| Test files | `test_snake_case.py` / `test_snake_case.cpp` | `test_projective.py` |
| Benchmark files | `bench_snake_case.py` | `bench_kalman_gain.py` |

- Headers use `#pragma once` (not include guards).
- Each file starts with a license header (when applicable).
- Line limit: **100 characters**.
- Indentation: **4 spaces** (no tabs).

---

## 3. Naming Conventions

### C++

| Element | Style | Example |
|---------|-------|---------|
| Classes / Structs | `PascalCase` | `IMMEstimator`, `UKFModel` |
| Methods / Functions | `camelCase` | `predict()`, `getState()`, `applyCorrection()` |
| Member variables | `mCamelCase` | `mStateDim`, `mModelProbabilities` |
| Local variables | `camelCase` | `stateDim`, `tangentOffset` |
| Constants | `kPascalCase` | `kSingularValueFloor`, `kDefaultGamma` |
| Enum values | `kPascalCase` | `kConstantVelocity`, `kCTRV` |
| Macros | `UPPER_SNAKE_CASE` | `UKF_CPP_VERSION` |
| Namespaces | `snake_case` | `quak`, `vistralis::tracking` |
| Template params | `PascalCase` | `typename Scalar`, `int Dim` |

### Naming Rules

- **No abbreviations.** Write `measurement`, not `meas`. Write `stateSize`, not `stateSz`.
- **Accepted short forms:** `Num`, `Dim`, `z` (mathematical), `Q`, `R`, `P`, `S` (Kalman letter names).
- **Proper English order:** `mRawMeasurementDim`, not `mMeasRawDim`.
- **Loop variables:** Use descriptive names — `manifold`, not `m`.
- **Be consistent:** If one variable is `mMeasurementSigmas`, don't use `mMeasResiduals` elsewhere.

### Python

| Element | Style | Example |
|---------|-------|---------|
| Functions / methods | `snake_case` | `predict()`, `get_state()` |
| Classes | `PascalCase` | `IMMEstimator`, `TrackManager` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_PROB_FLOOR` |
| Variables | `snake_case` | `state_dim`, `model_count` |
| Private members | `_leading_underscore` | `_state`, `_covariance` |

### Method Naming Patterns

```cpp
// Getters/setters: get/set prefix
MatrixXd getStateErrorCovariance() const;
void setProcessNoise(const MatrixXd& Q);

// Actions: verbObject
PredictionResult predictState(double dt);
void correctWithMeasurement(const VectorXd& z);
void updateModelProbabilities();

// Queries: is/has/can prefix
bool isPositiveDefinite() const;
bool hasConverged() const;

// Factory: create prefix
static UKF createFromModel(std::shared_ptr<UKFModel> model);
```

---

## 4. Code Layout

### Braces

Allman style — opening brace on its own line:

```cpp
if (condition)
{
    doSomething();
}
else
{
    doOther();
}

for (int i = 0; i < n; ++i)
{
    process(i);
}

class MyClass
{
public:
    void method()
    {
        // ...
    }
};
```

### Class Layout Order

```cpp
class Estimator {
public:
    // 1. Type aliases
    using StateVector = Eigen::VectorXd;

    // 2. Constructors / Destructor
    Estimator(int dim);
    ~Estimator() = default;

    // 3. Public methods (most important first)
    PredictionResult predict(double dt);
    void correct(const VectorXd& measurement);

    // 4. Getters / Setters
    VectorXd getState() const;
    MatrixXd getCovariance() const;

protected:
    // 5. Protected methods
    void computeSigmaPoints();

private:
    // 6. Private methods
    void regularizeCovariance();

    // 7. Member variables (always last)
    int mStateDim;
    VectorXd mState;
    MatrixXd mCovariance;
};
```

### Namespaces

```cpp
namespace vistralis {
namespace tracking {

class Tracker { ... };

}  // namespace tracking
}  // namespace vistralis
```

No indentation inside namespace blocks.

---

## 5. Eigen & Numerical Computing

### Mandatory Practices

```cpp
// ✅ LDLT for solving with SPD matrices (with fallback)
auto ldlt = S.ldlt();
if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
    K = Pxz * ldlt.solve(MatrixXd::Identity(n, n));
} else {
    // SVD fallback for degenerate cases
}

// ✅ Log-space for likelihoods (never compute raw determinant)
double logDetS = ldlt.vectorD().array().abs().log().sum();
double logLikelihood = -0.5 * (measDim * log(2*M_PI) + logDetS + mahal);

// ✅ Symmetrize after covariance update
P = P - K * S * K.transpose();
P = (P + P.transpose()) * 0.5;

// ✅ Eigenvalue floor to prevent collapse
SelfAdjointEigenSolver<MatrixXd> es(P);
if (es.eigenvalues().minCoeff() < kEigenvalueFloor) {
    P += MatrixXd::Identity(n, n) * (kCovarianceRegularization - es.eigenvalues().minCoeff());
}
```

### Forbidden Patterns

```cpp
// ❌ Raw inverse — numerically unstable
MatrixXd K = Pxz * Pvv.inverse();

// ❌ Raw determinant — overflows/underflows
double logDet = std::log(S.determinant());

// ❌ Unchecked decomposition
MatrixXd L = P.llt().matrixL();  // No .info() check!

// ❌ Exp without log-sum-exp guard
double p = std::exp(logLikelihood);  // overflow risk

// ❌ Heap allocation in hot loops
std::vector<VectorXd> sigmas;  // inside predict/correct
for (...) sigmas.push_back(...);
```

### Constants

Numerical thresholds must be named constants, not magic numbers:

```cpp
namespace vistralis::constants {
    constexpr double kSingularValueFloor = 1e-6;
    constexpr double kCovarianceRegularization = 1e-9;
    constexpr double kEigenvalueFloor = 1e-10;
    constexpr double kSmallAngleThreshold = 1e-12;
    constexpr double kSvdRelativeTolerance = 1e-8;
}
```

---

## 6. Function / Class Interface Design

### Argument Order

```
output_or_inout, input, parameters, flags
```

```cpp
// Input-heavy: const ref
void correct(const VectorXd& measurement);

// Output via return (preferred for small types)
PredictionResult predict(double dt);

// Output via reference (for large types to avoid allocation)
void computeCovariance(MatrixXd& result);

// Default arguments for optional params
PredictionResult predict(double dt, const VectorXd& control = VectorXd());
```

### Error Handling

- **Throw exceptions** for programming errors (null pointers, dimension mismatches).
- **Return status** for runtime conditions (divergence, track loss).
- **Never use return codes** for critical failures.

```cpp
// ✅ Exception for programming error
if (measurement.size() != mTangentMeasurementDim) {
    throw std::invalid_argument("Measurement dimension mismatch");
}

// ✅ NaN guard (debug builds)
#ifndef NDEBUG
if (!mState.allFinite()) {
    std::cerr << "NaN detected in filter state" << std::endl;
}
#endif
```

---

## 7. Documentation

- Use `///` Doxygen-style comments for public API:

```cpp
/// Predict the state forward by dt seconds.
/// @param dt Time step in seconds (must be > 0).
/// @param control Optional control input vector.
/// @return PredictionResult containing predicted measurement mean and innovation covariance.
PredictionResult predict(double dt, const VectorXd& control = VectorXd());
```

- Inline comments for non-obvious logic only. Don't comment what the code already says.

---

## 8. Testing

- Tests use `pytest` (Python) or `gtest` (C++).
- Test names: `test_<what>_<expected_behavior>`.
- One assertion per test when practical.
- Use numerical tolerances, not exact equality, for floating-point comparisons.

```python
def test_predict_returns_finite_state():
    ...

def test_imm_model_probability_evolution():
    ...
```

---

## 9. Tooling

| Tool | Purpose | Config |
|------|---------|--------|
| `clang-format` | C++ formatting | `.clang-format` |
| `ruff` | Python formatting + linting | `pyproject.toml [tool.ruff]` |
| NCAssessment | Numerical stability audit | `.agents/workflows/ncassessment.md` |

Run before committing:
```bash
clang-format -i src/*.hpp src/*.cpp
ruff format tests/ examples/
ruff check tests/ examples/
```

---

## 10. Python Bindings (nanobind)

- Bind C++ classes with `snake_case` method names to match Python convention.
- Use `nb::arg("name")` for all parameters.
- Provide docstrings via `.doc()`.

```cpp
nb::class_<UKF>(m, "UKF")
    .def("predict", &UKF::predict, nb::arg("dt"), nb::arg("control") = VectorXd(),
         "Predict state forward by dt seconds")
    .def("correct", &UKF::correct, nb::arg("measurement"),
         "Correct state with measurement");
```

Note: C++ uses `camelCase` methods, Python bindings expose them as `snake_case`.
