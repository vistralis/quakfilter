# Data Association for Multi-Object Tracking

> **Status**: Design document — theoretical framework for quak v0.2.0.
> No implementation yet. All content is original.

This document covers the mathematical foundations, algorithmic choices, and
architectural recommendations for adding data association to quak.  The goal is
a **composable** association layer: users pick a cost function, a gating
strategy, and an assignment solver, then wire them together.

---

## 1  The Assignment Problem

At every time step a tracker holds **N** existing tracks and receives **M** new
detections.  The task is to decide which detection (if any) should update which
track.

### 1.1  Bipartite Graph Formulation

Model the problem as a **weighted bipartite graph** G = (T ∪ D, E):

- **T = {t₁ … tₙ}** — set of tracks (left vertices)
- **D = {d₁ … dₘ}** — set of detections (right vertices)
- **E** — edges whose weights are the **cost** (or distance) of assigning dⱼ to tᵢ

An **assignment** is a set of edges A ⊆ E such that each vertex appears at most
once.  In MOT this means:

- Each track is assigned to **at most one** detection (injective).
- Each detection is assigned to **at most one** track (injective).
- Some tracks may be **unassigned** (prediction only — missed detection).
- Some detections may be **unassigned** (new object or clutter).

### 1.2  Cost Matrix

Collect all pairwise costs into the **cost matrix** C ∈ ℝ^{M×N}:

$$C_{ij} = \text{cost of assigning detection } d_i \text{ to track } t_j$$

The optimal assignment minimises (or maximises) the total cost:

$$A^* = \arg\min_{A} \sum_{(i,j) \in A} C_{ij}$$

subject to the injection constraints.

---

## 2  Gating

Before solving the assignment, **gating** prunes impossible associations to
reduce computation and prevent absurd matches.

### 2.1  Mahalanobis Gate

Given a track t with predicted measurement mean ẑ and innovation covariance S,
the **Mahalanobis distance** of detection z is:

$$d^2_M(z, t) = (z - \hat{z})^\top S^{-1} (z - \hat{z})$$

Under the linear-Gaussian model, d²_M follows a χ² distribution with m degrees
of freedom (m = measurement dimension).  Standard gates:

| Confidence | χ²(6) threshold | χ²(7) threshold |
|------------|-----------------|-----------------|
| 95%        | 12.59           | 14.07           |
| 99%        | 16.81           | 18.48           |
| 99.5%      | 18.55           | 20.28           |

Entries with d²_M above the threshold are set to ∞ (or a large bound value) in
the cost matrix, effectively removing them from the assignment.

**Why Mahalanobis and not Euclidean?**  Euclidean distance ignores the filter's
uncertainty.  A track with high velocity uncertainty (large P) should accept
detections from a wider spatial region than a well-localised track.  Mahalanobis
naturally scales by uncertainty.

### 2.2  IoU Gate (Image Plane)

For camera-based tracking, the Intersection over Union of bounding boxes is a
natural measure:

$$\text{IoU}(d, t) = \frac{|B_d \cap B_t|}{|B_d \cup B_t|}$$

Gating: discard pairs with IoU < τ (typically τ = 0.1–0.3).

IoU is often used as a **cost** directly (as 1 − IoU) rather than just for
gating.

### 2.3  Connected Component Decomposition

After gating, the bipartite graph is often **sparse** — many pairs are
disconnected.  The standard optimisation: compute the connected components of
the gated graph, then solve independent assignment subproblems for each
component.  This reduces the effective problem size dramatically in typical
scenarios (most tracks have ≤ 3 candidate detections nearby).

---

## 3  Assignment Algorithms

### 3.1  Hungarian Algorithm (Kuhn-Munkres)

The **Hungarian algorithm** solves the linear assignment problem optimally in
**O(n³)** time, where n = max(M, N).

#### 3.1.1  Intuition

The algorithm operates on the cost matrix through a sequence of row and column
reductions:

1. **Row reduction**: subtract the minimum of each row from all entries in that
   row.  Now each row has at least one zero.
2. **Column reduction**: subtract the minimum of each column.  Now each column
   also has at least one zero.
3. **Zero covering**: find the minimum number of lines (rows + columns) needed
   to cover all zeros.  If this equals n, an optimal assignment through zeros
   exists.
4. **Augmentation**: if fewer than n lines suffice, adjust the matrix to create
   more zeros.  Find the minimum uncovered value, subtract it from all uncovered
   entries, and add it to all doubly-covered entries.  Repeat from step 3.

The key insight: subtracting a constant from a row (or column) does not change
the optimal assignment, only the total cost.

#### 3.1.2  Properties

- **Optimal**: always finds the minimum-cost assignment.
- **Complexity**: O(n³) time, O(n²) space.
- **Square matrix**: the standard algorithm requires a square matrix.  Pad the
  smaller dimension with dummy rows/columns of cost zero (or bound_value for
  maximisation).
- **Deterministic**: the same cost matrix always produces the same assignment.

#### 3.1.3  Practical Considerations

- For very small subproblems (1×1, 2×2) bypass the full algorithm with direct
  comparison — the overhead of initialising the algorithm exceeds the benefit.
- For sparse cost matrices, connected component decomposition yields independent
  subproblems (often 1×1 or 2×3), making the effective cost sub-linear.

### 3.2  Global Nearest Neighbour (GNN)

The simplest meaningful baseline:

1. Compute the cost matrix C.
2. Find the global minimum entry (i*, j*).
3. Assign d_{i*} to t_{j*}.
4. Remove row i* and column j* from the matrix.
5. Repeat until no valid (gated) entries remain.

**Complexity**: O(MN · min(M,N)) — dominated by the iterative minimum search.

**Pros**: trivial to implement, fast for small problems.
**Cons**: greedy, not globally optimal.  A single bad greedy choice can cascade
into several incorrect assignments.

### 3.3  Auction Algorithm

An alternative to Hungarian for large-scale problems.  Objects "bid" for tracks
through iterative price adjustments:

1. Assign each detection to its **preferred track** (lowest cost minus current
   price).
2. If two detections compete for the same track, raise the track's price by the
   difference to the second-best bid plus ε.
3. Repeat until all assignments are stable.

**Complexity**: O(Mn · log(nC/ε)) where C is the maximum cost value.

**Pros**: naturally parallelisable, handles large sparse problems well.
**Cons**: approximate (depends on ε), more complex to implement, convergence
can be slow on pathological inputs.

### 3.4  Joint Probabilistic Data Association Filter (JPDAF)

JPDAF does not produce a **binary** assignment.  Instead, it computes a
**probability that each detection originated from each track**, then updates
each track with a **weighted combination** of all detections.

#### 3.4.1  Association Probabilities

For track tₖ and detection dⱼ, define the **likelihood**:

$$l_{kj} = \mathcal{N}(z_j;\; \hat{z}_k, S_k)$$

where ẑₖ and Sₖ are the predicted measurement mean and innovation covariance
of track k.

The association probability β_{kj} (probability that detection j originated
from track k) is computed by marginalising over all feasible **joint**
association events:

$$\beta_{kj} = \sum_{\theta \in \Theta : \theta(j)=k} P(\theta | Z)$$

where Θ is the set of all feasible (injective) association hypotheses and Z is
the set of all detections.

#### 3.4.2  Combined Innovation

Instead of choosing one detection, JPDAF forms a **weighted innovation**:

$$\nu_k = \sum_{j=1}^{M} \beta_{kj} (z_j - \hat{z}_k)$$

The Kalman update then uses νₖ as the measurement residual, with an inflated
innovation covariance:

$$\tilde{S}_k = S_k + \sum_{j=1}^{M} \beta_{kj}
   [(z_j - \hat{z}_k)(z_j - \hat{z}_k)^\top] - \nu_k \nu_k^\top$$

#### 3.4.3  Properties

- **Soft assignment**: each track may incorporate information from multiple
  detections, weighted by probability.
- **Best for ambiguity**: when two tracks are close (occluding), JPDAF naturally
  spreads association weight rather than forcing a binary choice.
- **Complexity**: computing the marginal association probabilities involves
  summing over all joint hypotheses — exponential in the worst case, but
  feasible with gating (typically ≤ 3–5 detections per track).
- **Limitation**: tends to merge tracks in prolonged close proximity because
  both tracks receive similar weighted innovations.

### 3.5  Probabilistic Multi-Hypothesis Tracker (PMHT)

PMHT uses the EM algorithm with an **independence assumption**: it treats each
measurement as independently associated (not injective).

- **E-step**: compute association weights w_{kj} ∝ π_k · N(z_j; ẑ_k, S_k)
  where π_k are prior mixing weights.
- **M-step**: update each track with the weighted measurements.

The critical difference from JPDAF: PMHT allows multiple detections to be
assigned to the same track (many-to-one), which violates physical reality but
simplifies computation.

### 3.6  Summary Comparison

| Algorithm  | Assignment | Optimality  | Complexity      | Handles Ambiguity |
|------------|-----------|-------------|-----------------|-------------------|
| GNN        | Binary    | Greedy      | O(MN·min(M,N))  | No                |
| Hungarian  | Binary    | **Optimal** | O(n³)           | No                |
| Auction    | Binary    | ε-optimal   | O(Mn·log(nC/ε)) | No                |
| JPDAF      | Soft      | Bayesian    | O(exp) w/ gating| **Yes**           |
| PMHT       | Soft      | EM          | O(MN)           | Yes (many-to-one) |

**Recommendation for quak**: implement **Hungarian with Mahalanobis gating** as
the default (covers 90%+ of use cases), with **JPDAF** as an optional mode for
high-ambiguity scenarios.

---

## 4  Cost Function Composition

The cost matrix C is the central data structure.  Its quality determines the
quality of the assignment.  A composable cost function architecture lets users
combine spatial, appearance, and semantic cues.

### 4.1  Spatial Cost

Derived from the filter's own innovation statistics:

- **Mahalanobis distance**: d²_M(z, t) — uses innovation covariance, accounts
  for both prediction uncertainty and measurement noise.
- **IoU cost**: 1 − IoU(bbox_z, bbox_t) — for bounding-box–based tracking.
- **Euclidean distance**: ‖pos_z − pos_t‖₂ — simple fallback when covariance
  is unavailable.

### 4.2  Appearance Cost

Based on visual similarity between detection and track features:

- **Cosine distance**: 1 − (f_z · f_t) / (‖f_z‖ · ‖f_t‖)
- **Feature gallery matching**: compare the detection feature against the
  track's **entire feature history** (see §5), take the maximum or
  temporally-weighted average similarity.

### 4.3  Classification Cost

Penalise associations between detections and tracks of different classes:

- **Bhattacharyya distance**: d_B(p, q) = −ln Σ√(pᵢ · qᵢ) — a proper metric
  on probability distributions.
- **Binary gate**: if argmax(p) ≠ argmax(q), set cost to ∞.

### 4.4  Composite Cost

The final cost is a weighted combination:

$$C_{ij} = w_s \cdot c_{\text{spatial}}(d_i, t_j)
         + w_a \cdot c_{\text{appearance}}(d_i, t_j)
         + w_c \cdot c_{\text{class}}(d_i, t_j)$$

Weights (w_s, w_a, w_c) are user-configurable.  When a component is unavailable
(e.g., no appearance features), it is excluded and weights are renormalised.

**Implementation note**: each cost component should normalise its output to a
common range (e.g., [0, 1]) before composition, otherwise the weights become
scale-dependent and difficult to tune.

---

## 5  Visual Feature Gallery

A **feature gallery** maintains a history of visual embeddings for each track,
enabling robust appearance matching even when a single observation is noisy or
partially occluded.

### 5.1  Multi-Channel Design

Different visual encoders produce features of different dimensions and capture
different aspects of appearance:

| Channel   | Encoder    | Dim  | Captures                     |
|-----------|------------|------|------------------------------|
| `body`    | PersonViT  | 768  | Full-body appearance, pose   |
| `face`    | FaceViT    | 512  | Facial identity              |
| `vehicle` | VehicleNet | 256  | Vehicle type, colour, plate  |
| `lidar`   | PointNet   | 128  | 3D shape signature           |

Each channel maintains an **independent** history ring buffer.  Channels are
created on demand — a track that has never received face features simply has no
`face` channel.

### 5.2  Similarity Computation

For a detection feature f and a track's feature history matrix
F = [f₁ | f₂ | … | fₖ] (each column is a historical feature):

**Batch cosine similarity** (vectorised):

$$s = \frac{F^\top f}{\|F_{\text{col}}\| \cdot \|f\|} \in \mathbb{R}^k$$

This is a single matrix-vector multiply plus element-wise normalisation —
highly efficient in Eigen.

### 5.3  Temporal Weighting

Recent observations should be weighted more heavily than old ones.  Given k
features ordered oldest-to-newest, define temporal weights:

$$w_i = \exp(-\lambda \cdot (k - i))$$

where λ is the decay rate.  The **gallery similarity** is then:

$$\text{sim}(f, F) = \frac{\sum_i w_i \cdot s_i}{\sum_i w_i}$$

Setting λ = 0 gives uniform weighting (mean similarity);
λ → ∞ uses only the most recent feature (no history).

Alternatively, use `max(s)` — the **best historical match** — which is more
robust to temporary appearance changes (e.g., brief occlusion by another object).

### 5.4  Gallery Size Management

- **Fixed-size ring buffer** (e.g., 100 entries per channel) with FIFO eviction.
- **Feature quality filtering**: only store features with detection confidence
  above a threshold (discard blurry/occluded crops).
- **Subsampling**: if the tracker runs at high frame rate (e.g., 30 fps), store
  features only every N frames to maintain temporal diversity.

---

## 6  Classification Fusion

When a detection is associated with a track, the track's class distribution
should be updated to incorporate the new evidence.

### 6.1  Bayesian Update

Let p(c) be the track's current class distribution (prior) and l(c|z) be the
detection's class likelihood.  The posterior is:

$$p(c|z) = \frac{l(c|z) \cdot p(c)}{\sum_{c'} l(c'|z) \cdot p(c')}$$

In practice, work in **log-space**:

$$\log p(c|z) = \log l(c|z) + \log p(c) - \log Z$$

where Z is the normalisation constant, computed via the log-sum-exp trick:

$$\log Z = \text{logsumexp}_c [\log l(c|z) + \log p(c)]$$

### 6.2  Sigmoid vs Softmax Inputs

Modern detectors may produce:

- **Softmax outputs**: probabilities sum to 1.  Standard closed-world
  assumption — the object *must* be one of the known classes.
- **Sigmoid outputs**: each class probability is independent.  Open-world — the
  object may be none, one, or multiple of the known classes.  The sum may be
  less than, equal to, or greater than 1.

For sigmoid detectors, define an **implicit unknown class** with probability:

$$p(\text{unknown}) = \max(0,\; 1 - \sum_c p(c))$$

The Bayesian fusion then operates over the extended (N+1)-class simplex.  If
Σp > 1 (overconfident sigmoid), renormalise before fusion.

### 6.3  Forgetting Factor

Without decay, the accumulated posterior from hundreds of observations becomes
extremely concentrated.  If an object genuinely changes class (e.g., a
partially visible object resolves from "unknown" to "person"), the prior resists
change.

Apply a **forgetting factor** α ∈ [0, 1]:

$$\log p'(c) = (1 - \alpha) \cdot \log p(c) + \alpha \cdot \log u(c)$$

where u(c) = 1/N is the uniform distribution.  This flattens the prior before
each update, controlling how quickly old evidence is forgotten:

- α = 0: full memory (standard Bayes)
- α = 0.01: slow forgetting (suitable for static objects)
- α = 0.1: fast forgetting (suitable for ambiguous, dynamic scenes)

### 6.4  Similarity Metrics for Classification

| Metric               | Formula                                        | Range | Proper Metric? |
|----------------------|------------------------------------------------|-------|---------------|
| L2 (Euclidean)       | ‖p − q‖₂                                      | [0, √2] | Yes         |
| Bhattacharyya coeff. | Σ √(pᵢqᵢ)                                     | [0, 1]  | No (coeff)  |
| Bhattacharyya dist.  | −ln Σ √(pᵢqᵢ)                                 | [0, ∞)  | Yes         |
| Hellinger distance   | √(1 − Σ √(pᵢqᵢ))                              | [0, 1]  | Yes         |
| Jensen-Shannon div.  | ½ KL(p‖m) + ½ KL(q‖m), m=(p+q)/2              | [0, 1]  | Yes (√JSD)  |

**Recommendation**: Bhattacharyya coefficient for similarity scoring (bounded,
efficient), Hellinger distance when a proper metric is needed for gating.

---

## 7  Tracked Object Model

A tracked object bundles the filter state with rich metadata that does not
participate in the predict/correct cycle.

### 7.1  Separation of Concerns

| Layer | Owns | Updated by |
|-------|------|-----------|
| Filter state (UKF/IMM) | position, velocity, acceleration, orientation, angular velocity, covariance | predict(), correct() |
| Object metadata | class distribution, feature gallery, extent (L×W×H), age, hit/miss counts | association + lifecycle logic |

The filter state lives in the **TrackPool** (C++, contiguous memory, OpenMP).
Object metadata lives in a parallel registry keyed by the same UUID.  The state
is accessed from the metadata layer via **zero-copy reference** — not copied,
not serialised.

### 7.2  Extent (Shape)

The physical size of the object is **not part of the filter state** for
rigid-body models (the filter tracks position and orientation, not size).
Extent is stored as metadata:

- **3D bounding box**: length, width, height — fixed at detection time, or
  smoothed with an exponential moving average.
- **2D bounding box**: computed on-the-fly by projecting the 3D state + extent
  through the camera model (already supported by ProjectiveBBoxSensor).

### 7.3  Lifecycle

| State       | Condition                    | Transitions to |
|-------------|------------------------------|---------------|
| **Tentative** | Newly created from unmatched detection | Confirmed / Deleted |
| **Confirmed** | Hit count ≥ threshold (e.g., 3 of last 5 frames) | Suspended / Deleted |
| **Suspended** | Miss count ≥ threshold (e.g., 5 consecutive misses) | Confirmed / Deleted |
| **Deleted**   | Miss count ≥ max age, or explicit removal | — |

TrackPool already supports active/suspended partitioning.  The lifecycle
manager is a thin policy layer on top.

---

## 8  Multi-Stage Association

For challenging scenarios, a **cascade** of association stages with different
cost functions at each stage can outperform a single-stage approach.  This
is the pattern used by DeepSORT and ByteTrack:

### 8.1  Two-Stage Example (ByteTrack-style)

1. **Stage 1 — High confidence**: associate high-confidence detections (score >
   τ_high) with tracks using IoU cost + Hungarian.
2. **Stage 2 — Low confidence**: associate remaining low-confidence detections
   with **unmatched tracks only**, using IoU cost + Hungarian.

This recovers occluded objects that produce low-confidence detections, which
would be discarded in a single-stage approach.

### 8.2  Three-Stage Example (DeepSORT-style)

1. **Appearance matching**: confirmed tracks matched to detections via
   appearance (cosine distance) + Mahalanobis gate + Hungarian.
2. **IoU matching**: unmatched tracks matched to remaining detections via IoU
   cost + Hungarian.
3. **Tentative matching**: tentative tracks matched to remaining detections
   via IoU cost + Hungarian.

---

## 9  Implementation Recommendations

### 9.1  Architecture

The association layer should be structured as three independent, composable
components:

1. **CostFunction** — computes M×N cost matrix from detections and tracks.
   Abstract base with concrete implementations for Mahalanobis, IoU, appearance,
   classification, and composite costs.

2. **AssignmentSolver** — takes a cost matrix and returns assignments.  Abstract
   base with implementations for Hungarian, GNN, and JPDAF.

3. **LifecyclePolicy** — manages track creation, confirmation, suspension, and
   deletion based on hit/miss counts and age.

All three components should be implemented in **C++** with nanobind Python
bindings.  The trampoline pattern (nanobind `Trampoline` / pybind11
`py::trampoline`) allows users to subclass CostFunction in Python for custom
cost components that require non-C++ dependencies (e.g., a PyTorch encoder).

### 9.2  Performance Targets

- **Hungarian**: < 1 ms for 100×100 cost matrix.
- **Mahalanobis batch**: compute M×N distances in a single Eigen operation,
  avoid per-pair function calls.
- **Feature gallery similarity**: single `mat.T * vec` operation per channel
  per track — no loops over history entries.
- **Connected component decomposition**: O(M+N+E) BFS, avoids solving full
  M×N Hungarian when the graph is sparse.

### 9.3  C++ Data Structures

- **Cost matrix**: `Eigen::MatrixXd` (M rows × N cols).  Avoid dynamic
  allocation per frame by pre-allocating for the expected max tracks/detections
  and using `conservativeResize` only when exceeded.
- **Feature gallery**: `std::deque<Eigen::VectorXd>` per channel with
  precomputed `Eigen::MatrixXd` (feature_dim × history) — updated
  incrementally on each `addFeature()` call.
- **Classification state**: `Eigen::VectorXd` in log-space.  Avoid
  raw `.inverse()` or `.determinant()` — exponentiate only for display.
- **Assignment result**: `std::vector<std::pair<size_t, size_t>>` for matched
  pairs, `std::vector<size_t>` for unmatched rows/columns.

### 9.4  API Surface (Preview)

The final API, exposed through nanobind, should look approximately like:

- `FeatureGallery.addFeature(channel, feature)` — add observation
- `FeatureGallery.similarity(channel, query, decay)` — weighted similarity
- `TrackedObject.features` — the gallery
- `TrackedObject.classProbabilities` — Bayesian posterior
- `TrackedObject.extent` — 3D size
- `CostFunction.__call__(detections, tracks) → MatrixXd` — cost matrix
- `AssignmentSolver.__call__(cost_matrix, threshold) → (matches, unmatched_det, unmatched_trk)`
- `MultiObjectTracker.step(detections, dt)` — full predict→associate→update cycle
