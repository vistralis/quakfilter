# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""quak — QUAternion C++ Kalman filter.

C++ core classes are available via ``quak._core``. This package re-exports
them alongside Pythonic wrappers for UKF and IMMEstimator.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("quakfilter")
except PackageNotFoundError:
    __version__ = "unknown"

# ── C++ extension (models, manifolds, sensors) ──────────────────────────
from quak._core import (  # noqa: F401
    # Sensors
    CameraParams,
    # Manifolds
    Euclidean,
    Manifold,
    # Models
    MeasurementModel,
    # Noise
    NoiseModel,
    PoseSensor,
    ProjectiveBBoxModel,
    ProjectiveBBoxSensor,
    Quaternion,
    RigidBodyCAModel,
    RigidBodyCTRVModel,
    RigidBodyCVModel,
    RigidBodyHelicalModel,
    RigidBodyPoseSensorModel,
    RigidBodySingerModel,
    # Temporal types
    StateEstimate,
    SystemModel,
    TimeMode,
    UKFModel,
    # Results
    UKFPredictionResult,
    VelocitySensor,
)
from quak.imm import IMMEstimator  # noqa: F401

# ── Python wrappers (preferred API) ─────────────────────────────────────
from quak.time_conversions import (  # noqa: F401
    from_epoch_seconds,
    to_duration_seconds,
    to_epoch_seconds,
)
from quak.track_pool import TrackPool  # noqa: F401
from quak.ukf import UKF  # noqa: F401

__all__ = [
    # Wrappers
    "UKF",
    "IMMEstimator",
    "TrackPool",
    # Manifolds
    "Manifold",
    "Euclidean",
    "Quaternion",
    # Models
    "UKFModel",
    "SystemModel",
    "MeasurementModel",
    "RigidBodyCVModel",
    "RigidBodyPoseSensorModel",
    "RigidBodyHelicalModel",
    "RigidBodyCTRVModel",
    "RigidBodyCAModel",
    "RigidBodySingerModel",
    # Sensors
    "PoseSensor",
    "VelocitySensor",
    "ProjectiveBBoxModel",
    "ProjectiveBBoxSensor",
    "CameraParams",
    "NoiseModel",
    # Results
    "UKFPredictionResult",
    # Temporal types
    "StateEstimate",
    "TimeMode",
    # Time conversions
    "to_epoch_seconds",
    "from_epoch_seconds",
    "to_duration_seconds",
]
