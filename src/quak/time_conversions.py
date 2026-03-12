# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time conversion utilities for quakfilter.

Converts between common Python time representations (float, datetime, numpy.datetime64)
and the epoch-seconds format used internally by the C++ temporal layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def to_epoch_seconds(t: float | datetime | np.datetime64) -> float:
    """Convert a time value to epoch seconds (float).

    Args:
        t: Time as float (epoch seconds, passed through unchanged),
           datetime (converted via timestamp()), or
           numpy.datetime64 (converted via epoch nanoseconds).

    Returns:
        Epoch seconds as float.

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    if isinstance(t, np.datetime64):
        return t.astype("datetime64[ns]").astype(np.int64) / 1e9
    raise TypeError(f"Unsupported time type: {type(t).__name__}")


def from_epoch_seconds(seconds: float) -> np.datetime64:
    """Convert epoch seconds to numpy.datetime64.

    Args:
        seconds: Epoch seconds as float.

    Returns:
        numpy.datetime64 with nanosecond resolution.
    """
    return np.datetime64(int(seconds * 1e9), "ns")


def to_duration_seconds(d: float | timedelta | np.timedelta64) -> float:
    """Convert a duration value to seconds (float).

    Args:
        d: Duration as float (seconds, passed through unchanged),
           timedelta (converted via total_seconds()), or
           numpy.timedelta64 (converted via nanoseconds).

    Returns:
        Duration in seconds as float.

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(d, (int, float)):
        return float(d)
    if isinstance(d, timedelta):
        return d.total_seconds()
    if isinstance(d, np.timedelta64):
        return d / np.timedelta64(1, "s")
    raise TypeError(f"Unsupported duration type: {type(d).__name__}")
